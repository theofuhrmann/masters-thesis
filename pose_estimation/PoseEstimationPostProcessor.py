import os
import torch
import numpy as np
import numpy.core.multiarray as multiarray
from dotenv import load_dotenv

class PoseEstimationPostProcessor:
    def __init__(self, dataset_path, instruments_left_to_right):
        load_dotenv()
        self.dataset_path = dataset_path
        self.instruments = instruments_left_to_right
        # safe‐globals for torch.load
        torch.serialization.add_safe_globals([multiarray._reconstruct])
        _orig_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = patched_load

        # will hold raw and processed results
        self.raw = {}      # raw[artist][song] = list of frames × subjects
        self.processed = {}  # processed[artist][song] = {'keypoints':…, 'keypoint_scores':…}

    def load_results(self, avoid_artists=None):
        """Load each pose_estimation.pkl into self.raw."""
        avoid = set(avoid_artists or [])
        for artist in os.listdir(self.dataset_path):
            if artist in avoid: continue
            artist_dir = os.path.join(self.dataset_path, artist)
            if not os.path.isdir(artist_dir):
                continue
            self.raw[artist] = {}
            for song in os.listdir(artist_dir):
                song_dir = os.path.join(artist_dir, song)
                pkl_file = os.path.join(song_dir, 'pose_estimation.pkl')
                if not os.path.isfile(pkl_file):
                    continue
                with open(pkl_file, 'rb') as f:
                    frames = torch.load(f)
                self.raw[artist][song] = frames
                print(f"Loaded {len(frames)} frames for {artist}/{song}")

    def calculate_subject_centers(self, frames):
        """Return list of [x,y] for consistently detected subjects."""
        all_centers = []
        for fi, frame in enumerate(frames):
            for si, subj in enumerate(frame):
                x0,y0,x1,y1 = subj['bbox'][0]
                cx, cy = (x0+x1)/2, (y0+y1)/2
                all_centers.append((si, cx, cy))
        # cluster by proximity
        refs = []
        for si, cx, cy in all_centers:
            placed = False
            for r in refs:
                if np.hypot(cx-r['cx'], cy-r['cy']) < 100:
                    # update avg
                    r['cx'] = (r['cx']*r['count'] + cx)/(r['count']+1)
                    r['cy'] = (r['cy']*r['count'] + cy)/(r['count']+1)
                    r['count'] += 1
                    placed = True
                    break
            if not placed:
                refs.append({'cx':cx, 'cy':cy, 'count':1})
        # filter infrequent
        min_count = len(frames)*0.66
        centers = [(r['cx'], r['cy']) for r in refs if r['count']>min_count]
        return centers

    def reorder_and_map(self, frames):
        """
        1) pick bottom-3 by y
        2) reorder each frame to match
        3) map subject-idx→instrument by x left→right
        returns (consistent_dict, centers)
        """
        centers = self.calculate_subject_centers(frames)
        # pick bottom num_instruments
        centers.sort(key=lambda c: c[1], reverse=True)
        centers = centers[:len(self.instruments)]
        # map subject indices per frame
        reorganized = []
        for frame in frames:
            slot = [None]*len(centers)
            for subj in frame:
                x0,y0,x1,y1 = subj['bbox'][0]
                cx = (x0+x1)/2; cy = (y0+y1)/2
                # find nearest center
                dists = [np.hypot(cx-cx0, cy-cy0) for cx0,cy0 in centers]
                idx = int(np.argmin(dists))
                if dists[idx]<100:
                    slot[idx] = subj
            reorganized.append(slot)
        # assign instruments based on x‐order
        xs = [c[0] for c in centers]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        subj2inst = {si: self.instruments[pos] for pos,si in enumerate(order)}
        # build dict of per-instrument time lists
        out = {'keypoints':{}, 'keypoint_scores':{}}
        for si, inst in subj2inst.items():
            out['keypoints'][inst] = []
            out['keypoint_scores'][inst] = []
            for slot in reorganized:
                subj = slot[si]
                if subj:
                    out['keypoints'][inst].append(subj['keypoints'])
                    out['keypoint_scores'][inst].append(subj['keypoint_scores'])
                else:
                    out['keypoints'][inst].append(None)
                    out['keypoint_scores'][inst].append(None)
        return out, centers

    def sanitize_nested_list(self, lst, inner_shape, fill_value=np.nan, dtype=np.float32):
        F = len(lst)
        L, V = inner_shape
        arr = np.full((F, L, V), fill_value, dtype=dtype)
        for i, entry in enumerate(lst):
            if entry is None: continue
            for j, item in enumerate(entry[:L]):
                if item is None: continue
                a = np.array(item)
                if a.ndim==0 and V==1:
                    arr[i,j,0] = a
                elif a.ndim==1 and a.shape[0]==V:
                    arr[i,j,:] = a
        return arr

    def check_none_alignment(self, kps, scs):
        mism = []
        for i,(kp,sc) in enumerate(zip(kps,scs)):
            if (kp is None) ^ (sc is None):
                mism.append((i, kp is None, sc is None))
        if mism:
            print(f" MISMATCHES: {mism[:5]}")
        else:
            print(" ✅ None aligned")

    def save_numpy(self):
        """Sanitize and save .npy for each instrument/key."""
        for artist, songs in self.processed.items():
            for song, data in songs.items():
                base = os.path.join(self.dataset_path, artist, song)
                for dtype, content in data.items():
                    shape = (133,1) if dtype.endswith('scores') else (133,2)
                    for inst, lst in content.items():
                        arr = self.sanitize_nested_list(lst, shape)
                        outf = os.path.join(base, inst, f"{dtype}.npy")
                        os.makedirs(os.path.dirname(outf), exist_ok=True)
                        np.save(outf, arr)
                        print(f" → saved {outf}")

    def run(self, avoid_artists=None):
        self.load_results(avoid_artists=avoid_artists)
        avoid = set(avoid_artists or [])
        for artist, songs in self.raw.items():
            if artist in avoid: continue
            self.processed[artist] = {}
            for song, frames in songs.items():
                print(f"\nProcessing {artist}/{song}")
                proc, centers = self.reorder_and_map(frames)
                self.processed[artist][song] = proc
                print(" Reference centers:", centers)
                # sanity‐check alignment
                for inst in proc['keypoints']:
                    print(f"Checking {inst}")
                    self.check_none_alignment(proc['keypoints'][inst],
                                              proc['keypoint_scores'][inst])
        self.save_numpy()
