import json
import os

import numpy as np
import numpy.core.multiarray as multiarray
import torch
from dotenv import load_dotenv

NUMBER_OF_KEYPOINTS = 133
CLUSTER_DISTANCE_THRESHOLD = 100
HEIGHT_OUTLIER_THRESHOLD = 150


class PoseEstimationPostProcessor:
    def __init__(self, dataset_path):
        load_dotenv()
        self.dataset_path = dataset_path
        # safe‐globals for torch.load
        torch.serialization.add_safe_globals([multiarray._reconstruct])
        _orig_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = patched_load
        with open(
            os.path.join(dataset_path, "dataset_metadata.json"), "rb"
        ) as f:
            self.metadata = json.load(f)

    def get_adaptive_thresholds(self, frames):
        """
        Calculate adaptive thresholds based on frame dimensions and detected subject distribution.
        """
        if not frames:
            return CLUSTER_DISTANCE_THRESHOLD, HEIGHT_OUTLIER_THRESHOLD

        # Estimate frame dimensions from bounding boxes
        all_coords = []
        for frame in frames:
            for subj in frame:
                x0, y0, x1, y1 = subj["bbox"][0]
                all_coords.extend([x0, y0, x1, y1])

        if not all_coords:
            return CLUSTER_DISTANCE_THRESHOLD, HEIGHT_OUTLIER_THRESHOLD

        # Rough estimate of frame dimensions
        frame_width = max(all_coords) - min(all_coords)
        frame_height = max(
            [subj["bbox"][0][3] for frame in frames for subj in frame]
        ) - min([subj["bbox"][0][1] for frame in frames for subj in frame])

        # Adaptive cluster distance: ~8% of frame width
        adaptive_cluster_threshold = max(50, min(200, frame_width * 0.08))

        # Adaptive height threshold: ~15% of frame height
        adaptive_height_threshold = max(100, min(300, frame_height * 0.15))

        return adaptive_cluster_threshold, adaptive_height_threshold

    def calculate_subject_centers(self, frames):
        """Return list of [x,y] for consistently detected subjects."""
        # Get adaptive thresholds
        adaptive_cluster_threshold, _ = self.get_adaptive_thresholds(frames)

        all_centers = []
        for fi, frame in enumerate(frames):
            for si, subj in enumerate(frame):
                x0, y0, x1, y1 = subj["bbox"][0]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                all_centers.append((si, cx, cy))
        # cluster by proximity
        refs = []
        for si, cx, cy in all_centers:
            placed = False
            for r in refs:
                if (
                    np.hypot(cx - r["cx"], cy - r["cy"])
                    < adaptive_cluster_threshold
                ):
                    # update avg
                    r["cx"] = (r["cx"] * r["count"] + cx) / (r["count"] + 1)
                    r["cy"] = (r["cy"] * r["count"] + cy) / (r["count"] + 1)
                    r["count"] += 1
                    placed = True
                    break
            if not placed:
                refs.append({"cx": cx, "cy": cy, "count": 1})
        # filter infrequent
        min_count = len(frames) * 0.66
        centers = [(r["cx"], r["cy"]) for r in refs if r["count"] > min_count]
        print(
            f"  Using adaptive cluster threshold: {adaptive_cluster_threshold:.1f}"
        )
        return centers

    def reorder_and_map(self, frames, layout):
        """
        1) pick bottom-N by y
        2) filter out height outliers (artwork false positives)
        3) reorder each frame to match
        4) map subject-idx→instrument by x left→right
        returns (consistent_dict, centers)
        """
        centers = self.calculate_subject_centers(frames)

        # Filter out height outliers before selecting final centers
        print(f"  Found {len(centers)} potential subjects")
        self.debug_print_centers(centers, "Before height filtering")
        centers = self.filter_height_outliers(centers, frames)
        print(f"  After height filtering: {len(centers)} subjects")
        self.debug_print_centers(centers, "After height filtering")

        # pick bottom num_instruments
        centers.sort(key=lambda c: c[1], reverse=True)
        centers = centers[: len(layout)]
        self.debug_print_centers(
            centers, f"Final selected {len(layout)} subjects"
        )

        # Get adaptive cluster threshold for frame matching
        adaptive_cluster_threshold, _ = self.get_adaptive_thresholds(frames)

        # map subject indices per frame
        reorganized = []
        for frame in frames:
            slot = [None] * len(centers)
            for subj in frame:
                x0, y0, x1, y1 = subj["bbox"][0]
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                # find nearest center
                dists = [np.hypot(cx - cx0, cy - cy0) for cx0, cy0 in centers]
                idx = int(np.argmin(dists))
                if dists[idx] < adaptive_cluster_threshold:
                    slot[idx] = subj
            reorganized.append(slot)
        # assign instruments based on x‐order
        xs = [c[0] for c in centers]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        subj2inst = {si: layout[pos] for pos, si in enumerate(order)}
        # build dict of per-instrument time lists
        out = {"keypoints": {}, "keypoint_scores": {}}
        for si, inst in subj2inst.items():
            out["keypoints"][inst] = []
            out["keypoint_scores"][inst] = []
            for slot in reorganized:
                subj = slot[si]
                if subj:
                    out["keypoints"][inst].append(subj["keypoints"])
                    out["keypoint_scores"][inst].append(
                        subj["keypoint_scores"]
                    )
                else:
                    out["keypoints"][inst].append(None)
                    out["keypoint_scores"][inst].append(None)
        return out, centers

    def sanitize_nested_list(
        self, lst, inner_shape, fill_value=np.nan, dtype=np.float32
    ):
        F = len(lst)
        L, V = inner_shape
        arr = np.full((F, L, V), fill_value, dtype=dtype)
        for i, entry in enumerate(lst):
            if entry is None:
                continue
            for j, item in enumerate(entry[:L]):
                if item is None:
                    continue
                a = np.array(item)
                if a.ndim == 0 and V == 1:
                    arr[i, j, 0] = a
                elif a.ndim == 1 and a.shape[0] == V:
                    arr[i, j, :] = a
        return arr

    def check_none_alignment(self, kps, scs):
        mism = []
        for i, (kp, sc) in enumerate(zip(kps, scs)):
            if (kp is None) ^ (sc is None):
                mism.append((i, kp is None, sc is None))
        if mism:
            print(f" MISMATCHES: {mism[:5]}")
        else:
            print(" ✅ None aligned")

    def run(self, artist_filter=None, song_filter=None, force=False):
        """
        Process each song end-to-end before moving on,
        to avoid loading everything into memory at once.
        """
        for artist in os.listdir(self.dataset_path):
            if artist_filter and artist != artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_path, artist)
            if not os.path.isdir(artist_dir):
                continue
            for song in os.listdir(artist_dir):
                if song_filter and song != song_filter:
                    continue
                song_dir = os.path.join(artist_dir, song)

                song_metadata = self.metadata.get(artist, {}).get(song, {})
                if song_metadata != {} and song_metadata["moving_camera"]:
                    print(f"Skipping {artist}/{song}: {song_metadata}")
                    continue

                pkl_file = os.path.join(song_dir, "pose_estimation.pkl")
                if not os.path.isfile(pkl_file):
                    continue

                layout = song_metadata["layout"]
                skip = True
                for inst in layout:
                    inst_dir = os.path.join(song_dir, str(inst))
                    kps_file = os.path.join(inst_dir, "keypoints.npy")
                    scs_file = os.path.join(inst_dir, "keypoint_scores.npy")
                    if not (
                        os.path.isfile(kps_file) and os.path.isfile(scs_file)
                    ):
                        skip = False
                        break
                if skip and not force:
                    print(f"Skipping {artist}/{song} (already processed)")
                    continue

                print(f"\nProcessing {artist}/{song}")
                frames = torch.load(pkl_file)
                proc, centers = self.reorder_and_map(frames, layout)
                print(" Reference centers:", centers)
                # sanity‐check alignment
                for inst in proc["keypoints"]:
                    self.check_none_alignment(
                        proc["keypoints"][inst], proc["keypoint_scores"][inst]
                    )

                for dtype, content in proc.items():
                    self._save_numpy_for(artist, song, dtype, content)
                del frames, proc

    def _save_numpy_for(self, artist, song, dtype, content):
        """Helper to save one song's outputs and avoid storing them."""
        base = os.path.join(self.dataset_path, artist, song)
        shape = (
            (NUMBER_OF_KEYPOINTS, 1)
            if dtype.endswith("scores")
            else (NUMBER_OF_KEYPOINTS, 2)
        )
        for inst, lst in content.items():
            arr = self.sanitize_nested_list(lst, shape)
            outf = os.path.join(base, str(inst), f"{dtype}.npy")
            os.makedirs(os.path.dirname(outf), exist_ok=True)
            np.save(outf, arr)
            print(f" → saved {outf}")

    def filter_height_outliers(self, centers, frames):
        """
        Remove centers that are significantly higher than others (likely artwork).
        Uses adaptive outlier detection based on y-coordinate distribution.
        """
        if len(centers) <= 1:
            return centers

        # Get adaptive height threshold
        _, adaptive_height_threshold = self.get_adaptive_thresholds(frames)

        y_coords = [c[1] for c in centers]

        # Calculate statistics for y-coordinates
        median_y = np.median(y_coords)
        std_y = np.std(y_coords)
        q75 = np.percentile(y_coords, 75)

        # Use adaptive threshold based on data distribution and frame size
        # If there's high variance, use a more conservative threshold
        final_threshold = min(adaptive_height_threshold, max(50, std_y * 1.5))

        # Filter out centers that are too high (artwork territory)
        filtered_centers = []
        for cx, cy in centers:
            # Multiple criteria for filtering:
            # 1. Too far above median (static threshold)
            # 2. Too far above 75th percentile (adaptive threshold)
            if (cy < median_y - final_threshold) or (
                cy < q75 - final_threshold * 0.8
            ):
                print(
                    f"  Filtering out potential artwork at ({cx:.1f}, {cy:.1f}) - too high (median: {median_y:.1f}, threshold: {final_threshold:.1f})"
                )
                continue
            filtered_centers.append((cx, cy))

        return filtered_centers

    def debug_print_centers(self, centers, title="Centers"):
        """Print center coordinates for debugging."""
        print(f"  {title}:")
        for i, (cx, cy) in enumerate(centers):
            print(f"    Subject {i}: ({cx:.1f}, {cy:.1f})")
        if centers:
            y_coords = [c[1] for c in centers]
            print(
                f"    Y-coord stats: min={min(y_coords):.1f}, max={max(y_coords):.1f}, median={np.median(y_coords):.1f}"
            )
        print()
