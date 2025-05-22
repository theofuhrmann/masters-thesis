import os
import cv2
import torch
import numpy.core.multiarray as multiarray
from dotenv import load_dotenv
from tqdm import tqdm
from mmpose.apis import init_model, MMPoseInferencer
from mmpose.utils import register_all_modules

class PoseEstimator:
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        # register and patch torch.load
        register_all_modules()
        torch.serialization.add_safe_globals([multiarray._reconstruct])
        _orig_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = patched_load

        # init model + inferencer
        init_model(config_file, checkpoint_file, device=device)  # needed for low-level APIs
        self.inferencer = MMPoseInferencer(
            pose2d=os.path.splitext(os.path.basename(config_file))[0],
            pose2d_weights=checkpoint_file,
            device=device
        )

    def process_dataset(self, dataset_path, avoid_artists=None):
        avoid = set(avoid_artists or [])
        for artist in os.listdir(dataset_path):
            if artist in avoid: continue
            for song in os.listdir(os.path.join(dataset_path, artist)):
                if song not in ["Segment3"]:
                    continue
                song_dir = os.path.join(dataset_path, artist, song)
                video = self._find_video(song_dir)
                if video:
                    self._process_video(artist, song, song_dir, video)

    def _find_video(self, song_dir):
        for f in os.listdir(song_dir):
            if f.lower().endswith('.mov'):
                return os.path.join(song_dir, f)
        return None

    def _process_video(self, artist, song, song_dir, video_path):
        print(f"Processing {song} for {artist}")
        gen = self.inferencer(video_path, show=False)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        results = [r['predictions'][0] for r in tqdm(gen, total=frame_count)]
        out_pkl = os.path.join(song_dir, 'pose_estimation.pkl')
        with open(out_pkl, 'wb') as f:
            torch.save(results, f)
        print(f" → saved {out_pkl}")

def main():
    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")
    config_file = "../mmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py"
    checkpoint_file = "../mmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

    estimator = PoseEstimator(config_file, checkpoint_file, device='cuda:0')
    avoid_artists = [
        "Abhiram Bode",
        "Aditi Prahalad",
    ]
    estimator.process_dataset(dataset_path, avoid_artists=avoid_artists)

if __name__ == "__main__":
    main()