import os

import cv2
import numpy.core.multiarray as multiarray
import torch
from mmpose.apis import MMPoseInferencer, init_model
from mmpose.utils import register_all_modules
from tqdm import tqdm


class PoseEstimator:
    def __init__(self, config_file, checkpoint_file, device="cuda:0"):
        # register and patch torch.load
        register_all_modules()
        torch.serialization.add_safe_globals([multiarray._reconstruct])
        _orig_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = patched_load

        # init model + inferencer
        init_model(
            config_file, checkpoint_file, device=device
        )  # needed for low-level APIs
        self.inferencer = MMPoseInferencer(
            pose2d=os.path.splitext(os.path.basename(config_file))[0],
            pose2d_weights=checkpoint_file,
            device=device,
        )

    def process_dataset(
        self, dataset_path, artist_filter=None, song_filter=None, force=False
    ):
        for artist in os.listdir(dataset_path):
            if artist_filter and artist != artist_filter:
                continue
            artist_path = os.path.join(dataset_path, artist)
            print(f"Processing {artist}")
            for song in os.listdir(artist_path):
                if song_filter and song != song_filter:
                    continue
                song_dir = os.path.join(dataset_path, artist, song)
                video = self._find_video(song_dir)
                already_processed = os.path.exists(
                    os.path.join(song_dir, "pose_estimation.pkl")
                )
                if already_processed and not force:
                    print(f" → already processed {song}")
                    continue
                if video:
                    self._process_video(artist, song, song_dir, video)

    def _find_video(self, song_dir):
        for f in os.listdir(song_dir):
            if f.lower().endswith(".mov"):
                return os.path.join(song_dir, f)
        return None

    def _process_video(self, artist, song, song_dir, video_path):
        print(f"Processing {song} for {artist}")
        gen = self.inferencer(video_path, show=False, batch_size=4)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {video_path}")
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            results = [
                r["predictions"][0] for r in tqdm(gen, total=frame_count)
            ]
            out_pkl = os.path.join(song_dir, "pose_estimation.pkl")
            with open(out_pkl, "wb") as f:
                torch.save(results, f)
            print(f" → saved {out_pkl}")
