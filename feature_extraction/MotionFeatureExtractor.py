import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map
from tools.utils import smooth_keypoints


import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


import json
from sklearn.decomposition import PCA


class MotionFeatureExtractor:
    def __init__(
        self,
        dataset_dir: str,
        instruments: list,
        artist_filter: str = None,
        fps: int = 30,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
        pca_components: int = 3,
    ):
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.instruments = instruments
        self.artist_filter = artist_filter
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.smooth_win = smooth_win
        self.smooth_poly = smooth_poly
        self.body_parts_map = body_parts_map
        self.pca_components = pca_components

    def _compute_speed_accel(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # mask low-confidence
        mask = scores < self.conf_threshold
        mask = np.expand_dims(mask, -1)
        mask = np.broadcast_to(mask, keypoints.shape)
        keypoints = keypoints.copy()
        keypoints[mask] = np.nan

        # Compute velocity and acceleration using np.gradient
        velocity = np.gradient(keypoints, axis=0) * self.fps
        speed = np.where(np.isnan(velocity).any(axis=2),
                         np.nan, np.linalg.norm(velocity, axis=-1))

        acceleration = np.gradient(velocity, axis=0) * self.fps
        acceleration_magnitude = np.where(np.isnan(acceleration).any(axis=2),
                                          np.nan, np.linalg.norm(acceleration, axis=-1))

        return speed, acceleration_magnitude

    def _summarize(self, speed: np.ndarray, accel: np.ndarray) -> dict:
        summary = {
            "general": {
                "mean_speed": np.nanmean(speed, axis=1).tolist(),
                "mean_accel": np.nanmean(accel, axis=1).tolist(),
            }
        }
        for part, idxs in self.body_parts_map.items():
            summary[part] = {
                "mean_speed": np.nanmean(speed[:, idxs], axis=1).tolist(),
                "mean_accel": np.nanmean(accel[:, idxs], axis=1).tolist(),
            }
        return summary
    
    def compute_pca(self, motion_features: np.ndarray) -> list:
        valid_frames = ~np.isnan(motion_features).any(axis=1)
        motion_features_clean = motion_features[valid_frames]
        pca = PCA(n_components=self.pca_components)
        pca.fit(motion_features_clean)
        return pca.components_.tolist()

    def extract(self) -> dict:
        motion_features = {}
        for artist in tqdm(os.listdir(self.dataset_dir), desc="Artists"):
            if self.artist_filter and artist != self.artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_dir, artist)
            if not os.path.isdir(artist_dir) or artist.startswith("."):
                continue
            motion_features.setdefault(artist, {})
            for song in tqdm(os.listdir(artist_dir), desc="Songs", leave=False):
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                motion_features[artist].setdefault(song, {})
                for inst in self.instruments:
                    inst_dir = os.path.join(song_dir, inst)
                    print(f"Checking {artist}/{song}/{inst}")
                    if not os.path.isdir(inst_dir):
                        continue
                    try:
                        print(f"\nProcessing {artist}/{song}/{inst}")
                        kps = np.load(os.path.join(inst_dir, "keypoints.npy"))
                        scs = np.load(os.path.join(inst_dir, "keypoint_scores.npy"))
                        # drop lower body
                        kps = np.delete(kps, np.s_[11:23], axis=1)
                        scs = np.delete(scs, np.s_[11:23], axis=1)
                        kps = smooth_keypoints(kps=kps, smooth_poly=self.smooth_poly,
                                               smooth_win=self.smooth_win)
                        speed, acceleration = self._compute_speed_accel(kps, scs)
                        summary = self._summarize(speed, acceleration)
                        motion_features[artist][song][inst] = summary

                        # PCA computation and saving
                        motion_features_framewise = np.concatenate([speed, acceleration], axis=1)
                        pca_components = self.compute_pca(motion_features_framewise)
                        pca_output_path = os.path.join(
                            self.dataset_dir, artist, song, inst, "pca_components.json"
                        )
                        with open(pca_output_path, "w") as f:
                            json.dump(pca_components, f, indent=4)
                    except Exception as e:
                        print(f"Motion error {artist}/{song}/{inst}: {e}")
                        motion_features[artist][song][inst] = {}

        for artist, songs in motion_features.items():
            for song, insts in songs.items():
                for inst, summary in insts.items():
                    if not summary:
                        continue
                    outp = os.path.join(
                        self.dataset_dir, artist, song, inst, "motion_features.json"
                    )
                    with open(outp, "w") as f:
                        json.dump(summary, f, indent=4)
        return motion_features