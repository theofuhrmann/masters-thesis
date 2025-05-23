from body_parts_map import body_parts_map
from utils import smooth_keypoints


import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


import json
import os


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
                    if not os.path.isdir(inst_dir):
                        continue
                    try:
                        kps = np.load(os.path.join(inst_dir, "keypoints.npy"))
                        scs = np.load(os.path.join(inst_dir, "keypoint_scores.npy"))
                        # drop lower body
                        kps = np.delete(kps, np.s_[11:23], axis=1)
                        scs = np.delete(scs, np.s_[11:23], axis=1)
                        kps = smooth_keypoints(kps=kps, smooth_poly=self.smooth_poly,
                                               smooth_win=self.smooth_win)
                        speed, accel = self._compute_speed_accel(kps, scs)
                        summary = self._summarize(speed, accel)
                        motion_features[artist][song][inst] = summary
                    except Exception as e:
                        print(f"Motion error {artist}/{song}/{inst}: {e}")
                        motion_features[artist][song][inst] = {}
        # save out
        for artist, songs in motion_features.items():
            for song, insts in songs.items():
                for inst, summary in insts.items():
                    if not summary:
                        continue
                    outp = os.path.join(
                        self.dataset_dir, artist, song, inst, "motion_features_3.json"
                    )
                    with open(outp, "w") as f:
                        json.dump(summary, f, indent=4)
        return motion_features