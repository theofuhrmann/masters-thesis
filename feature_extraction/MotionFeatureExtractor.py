import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402
from tools.utils import smooth_keypoints  # noqa: E402


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

        with open(
            os.path.join(dataset_dir, "dataset_metadata.json"), "r"
        ) as f:
            self.dataset_metadata = json.load(f)

    def _get_occluded_parts(self, instrument, song_metadata):
        layout = song_metadata.get("layout")
        if instrument == layout[0]:
            return ["left_arm", "left_hand"]
        elif instrument == layout[-1]:
            return ["right_arm", "right_hand"]

        return []

    def _process_keypoints(self, keypoints, scores, occluded_parts):
        lower_body_indexes = list(range(11, 23))
        occluded_keypoints = []

        keypoints = np.delete(keypoints, lower_body_indexes, axis=1)
        scores = np.delete(scores, lower_body_indexes, axis=1)

        for part in occluded_parts:
            occluded_keypoints.extend(self.body_parts_map[part])

        # Combine all keypoints to remove and sort them
        keypoints_to_remove = sorted(set(occluded_keypoints))

        # Build old-to-new index mapping
        total_kpts = keypoints.shape[1]
        index_mapping = {}
        new_idx = 0
        for old_idx in range(total_kpts):
            if old_idx not in keypoints_to_remove:
                index_mapping[old_idx] = new_idx
                new_idx += 1

        # Update body_parts_map
        updated_body_parts_map = {}
        for part, indices in self.body_parts_map.items():
            if part in occluded_parts:
                continue
            new_indices = [
                index_mapping[i] for i in indices if i in index_mapping
            ]
            if new_indices:
                updated_body_parts_map[part] = new_indices

        # Delete keypoints and scores
        keypoints = np.delete(keypoints, keypoints_to_remove, axis=1)
        scores = np.delete(scores, keypoints_to_remove, axis=1)

        keypoints = smooth_keypoints(
            keypoints=keypoints,
            smooth_poly=self.smooth_poly,
            smooth_win=self.smooth_win,
        )

        return keypoints, scores, updated_body_parts_map

    def _compute_speed_and_acceleration(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # mask low-confidence
        mask = scores < self.conf_threshold
        mask = np.broadcast_to(mask, keypoints.shape)
        keypoints = keypoints.copy()
        keypoints[mask] = np.nan

        # Compute velocity and acceleration using np.gradient
        velocity = np.gradient(keypoints, axis=0) * self.fps
        speed = np.where(
            np.isnan(velocity).any(axis=2),
            np.nan,
            np.linalg.norm(velocity, axis=-1),
        )

        acceleration = np.gradient(velocity, axis=0) * self.fps
        acceleration_magnitude = np.where(
            np.isnan(acceleration).any(axis=2),
            np.nan,
            np.linalg.norm(acceleration, axis=-1),
        )

        return speed, acceleration_magnitude

    def _summarize(
        self,
        speed: np.ndarray,
        accel: np.ndarray,
        body_parts_map: dict,
    ) -> dict:
        summary = {
            "general": {
                "mean_speed": np.nanmean(speed, axis=1).tolist(),
                "mean_acceleration": np.nanmean(accel, axis=1).tolist(),
            }
        }
        for part, idxs in body_parts_map.items():
            summary[part] = {
                "mean_speed": np.nanmean(speed[:, idxs], axis=1).tolist(),
                "mean_acceleration": np.nanmean(
                    accel[:, idxs], axis=1
                ).tolist(),
            }

        return summary

    def compute_pca(self, motion_features: np.ndarray) -> dict:
        valid_frames = ~np.isnan(motion_features).any(axis=1)
        print(
            f"Computing PCA on {np.sum(valid_frames)} valid frames out of {motion_features.shape[0]} total frames."
        )
        motion_features_clean = motion_features[valid_frames]
        pca = PCA(n_components=self.pca_components)
        pca.fit(motion_features_clean)
        return {
            "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }

    def extract(self) -> dict:
        motion_features = {}
        for artist in tqdm(os.listdir(self.dataset_dir), desc="Artists"):
            if self.artist_filter and artist != self.artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_dir, artist)
            if not os.path.isdir(artist_dir) or artist.startswith("."):
                continue
            print(f"Processing artist: {artist}")
            motion_features.setdefault(artist, {})
            for song in tqdm(
                os.listdir(artist_dir), desc="Songs", leave=False
            ):
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                print(f"Processing song: {song}")
                motion_features[artist].setdefault(song, {})
                for instrument in self.instruments:
                    inst_dir = os.path.join(song_dir, instrument)
                    print(f"Checking {artist}/{song}/{instrument}")
                    if not os.path.isdir(inst_dir):
                        continue
                    try:
                        print(f"\nProcessing {artist}/{song}/{instrument}")
                        keypoints = np.load(
                            os.path.join(inst_dir, "keypoints.npy")
                        )
                        scores = np.load(
                            os.path.join(inst_dir, "keypoint_scores.npy")
                        )
                        metadata = self.dataset_metadata.get(artist, {}).get(
                            song, {}
                        )
                        occluded_parts = self._get_occluded_parts(
                            instrument, metadata
                        )
                        keypoints, scores, updated_body_parts_map = (
                            self._process_keypoints(
                                keypoints, scores, occluded_parts
                            )
                        )
                        speed, acceleration = (
                            self._compute_speed_and_acceleration(
                                keypoints, scores
                            )
                        )
                        summary = self._summarize(
                            speed,
                            acceleration,
                            updated_body_parts_map,
                        )
                        motion_features[artist][song][instrument] = summary
                    except Exception as e:
                        print(
                            f"Motion error {artist}/{song}/{instrument}: {e}"
                        )
                        motion_features[artist][song][instrument] = {}
                    try:
                        # PCA computation and saving
                        motion_features_framewise = np.concatenate(
                            [speed, acceleration], axis=1
                        )
                        pca_components = self.compute_pca(
                            motion_features_framewise
                        )
                        pca_output_path = os.path.join(
                            self.dataset_dir,
                            artist,
                            song,
                            instrument,
                            "pca_components.json",
                        )
                        with open(pca_output_path, "w") as f:
                            json.dump(pca_components, f, indent=4)
                    except Exception as e:
                        print(f"PCA error {artist}/{song}/{instrument}: {e}")

        for artist, songs in motion_features.items():
            for song, insts in songs.items():
                for instrument, summary in insts.items():
                    if not summary:
                        continue
                    outp = os.path.join(
                        self.dataset_dir,
                        artist,
                        song,
                        instrument,
                        "motion_features.json",
                    )
                    with open(outp, "w") as f:
                        json.dump(summary, f, indent=4)
        return motion_features
