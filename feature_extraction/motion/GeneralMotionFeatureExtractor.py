import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from .BaseMotionFeatureExtractor import BaseMotionFeatureExtractor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402


class GeneralMotionFeatureExtractor(BaseMotionFeatureExtractor):
    def __init__(
        self,
        dataset_dir: str,
        instruments: list,
        artist_filter: str = None,
        song_filter: str = None,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
        pca_components: int = 3,
        motion_output_filename: str = "motion_features.json",
        pca_output_filename: str = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            conf_threshold=conf_threshold,
            smooth_win=smooth_win,
            smooth_poly=smooth_poly,
        )
        self.instruments = instruments
        self.artist_filter = artist_filter
        self.song_filter = song_filter
        self.body_parts_map = body_parts_map
        self.pca_components = pca_components
        self.output_filename = motion_output_filename
        self.pca_output_filename = pca_output_filename

        with open(
            os.path.join(dataset_dir, "dataset_metadata.json"), "r"
        ) as f:
            self.dataset_metadata = json.load(f)

    def _compute_features(
        self, keypoints: np.ndarray, scores: np.ndarray, fps: float
    ) -> dict:
        # mask low-confidence
        mask = scores < self.conf_threshold
        mask = np.broadcast_to(mask, keypoints.shape)
        keypoints = keypoints.copy()
        keypoints[mask] = np.nan

        # Compute velocity and acceleration using np.gradient
        velocity = np.gradient(keypoints, axis=0) * fps
        speed = np.where(
            np.isnan(velocity).any(axis=2),
            np.nan,
            np.linalg.norm(velocity, axis=-1),
        )

        acceleration = np.gradient(velocity, axis=0) * fps
        acceleration_magnitude = np.where(
            np.isnan(acceleration).any(axis=2),
            np.nan,
            np.linalg.norm(acceleration, axis=-1),
        )

        return {"speed": speed, "acceleration": acceleration_magnitude}

    def _summarize(
        self, speed: np.ndarray, accel: np.ndarray, body_parts_map: dict
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

    def extract(
        self, all_body_parts: bool = False, force: bool = False
    ) -> dict:
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
                if self.song_filter and song != self.song_filter:
                    continue
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                if (
                    self.dataset_metadata[artist][song]["layout"]
                    != self.instruments
                ):
                    print(
                        f"Skipping {artist}/{song}: layout mismatch with instruments."
                    )
                    continue
                if (
                    self.dataset_metadata[artist][song][
                        "correct_body_detection"
                    ]
                    is not True
                ):
                    print(f"Skipping {artist}/{song}: body not detected.")
                    continue

                print(f"Processing song: {song}")
                motion_features[artist].setdefault(song, {})
                for instrument in self.instruments:
                    inst_dir = os.path.join(song_dir, instrument)
                    if not os.path.isdir(inst_dir):
                        continue
                    if (
                        os.path.exists(
                            os.path.join(inst_dir, self.output_filename)
                        )
                        and not force
                    ):
                        print(
                            f"Skipping {artist}/{song}/{instrument}: already processed."
                        )
                        continue
                    try:
                        keypoints = np.load(
                            os.path.join(inst_dir, "keypoints.npy")
                        )
                        scores = np.load(
                            os.path.join(inst_dir, "keypoint_scores.npy")
                        )
                        metadata = self.dataset_metadata.get(artist, {}).get(
                            song, {}
                        )
                        occluded_parts = (
                            []
                            if all_body_parts
                            else self._get_occluded_parts(instrument, metadata)
                        )
                        keypoints, scores, updated_body_parts_map = (
                            self._process_keypoints(
                                keypoints, scores, occluded_parts
                            )
                        )
                        features = self._compute_features(
                            keypoints,
                            scores,
                            self.dataset_metadata[artist][song]["fps"],
                        )
                        summary = self._summarize(
                            features["speed"],
                            features["acceleration"],
                            updated_body_parts_map,
                        )
                        motion_features[artist][song][instrument] = summary
                    except Exception as e:
                        print(
                            f"Motion error {artist}/{song}/{instrument}: {e}"
                        )
                        motion_features[artist][song][instrument] = {}
                    try:
                        if self.pca_output_filename is not None:
                            motion_features_framewise = np.concatenate(
                                [features["speed"], features["acceleration"]],
                                axis=1,
                            )
                            pca_components = self.compute_pca(
                                motion_features_framewise
                            )
                            pca_output_path = os.path.join(
                                self.dataset_dir,
                                artist,
                                song,
                                instrument,
                                self.pca_output_filename,
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
                        self.output_filename,
                    )
                    with open(outp, "w") as f:
                        json.dump(summary, f, indent=4)
        return motion_features
