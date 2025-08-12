import json
import os
import sys

import numpy as np
from tqdm import tqdm

from .BaseMotionFeatureExtractor import BaseMotionFeatureExtractor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402


class ViolinMotionFeatureExtractor(BaseMotionFeatureExtractor):
    def __init__(
        self,
        dataset_dir: str,
        artist_filter: str = None,
        song_filter: str = None,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
        motion_output_filename: str = "violin_motion_features.json",
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            conf_threshold=conf_threshold,
            smooth_win=smooth_win,
            smooth_poly=smooth_poly,
        )
        self.artist_filter = artist_filter
        self.song_filter = song_filter
        self.body_parts_map = body_parts_map
        self.output_filename = motion_output_filename

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

        wrist = keypoints[:, self.wrist_idx, :]
        elbow = keypoints[:, self.elbow_idx, :]
        shoulder = keypoints[:, self.shoulder_idx, :]

        # Wrist velocity
        wrist_velocity = np.gradient(wrist, axis=0) * fps
        wrist_speed = np.where(
            np.isnan(wrist_velocity).any(axis=1),
            np.nan,
            np.linalg.norm(wrist_velocity, axis=-1),
        )

        # Elbow angle (angle between shoulder-elbow and wrist-elbow)
        vec1 = shoulder - elbow
        vec2 = wrist - elbow
        norm1 = np.linalg.norm(vec1, axis=1)
        norm2 = np.linalg.norm(vec2, axis=1)
        dot = np.einsum("ij,ij->i", vec1, vec2)
        elbow_angle = np.degrees(
            np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
        )

        # Arm extension (distance shoulder-wrist)
        arm_extension = np.linalg.norm(wrist - shoulder, axis=1)
        return {
            "wrist_speed": wrist_speed.tolist(),
            "elbow_angle": elbow_angle.tolist(),
            "arm_extension": arm_extension.tolist(),
        }

    def extract(self, force: bool = False) -> dict:
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

                if self.dataset_metadata[artist][song]["layout"] != [
                    "violin",
                    "vocal",
                    "mridangam",
                ]:
                    continue

                inst_dir = os.path.join(artist_dir, song, "violin")
                if not os.path.isdir(inst_dir) or song.startswith("."):
                    continue
                print(f"Processing song: {song}")
                motion_features[artist].setdefault(song, {})
                if (
                    os.path.exists(
                        os.path.join(inst_dir, self.output_filename)
                    )
                    and not force
                ):
                    print(f"Skipping {artist}/{song}: already processed.")
                    continue
                # try:
                keypoints = np.load(os.path.join(inst_dir, "keypoints.npy"))
                scores = np.load(os.path.join(inst_dir, "keypoint_scores.npy"))
                metadata = self.dataset_metadata.get(artist, {}).get(song, {})
                occluded_parts = self._get_occluded_parts("violin", metadata)
                print(f"Original keypoints shape: {keypoints.shape}")
                keypoints, scores, updated_body_parts_map = (
                    self._process_keypoints(keypoints, scores, occluded_parts)
                )
                print(f"Processed keypoints shape: {keypoints.shape}")
                self.shoulder_idx = updated_body_parts_map["right_arm"][0]
                self.elbow_idx = updated_body_parts_map["right_arm"][1]
                self.wrist_idx = updated_body_parts_map["right_arm"][2]

                motion_features[artist][song] = self._compute_features(
                    keypoints,
                    scores,
                    self.dataset_metadata[artist][song]["fps"],
                )

                # except Exception as e:
                #     print(
                #         f"Motion error {artist}/{song}: {e}"
                #     )
                #     motion_features[artist][song] = {}

        for artist, songs in motion_features.items():
            for song, features in songs.items():
                outp = os.path.join(
                    self.dataset_dir,
                    artist,
                    song,
                    "violin",
                    self.output_filename,
                )
                with open(outp, "w") as f:
                    json.dump(features, f, indent=4)
        return motion_features
