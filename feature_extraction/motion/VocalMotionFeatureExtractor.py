import json
import os
import sys

import numpy as np
from tqdm import tqdm

from .BaseMotionFeatureExtractor import BaseMotionFeatureExtractor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map, face_parts_map  # noqa: E402
from tools.utils import process_3ddfa_keypoints, smooth_keypoints  # noqa: E402


class VocalMotionFeatureExtractor(BaseMotionFeatureExtractor):
    def __init__(
        self,
        dataset_dir: str,
        artist_filter: str = None,
        song_filter: str = None,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
        motion_output_filename: str = "vocal_motion_features.json",
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

    def _compute_features(self, keypoints: np.ndarray) -> dict:
        mouth_pts = keypoints[:, self.mouth_idxs, :]

        x = mouth_pts[:, :, 0]
        y = mouth_pts[:, :, 1]
        mouth_area = 0.5 * np.abs(
            np.sum(
                x * np.roll(y, shift=-1, axis=1)
                - y * np.roll(x, shift=-1, axis=1),
                axis=1,
            )
        )

        jaw_to_nose = np.linalg.norm(
            keypoints[:, self.nose_idx, :] - keypoints[:, self.chin_idx, :],
            axis=1,
        )

        return {
            "mouth_area": mouth_area.tolist(),
            "jaw_to_nose": jaw_to_nose.tolist(),
        }

    def _mask_false_positive_frames(
        self, artist: str, song: str, keypoints: np.ndarray
    ) -> np.ndarray:
        try:
            song_meta = self.dataset_metadata.get(artist, {}).get(song, {})
            intervals = song_meta.get("face_false_positive_frames")
            if not intervals:
                return keypoints
            n_frames = keypoints.shape[0]
            for start, end in intervals:
                if start is None or end is None:
                    continue
                if end < 0 or start >= n_frames:
                    continue
                s = max(0, int(start))
                e = min(n_frames - 1, int(end))
                if s <= e:
                    keypoints[s : e + 1] = np.nan
        except Exception as e:
            print(
                f"Warning: could not apply false positive mask for {artist}/{song}: {e}"
            )
        return keypoints

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
                inst_dir = os.path.join(artist_dir, song, "vocal")
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
                try:
                    keypoints = np.load(
                        os.path.join(inst_dir, "face_keypoints.npy")
                    )
                    keypoints = process_3ddfa_keypoints(keypoints)
                    keypoints = smooth_keypoints(
                        keypoints=keypoints,
                        smooth_poly=self.smooth_poly,
                        smooth_win=self.smooth_win,
                    )
                    keypoints = self._mask_false_positive_frames(
                        artist, song, keypoints
                    )

                    self.chin_idx = face_parts_map["face_contour"][8]
                    self.nose_idx = face_parts_map["nose"][6]
                    self.mouth_idxs = face_parts_map["mouth"]

                    motion_features[artist][song] = self._compute_features(
                        keypoints
                    )

                except Exception as e:
                    print(f"Motion error {artist}/{song}: {e}")
                    motion_features[artist][song] = {}

        for artist, songs in motion_features.items():
            for song, features in songs.items():
                outp = os.path.join(
                    self.dataset_dir,
                    artist,
                    song,
                    "vocal",
                    self.output_filename,
                )
                with open(outp, "w") as f:
                    json.dump(features, f, indent=4)
        return motion_features
