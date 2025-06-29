import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from abc import ABC, abstractmethod


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402
from tools.utils import smooth_keypoints, normalize_keypoints  # noqa: E402


class BaseMotionFeatureExtractor(ABC):
    def __init__(
        self,
        dataset_dir: str,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
    ):
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.conf_threshold = conf_threshold
        self.smooth_win = smooth_win
        self.smooth_poly = smooth_poly
        self.body_parts_map = body_parts_map

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
        keypoints_to_remove = []
        for part in ["left_leg", "right_leg", "left_foot", "right_foot"]:
            keypoints_to_remove.extend(self.body_parts_map[part])

        for part in occluded_parts:
            keypoints_to_remove.extend(self.body_parts_map[part])

        keypoints_to_remove = sorted(set(keypoints_to_remove))

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

        # Normalize keypoints before deleting undesired parts to center 
        # them around the body centroid
        keypoints = normalize_keypoints(
            keypoints,
            scale_factor=1.0
        )

        # Delete keypoints and scores
        keypoints = np.delete(keypoints, keypoints_to_remove, axis=1)
        scores = np.delete(scores, keypoints_to_remove, axis=1)

        keypoints = smooth_keypoints(
            keypoints=keypoints,
            smooth_poly=self.smooth_poly,
            smooth_win=self.smooth_win,
        )

        return keypoints, scores, updated_body_parts_map

    @abstractmethod
    def _compute_features(
        self, keypoints: np.ndarray, scores: np.ndarray, fps: float
    ) -> dict:
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )