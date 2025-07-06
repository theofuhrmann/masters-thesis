import numpy as np
from scipy.signal import savgol_filter
import torch
import os
import sys
from einops import rearrange
from dotenv import load_dotenv


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv()

vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from vovit.core.kabsch import register_sequence_of_landmarks # type: ignore # noqa: E402


def smooth_keypoints(
    keypoints: np.ndarray,
    smooth_win: int = 5,
    smooth_poly: int = 2,
    mode: str = "nearest",
) -> np.ndarray:
    """Smooth keypoints using Savitzky-Golay filter."""
    _, K, D = keypoints.shape
    out = keypoints.copy()
    for k in range(K):
        for d in range(D):
            out[:, k, d] = savgol_filter(
                keypoints[:, k, d],
                window_length=smooth_win,
                polyorder=smooth_poly,
                mode=mode,
            )
    return out


def normalize_keypoints(keypoints, scale_factor=1.0):
    """
    Normalize 2D keypoints per frame by centering them around the mean position of all keypoints in each frame.

    Parameters:
    keypoints (numpy.ndarray): Array of shape (n_frames, n_keypoints, 2).
    scale_factor (float): Scaling factor.

    Returns:
    numpy.ndarray: Normalized keypoints of same shape.
    """
    keypoints = np.array(keypoints)
    normalized = np.full_like(keypoints, np.nan)  # Initialize output with NaNs

    for i, frame in enumerate(keypoints):
        if np.isnan(frame).any():
            continue  # Leave as NaN
        mean_point = np.mean(frame, axis=0, keepdims=True)
        centered = frame - mean_point
        max_distance = np.max(np.linalg.norm(centered, axis=1, keepdims=True))
        normalized[i] = centered / (max_distance + 1e-8) * scale_factor

    return normalized


def process_3ddfa_keypoints(keypoints: np.ndarray) -> np.ndarray:
    keypoints = torch.from_numpy(keypoints).unsqueeze(0).float()
    mean_face_path = os.path.join(os.path.dirname(__file__), "speech_mean_face.npy")
    mean_face = torch.from_numpy(np.load(mean_face_path)).float()
    face_ld = torch.stack(
        [
            register_sequence_of_landmarks(
                frames[..., :48],
                mean_face[:, :48],
                per_frame=True,
                display_sequence=frames,
            )
            for frames in keypoints
        ],
        dim=0,
    )
    return rearrange(face_ld, "b t c j -> b t j c")[:, :, :, :2].squeeze(0).numpy()