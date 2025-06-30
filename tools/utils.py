import numpy as np
from scipy.signal import savgol_filter


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
