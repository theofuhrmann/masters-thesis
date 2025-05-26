import numpy as np
from scipy.signal import savgol_filter


def smooth_keypoints(
    kps: np.ndarray,
    smooth_win: int = 5,
    smooth_poly: int = 2,
    mode: str = "nearest",
) -> np.ndarray:
    """Smooth keypoints using Savitzky-Golay filter."""
    _, K, D = kps.shape
    out = kps.copy()
    for k in range(K):
        for d in range(D):
            out[:, k, d] = savgol_filter(
                kps[:, k, d],
                window_length=smooth_win,
                polyorder=smooth_poly,
                mode=mode,
            )
    return out
