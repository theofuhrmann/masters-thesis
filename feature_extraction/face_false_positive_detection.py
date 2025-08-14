import argparse
import json
import os
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")


def load_metadata(path: str) -> Dict:
    with open(os.path.join(path, "dataset_metadata.json"), "r") as f:
        return json.load(f)


def detect_false_positive_frames(
    face_keypoints: np.ndarray,
    body_keypoints: np.ndarray,
    distance_threshold: float,
) -> List[tuple]:
    """Return intervals of frames where the face detection is likely a false positive.

    Heuristic: compute centroid of all face keypoints (x,y) and distance to the
    body keypoints nose (index 0). If distance > distance_threshold -> false positive.

    Returns:
        List of tuples (start_frame, end_frame) representing false positive intervals.
    """
    # Expect shapes: face_keypoints (F, 3, FKP) and body_keypoints (F, BKP, 2)
    min_frames = min(len(face_keypoints), len(body_keypoints))
    if min_frames == 0:
        return []

    # face_keypoints shape: (F, 3, FKP) - take first 2 channels (x,y) and transpose
    face_xy = face_keypoints[:min_frames, :2, :].transpose(
        0, 2, 1
    )  # (F, FKP, 2)
    body_xy = body_keypoints[:min_frames, :, :2]  # (F, BKP, 2)

    centroid = face_xy.mean(axis=1)  # (F, 2)
    nose = body_xy[:, 0, :2]  # assuming 0-index is nose

    dists = np.linalg.norm(centroid - nose, axis=1)
    false_positive_indices = np.where(dists > distance_threshold)[0]

    # Convert indices to intervals
    if len(false_positive_indices) == 0:
        return []

    intervals = []
    start = false_positive_indices[0]
    end = start

    for i in range(1, len(false_positive_indices)):
        if false_positive_indices[i] == end + 1:
            end = false_positive_indices[i]
        else:
            intervals.append((int(start), int(end)))
            start = false_positive_indices[i]
            end = start

    intervals.append((int(start), int(end)))
    return intervals


def process(
    dataset_metadata: Dict,
    distance_threshold: float,
    instrument: str = "vocal",
):
    summary = {}

    for artist, songs in tqdm(dataset_metadata.items(), desc="Artists"):
        for song, meta in songs.items():
            if meta.get("face_detected") is not False:  # deprecated
                continue
            vocal_dir = os.path.join(DATASET_PATH, artist, song, instrument)
            face_kp_path = os.path.join(vocal_dir, "face_keypoints.npy")
            body_kp_path = os.path.join(vocal_dir, "keypoints.npy")
            if not (
                os.path.isfile(face_kp_path) and os.path.isfile(body_kp_path)
            ):
                continue
            try:
                face_kp = np.load(face_kp_path)
                body_kp = np.load(body_kp_path)
            except Exception as e:
                print(f"Failed loading keypoints for {artist}/{song}: {e}")
                continue

            false_frames = detect_false_positive_frames(
                face_kp, body_kp, distance_threshold
            )

            out_obj = {
                "total_frames": int(min(len(face_kp), len(body_kp))),
                "false_positive_frames": false_frames,
                "num_false_positive_frames": len(false_frames),
                "distance_threshold": distance_threshold,
            }

            # Save per song
            out_path = os.path.join(
                vocal_dir, "face_false_positive_frames.json"
            )
            try:
                with open(out_path, "w") as f:
                    json.dump(out_obj, f, indent=2)
            except Exception as e:
                print(f"Failed writing output for {artist}/{song}: {e}")
                continue

            summary.setdefault(artist, {})[song] = out_obj

    # Global summary
    global_out = os.path.join(DATASET_PATH, "face_false_positive_summary.json")
    with open(global_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {global_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect false positive face detections using nose-face centroid distance"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=120.0,
        help="Pixel distance threshold above which a frame is considered a false positive",
    )
    args = parser.parse_args()

    if DATASET_PATH is None:
        raise ValueError("DATASET_PATH environment variable not set.")

    metadata = load_metadata(DATASET_PATH)
    process(metadata, args.distance_threshold, instrument="vocal")


if __name__ == "__main__":
    main()
