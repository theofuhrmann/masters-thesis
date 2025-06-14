import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402


def extract_audio_chunk(audio_path, start, duration, out_dir, instrument=None):
    if instrument and instrument == "mridangam":
        left_path = audio_path.replace("mridangam", "mridangam-left")
        right_path = audio_path.replace("mridangam", "mridangam-right")
        y_left, sr = librosa.load(left_path, offset=start, duration=duration)
        y_right, _ = librosa.load(right_path, offset=start, duration=duration)
        y = (y_left + y_right) / 2.0
    else:
        y, sr = librosa.load(audio_path, offset=start, duration=duration)
    audio_out = os.path.join(
        out_dir,
        f"audio_{instrument + '_' if instrument else ''}{start:.1f}_{duration:.1f}.wav",
    )
    sf.write(audio_out, y, sr)


def extract_face_pose_keypoints(
    keypoints_path,
    start,
    duration,
    fps,
    out_path,
):
    keypoints = np.load(keypoints_path)

    start_frame = int(start * fps)
    end_frame = int((start + duration) * fps)
    if end_frame > len(keypoints):
        end_frame = len(keypoints)

    np.save(
        os.path.join(
            out_path, f"keypoints_face_{start:.1f}_{duration:.1f}.npy"
        ),
        keypoints[start_frame:end_frame],
    )


def extract_body_pose_keypoints(
    keypoints_path,
    start,
    duration,
    fps,
    out_path,
):
    keypoints = np.load(keypoints_path)
    keypoints = np.transpose(keypoints, (0, 2, 1))
    body_parts_to_remove = [
        "left_leg",
        "right_leg",
        "left_foot",
        "right_foot",
        "face",
    ]

    keypoints_to_remove = []
    for part in body_parts_to_remove:
        keypoints_to_remove.extend(body_parts_map[part])

    keypoints_to_remove = [i for i in keypoints_to_remove if i not in [11, 12]]
    body_keypoints = np.delete(keypoints, keypoints_to_remove, axis=2)

    start_frame = int(start * fps)
    end_frame = int((start + duration) * fps)
    if end_frame > len(keypoints):
        end_frame = len(keypoints)

    np.save(
        os.path.join(
            out_path, f"keypoints_body_{start:.1f}_{duration:.1f}.npy"
        ),
        body_keypoints[start_frame:end_frame],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artist")
    p.add_argument("song")
    p.add_argument(
        "--start", type=float, default=0.0, help="start time in seconds"
    )
    p.add_argument(
        "--duration", type=float, default=4.0, help="chunk duration in seconds"
    )
    p.add_argument(
        "--instrument",
        type=str,
        default="vocal",
        choices=["vocal", "violin", "mridangam"],
        help="instrument to process (default: vocal)",
    )
    args = p.parse_args()

    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")
    fps = int(os.getenv("FPS"))

    out_dir = os.path.join("chunks", args.artist, args.song)
    os.makedirs(out_dir, exist_ok=True)

    audio_path = os.path.join(
        dataset_path, args.artist, args.song, f"{args.song}.wav"
    )

    target_audio_path = os.path.join(
        dataset_path,
        args.artist,
        args.song,
        f"{args.song}.multitrack-{args.instrument}.wav",
    )

    extract_audio_chunk(audio_path, args.start, args.duration, out_dir)
    extract_audio_chunk(
        target_audio_path, args.start, args.duration, out_dir, args.instrument
    )

    keypoints_path = os.path.join(
        dataset_path, args.artist, args.song, args.instrument, "keypoints.npy"
    )

    extract_body_pose_keypoints(
        keypoints_path, args.start, args.duration, fps, out_dir
    )

    if args.instrument == "vocal":
        keypoints_path = os.path.join(
            dataset_path,
            args.artist,
            args.song,
            args.instrument,
            "face_keypoints.npy",
        )
        extract_face_pose_keypoints(
            keypoints_path, args.start, args.duration, fps, out_dir
        )


if __name__ == "__main__":
    main()
