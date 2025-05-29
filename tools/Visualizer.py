import json
import os
import subprocess
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402
from tools.utils import smooth_keypoints  # noqa: E402


class GestureVisualizer:
    def __init__(
        self,
        dataset_path: str,
        artist: str,
        song: str,
        instruments: list,
        window_size: int = 5,
        smoothing_mode: str = "nearest",
        smoothing_poly_order: int = 2,
    ):
        load_dotenv()
        self.dataset_path = dataset_path
        self.artist = artist
        self.song = song
        self.instruments = instruments

        # filepaths
        self.base_directory = os.path.join(dataset_path, artist, song)
        mov_files = [
            f
            for f in os.listdir(self.base_directory)
            if f.lower().endswith(".mov")
        ]
        assert (
            len(mov_files) == 1
        ), f"Expected one .mov file in {self.base_directory}, found {len(mov_files)}"
        self.video_file = os.path.join(self.base_directory, mov_files[0])

        # params for smoothing
        self.window_size = window_size
        self.smoothing_mode = smoothing_mode
        self.smoothing_poly_order = smoothing_poly_order

        # data holders
        self.keypoints = {}
        self.keypoint_scores = {}
        self.correlation_windows = {}
        self.motion_features = {}
        self.audio_features = {}

        # load all data
        self._load_pose_data()
        self._preprocess_keypoints()
        self._load_motion_audio()
        self._normalize_features()

    def _load_pose_data(self):
        for instrument in self.instruments:
            keypoints_file = os.path.join(
                self.base_directory, instrument, "keypoints.npy"
            )
            keypoint_scores_file = os.path.join(
                self.base_directory, instrument, "keypoint_scores.npy"
            )
            correlation_windows_file = os.path.join(
                self.base_directory,
                instrument,
                "strong_correlation_05s_windows.json",
            )
            self.keypoints[instrument] = np.load(
                keypoints_file, allow_pickle=True
            )
            self.keypoint_scores[instrument] = np.load(
                keypoint_scores_file, allow_pickle=True
            )
            if not os.path.exists(correlation_windows_file):
                print(
                    f"Warning: Correlation windows file not found for {instrument} in {self.base_directory}"
                )
                self.correlation_windows[instrument] = {}
            else:
                with open(correlation_windows_file) as f:
                    self.correlation_windows[instrument] = json.load(f)

    def _fill_nan_frames(self, data_array: np.ndarray) -> np.ndarray:
        output_array = data_array.copy()
        all_nan = np.isnan(output_array[..., 0]).all(axis=1) & np.isnan(
            output_array[..., 1]
        ).all(axis=1)
        valid = np.nonzero(~all_nan)[0]
        for i in np.nonzero(all_nan)[0]:
            previous_valid = valid[valid < i]
            next_valid = valid[valid > i]
            if previous_valid.size and next_valid.size:
                output_array[i] = (
                    output_array[previous_valid[-1]]
                    + output_array[next_valid[0]]
                ) * 0.5
            elif previous_valid.size:
                output_array[i] = output_array[previous_valid[-1]]
            elif next_valid.size:
                output_array[i] = output_array[next_valid[0]]
        return output_array

    def _preprocess_keypoints(self):
        for instrument in self.instruments:
            raw_keypoints = self.keypoints[instrument]
            filled_keypoints = self._fill_nan_frames(raw_keypoints)
            self.keypoints[instrument] = smooth_keypoints(
                keypoints=filled_keypoints,
                smooth_poly=self.smoothing_poly_order,
                smooth_win=self.window_size,
                mode=self.smoothing_mode,
            )

    def _load_motion_audio(self):
        for instrument in self.instruments:
            motion_features_file = os.path.join(
                self.base_directory, instrument, "motion_features.json"
            )
            audio_features_file = os.path.join(
                self.base_directory, instrument, "audio_features.json"
            )
            with open(motion_features_file) as f:
                self.motion_features[instrument] = json.load(f)
            with open(audio_features_file) as f:
                self.audio_features[instrument] = json.load(f)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        percentile_5, percentile_95 = np.percentile(data, [5, 95])
        clipped_data = np.clip(data, percentile_5, percentile_95)
        min_val, max_val = clipped_data.min(), clipped_data.max()
        return (
            (clipped_data - min_val) / (max_val - min_val)
            if max_val > min_val
            else clipped_data
        )

    def _normalize_features(self):
        self.onset_strength = {}
        self.rms_energy = {}
        self.mean_speed = {}
        self.mean_acceleration = {}
        for instrument in self.instruments:
            self.onset_strength[instrument] = self._normalize(
                np.array(self.audio_features[instrument]["onset_env"])
            )
            self.rms_energy[instrument] = self._normalize(
                np.array(self.audio_features[instrument]["rms"])
            )
            speed = np.nan_to_num(
                self.motion_features[instrument]["general"]["mean_speed"],
                nan=0.0,
            )
            acceleration = np.nan_to_num(
                self.motion_features[instrument]["general"][
                    "mean_acceleration"
                ],
                nan=0.0,
            )
            self.mean_speed[instrument] = self._normalize(speed)
            self.mean_acceleration[instrument] = self._normalize(acceleration)

    def visualize(
        self,
        output_file: str,
        start_time: float = 0,
        end_time: float = None,
        confidence_threshold: float = 5,
    ):
        video_capture = cv2.VideoCapture(self.video_file)
        assert video_capture.isOpened(), "Cannot open video"
        frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = end_time or (total_frames / frames_per_second)
        start_frame = int(start_time * frames_per_second)
        end_frame = int(end_time * frames_per_second)

        temp_video_file = "temp_novid.mp4"
        video_writer = cv2.VideoWriter(
            temp_video_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames_per_second,
            (frame_width, frame_height),
        )
        frame_index = 0
        while True:
            ret, frame = video_capture.read()
            if not ret or frame_index >= end_frame:
                break
            if frame_index >= start_frame:
                for instrument in self.instruments:
                    if frame_index < len(self.keypoints[instrument]):
                        keypoints = self.keypoints[instrument][frame_index]
                        keypoint_scores = self.keypoint_scores[instrument][
                            frame_index
                        ]
                        # find correlated parts
                        correlated_parts = []
                        for body_part, types in self.correlation_windows[
                            instrument
                        ].items():
                            for win_list in types.values():
                                for w in win_list:
                                    if (
                                        w * frames_per_second
                                        <= frame_index
                                        < (w + 2) * frames_per_second
                                    ):
                                        correlated_parts.append(body_part)
                        # draw skeleton & features
                        self._draw_skeleton(
                            frame,
                            keypoints,
                            keypoint_scores,
                            instrument,
                            correlated_parts,
                            confidence_threshold,
                        )
                        self._draw_features(frame, instrument, frame_index)
                video_writer.write(frame)
            frame_index += 1
        video_capture.release()
        video_writer.release()
        self._add_audio(temp_video_file, output_file)
        os.remove(temp_video_file)

    def _draw_skeleton(
        self,
        frame,
        keypoints,
        keypoint_scores,
        instrument,
        correlated_parts=None,
        confidence_threshold=5,
        show_lower_body=False,
    ):
        for i, (x, y) in enumerate(keypoints):
            if keypoint_scores[i] > confidence_threshold and (
                show_lower_body or i < 11 or i > 23
            ):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
        for skeleton_start, skeleton_end in self._skeleton():
            if (
                keypoint_scores[skeleton_start] > confidence_threshold
                and keypoint_scores[skeleton_end] > confidence_threshold
                and (
                    show_lower_body
                    or (skeleton_start < 11 and skeleton_end < 11)
                    or (skeleton_start > 23 and skeleton_end > 23)
                )
            ):
                x1, y1 = keypoints[skeleton_start]
                x2, y2 = keypoints[skeleton_end]
                is_correlated = correlated_parts and (
                    "general" in correlated_parts
                    or any(
                        skeleton_start in body_parts_map[part]
                        and skeleton_end in body_parts_map[part]
                        for part in correlated_parts
                    )
                )
                color = (
                    (0, 255, 255)
                    if is_correlated
                    else self._colors()[instrument]
                )
                cv2.line(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )

    def _draw_features(self, frame, instrument, frame_index):
        features = [
            ("Onset", self.onset_strength[instrument][frame_index]),
            ("RMS", self.rms_energy[instrument][frame_index]),
            ("Speed", self.mean_speed[instrument][frame_index]),
            ("Accel", self.mean_acceleration[instrument][frame_index]),
        ]
        frame_height, frame_width = frame.shape[:2]
        for i, (label, value) in enumerate(features):
            text = f"{label}:{value:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            x = 20 + (frame_width - 40 - text_width) * (instrument == "violin")
            x = (frame_width // 2 - text_width // 2) * (
                instrument == "vocal"
            ) or x
            y = frame_height - 20 - i * 30
            cv2.rectangle(
                frame,
                (x - 5, y - text_height - 5),
                (x + text_width + 5, y + 5),
                (0, int(255 * value), int(255 * (1 - value))),
                -1,
            )
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    def _add_audio(self, temp_video_file, output_video_file):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_file,
            "-i",
            self.video_file,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            output_video_file,
        ]
        subprocess.run(command, check=True)
        print(f"→ saved {output_video_file}")

    @staticmethod
    def _skeleton():
        return [
            (0, 1),
            (1, 2),
            (0, 3),
            (0, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (11, 12),
            (5, 11),
            (6, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    @staticmethod
    def _colors():
        return {
            "vocal": (0, 255, 0),
            "violin": (255, 0, 0),
            "mridangam": (0, 0, 255),
        }


def main():
    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")
    artist = "Abhiram Bode"
    song = "Lekanna ninnu"
    gesture_visualizer = GestureVisualizer(
        dataset_path=dataset_path,
        artist=artist,
        song=song,
        instruments=["violin", "vocal", "mridangam"],
        window_size=5,
    )
    gesture_visualizer.visualize(
        output_file=os.path.join(
            dataset_path, artist, song, "visualized_latest.mp4"
        ),
        confidence_threshold=3,
    )


if __name__ == "__main__":
    main()
