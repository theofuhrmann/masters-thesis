import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from tools.body_parts_map import body_parts_map  # noqa: E402
from tools.utils import smooth_keypoints  # noqa: E402

NUMBER_OF_KEYPOINTS = 133


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
        motion_features_filename: str = None,
        audio_features_filename: str = None,
        pca_components_filename: str = None,
        correlation_windows_filename: str = None,
    ):
        load_dotenv()
        self.dataset_path = dataset_path
        self.artist = artist
        self.song = song
        self.instruments = instruments
        self.motion_features_filename = motion_features_filename
        self.audio_features_filename = audio_features_filename
        self.pca_components_filename = pca_components_filename
        self.correlation_windows_filename = correlation_windows_filename

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
        self.dataset_metadata = {}
        self.pca_components = {}
        self.pca_explained_variance = {}
        self.keypoints = {}
        self.keypoint_scores = {}
        self.correlation_windows = {}
        self.motion_features = {}
        self.audio_features = {}

        # load all data
        self._load_dataset_metadata()
        self._load_pose_data()
        self._preprocess_keypoints()
        if self.motion_features_filename and self.audio_features_filename:
            self._load_motion_audio()
            self._normalize_features()
        if self.pca_components_filename:
            self._load_pca()

    def _load_dataset_metadata(self):
        metadata_file = os.path.join(
            self.dataset_path, "dataset_metadata.json"
        )
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}"
            )
        with open(metadata_file) as f:
            self.dataset_metadata = json.load(f)[self.artist][self.song]

    def _load_pose_data(self):
        for instrument in self.instruments:
            body_keypoints_file = os.path.join(
                self.base_directory, instrument, "keypoints.npy"
            )
            body_keypoint_scores_file = os.path.join(
                self.base_directory, instrument, "keypoint_scores.npy"
            )
            self.keypoints[instrument] = {
                "body": np.load(body_keypoints_file, allow_pickle=True),
            }
            if instrument == "vocal":
                self.keypoints[instrument]["face"] = None
                face_keypoints_file = os.path.join(
                    self.base_directory, instrument, "face_keypoints.npy"
                )
                if os.path.exists(face_keypoints_file):
                    self.keypoints[instrument]["face"] = np.load(
                        face_keypoints_file, allow_pickle=True
                    ).transpose((0, 2, 1))

            self.keypoint_scores[instrument] = np.load(
                body_keypoint_scores_file, allow_pickle=True
            )
            self.correlation_windows[instrument] = {}
            if self.correlation_windows_filename:
                correlation_windows_file = os.path.join(
                    self.base_directory,
                    instrument,
                    self.correlation_windows_filename,
                )
                if os.path.exists(correlation_windows_file):
                    with open(correlation_windows_file) as f:
                        self.correlation_windows[instrument] = json.load(f)
                else:
                    print(
                        f"Warning: Correlation windows file not found for {instrument} in {self.base_directory}"
                    )

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
            raw_body_keypoints = self.keypoints[instrument]["body"]
            filled_body_keypoints = self._fill_nan_frames(raw_body_keypoints)
            self.keypoints[instrument]["body"] = smooth_keypoints(
                keypoints=filled_body_keypoints,
                smooth_poly=self.smoothing_poly_order,
                smooth_win=self.window_size,
                mode=self.smoothing_mode,
            )
            """
            if instrument == "vocal":
                raw_face_keypoints = self.keypoints[instrument]["face"]
                filled_face_keypoints = self._fill_nan_frames(
                    raw_face_keypoints
                )
                self.keypoints[instrument]["face"] = smooth_keypoints(
                    keypoints=filled_face_keypoints,
                    smooth_poly=self.smoothing_poly_order,
                    smooth_win=self.window_size,
                    mode=self.smoothing_mode,
                )
            """

    def _load_motion_audio(self):
        for instrument in self.instruments:
            motion_features_file = os.path.join(
                self.base_directory, instrument, self.motion_features_filename
            )
            audio_features_file = os.path.join(
                self.base_directory, instrument, self.audio_features_filename
            )
            with open(motion_features_file) as f:
                self.motion_features[instrument] = json.load(f)
            with open(audio_features_file) as f:
                self.audio_features[instrument] = json.load(f)

    def _load_pca(self):
        for instrument in self.instruments:
            pca_file = os.path.join(
                self.base_directory, instrument, self.pca_components_filename
            )
            with open(pca_file) as f:
                pca_data = json.load(f)
            self.pca_components[instrument] = np.array(pca_data["components"])
            self.pca_explained_variance[instrument] = np.array(
                pca_data["explained_variance_ratio"]
            )

    def _get_keypoint_pca_contributions(
        self, instrument, occluded_keypoints, top_n=None
    ):
        visible_keypoints = [
            i
            for i in list(range(NUMBER_OF_KEYPOINTS))
            if i not in occluded_keypoints
        ]
        top_component = self.pca_components[instrument][0]
        keypoint_importance = []
        for i in range(len(top_component) // 2):
            speed_component = top_component[i]
            acceleration_component = top_component[i + len(top_component) // 2]
            contribution = np.sum(
                np.abs(speed_component + acceleration_component)
            )
            original_kp_index = visible_keypoints[i]
            keypoint_importance.append((original_kp_index, contribution))

        keypoint_importance.sort(key=lambda x: x[1], reverse=True)
        return keypoint_importance[:top_n] if top_n else keypoint_importance

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
        add_audio: bool = True,
    ):
        video_capture = cv2.VideoCapture(self.video_file)
        assert video_capture.isOpened(), "Cannot open video"
        frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing frames")
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
                    if frame_index < len(self.keypoints[instrument]["body"]):
                        body_keypoints = self.keypoints[instrument]["body"][
                            frame_index
                        ]
                        body_keypoint_scores = self.keypoint_scores[
                            instrument
                        ][frame_index]
                        face_keypoints = (
                            self.keypoints[instrument]["face"][frame_index]
                            if instrument == "vocal"
                            and self.keypoints[instrument]["face"] is not None
                            else None
                        )
                        # find correlated parts
                        correlated_parts = []
                        if self.correlation_windows[instrument]:
                            for body_part, types in self.correlation_windows[
                                instrument
                            ].items():
                                for win_list in types.values():
                                    for window, value in win_list:
                                        if (
                                            window * frames_per_second
                                            <= frame_index
                                            < (window + 2) * frames_per_second
                                        ):
                                            correlated_parts.append(
                                                (body_part, value)
                                            )
                        # draw skeleton & features
                        self._draw_skeleton(
                            frame,
                            body_keypoints,
                            body_keypoint_scores,
                            instrument,
                            face_keypoints,
                            correlated_parts,
                            confidence_threshold,
                        )
                        if (
                            self.motion_features_filename
                            and self.audio_features_filename
                        ):
                            self._draw_features(frame, instrument, frame_index)
                video_writer.write(frame)
            frame_index += 1
            pbar.update(1)

        pbar.close()
        video_capture.release()
        video_writer.release()

        if add_audio:
            self._add_audio(temp_video_file, output_file, start_time, end_time)
            os.remove(temp_video_file)
        else:
            shutil.move(temp_video_file, output_file)

    def _draw_skeleton(
        self,
        frame,
        body_keypoints,
        body_keypoint_scores,
        instrument,
        face_keypoints=None,
        correlated_parts=None,
        confidence_threshold=5,
        show_lower_body=False,
        show_occluded_parts=False,
    ):
        occluded_keypoints = []
        occluded_keypoints.extend(body_parts_map["face"])
        occluded_keypoints.extend(body_parts_map["head"])

        if not show_lower_body:
            occluded_keypoints.extend(body_parts_map["left_leg"])
            occluded_keypoints.extend(body_parts_map["right_leg"])
            occluded_keypoints.extend(body_parts_map["left_foot"])
            occluded_keypoints.extend(body_parts_map["right_foot"])
        if not show_occluded_parts:
            if instrument == self.dataset_metadata["layout"][0]:
                occluded_keypoints.extend(body_parts_map["left_arm"])
                occluded_keypoints.extend(body_parts_map["left_hand"])
            elif instrument == self.dataset_metadata["layout"][-1]:
                occluded_keypoints.extend(body_parts_map["right_arm"])
                occluded_keypoints.extend(body_parts_map["right_hand"])

        for skeleton_start, skeleton_end in self._skeleton():
            if (
                body_keypoint_scores[skeleton_start] > confidence_threshold
                and body_keypoint_scores[skeleton_end] > confidence_threshold
                and skeleton_start not in occluded_keypoints
                and skeleton_end not in occluded_keypoints
            ):
                x1, y1 = body_keypoints[skeleton_start]
                x2, y2 = body_keypoints[skeleton_end]
                correlation_value = 0
                if correlated_parts:
                    for part, value in correlated_parts:
                        if "general" == part or (
                            skeleton_start in body_parts_map[part]
                            and skeleton_end in body_parts_map[part]
                        ):
                            correlation_value = value
                            break

                base_color = self._colors()[instrument]
                b, g, r = base_color
                if correlation_value > 0:
                    b = int(min(b + (255 - b) * correlation_value, 255))
                    g = int(min(g + (255 - g) * correlation_value, 255))
                    r = int(min(r + (255 - r) * correlation_value, 255))
                else:
                    b = int(max(b + b * correlation_value, 0))
                    g = int(max(g + g * correlation_value, 0))
                    r = int(max(r + r * correlation_value, 0))
                color = (b, g, r)

                cv2.line(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                )

        if self.pca_components and instrument in self.pca_components:
            contributions = self._get_keypoint_pca_contributions(
                instrument, occluded_keypoints
            )
            max_contribution = (
                max(c[1] for c in contributions) if contributions else 1
            )
            for i, (x, y) in enumerate(body_keypoints):
                if (
                    body_keypoint_scores[i] > confidence_threshold
                    and i not in occluded_keypoints
                ):
                    contribution_tuple = next(
                        (c for c in contributions if c[0] == i), None
                    )
                    contribution = (
                        contribution_tuple[1] if contribution_tuple else 0
                    )

                    intensity = (
                        int(255 * (contribution / max_contribution))
                        if max_contribution > 0
                        else 255
                    )
                    color = (intensity, intensity, intensity)
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        else:
            for i, (x, y) in enumerate(body_keypoints):
                if (
                    body_keypoint_scores[i] > confidence_threshold
                    and i not in occluded_keypoints
                ):
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)

        if instrument == "vocal" and face_keypoints is not None:
            for i, (x, y, _) in enumerate(face_keypoints):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 0), -1)

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

    def _add_audio(
        self, temp_video_file, output_video_file, start_time, end_time
    ):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            self.video_file,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-vn",
            "-acodec",
            "aac",
            "-strict",
            "experimental",
            "temp_audio.aac",
        ]
        subprocess.run(command, check=True)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_file,
            "-i",
            "temp_audio.aac",
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
        os.remove("temp_audio.aac")
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
            # left hand
            (91, 92),
            (92, 93),
            (93, 94),
            (94, 95),
            (91, 96),
            (96, 97),
            (97, 98),
            (98, 99),
            (91, 100),
            (100, 101),
            (101, 102),
            (102, 103),
            (91, 104),
            (104, 105),
            (105, 106),
            (106, 107),
            (91, 108),
            (108, 109),
            (109, 110),
            (110, 111),
            # right hand
            (112, 113),
            (113, 114),
            (114, 115),
            (115, 116),
            (112, 117),
            (117, 118),
            (118, 119),
            (119, 120),
            (112, 121),
            (121, 122),
            (122, 123),
            (123, 124),
            (112, 125),
            (125, 126),
            (126, 127),
            (127, 128),
            (112, 129),
            (129, 130),
            (130, 131),
            (131, 132),
        ]

    @staticmethod
    def _colors():
        return {
            "vocal": (0, 255, 0),
            "violin": (255, 0, 0),
            "mridangam": (0, 0, 255),
        }
