import os
import json
import numpy as np
from tqdm import tqdm
import librosa
from scipy.interpolate import interp1d
from dotenv import load_dotenv
from utils import smooth_keypoints
from .body_parts_map import body_parts_map


class MotionFeatureExtractor:
    def __init__(
        self,
        dataset_dir: str,
        instruments: list,
        artist_filter: str = None,
        fps: int = 30,
        conf_threshold: float = 5.0,
        smooth_win: int = 5,
        smooth_poly: int = 2,
    ):
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.instruments = instruments
        self.artist_filter = artist_filter
        self.fps = fps
        self.conf_threshold = conf_threshold
        self.smooth_win = smooth_win
        self.smooth_poly = smooth_poly
        self.body_parts_map = body_parts_map

    def _compute_speed_accel(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # mask low-confidence
        mask = scores < self.conf_threshold
        mask = np.expand_dims(mask, -1)
        mask = np.broadcast_to(mask, keypoints.shape)
        keypoints = keypoints.copy()
        keypoints[mask] = np.nan

        vel = np.diff(keypoints, axis=0) * self.fps
        speed = np.linalg.norm(vel, axis=-1)
        speed = np.vstack([np.full((1, speed.shape[1]), np.nan), speed])

        acc = np.diff(vel, axis=0) * self.fps
        accel = np.linalg.norm(acc, axis=-1)
        accel = np.vstack([np.full((2, accel.shape[1]), np.nan), accel])

        return speed, accel

    def _summarize(self, speed: np.ndarray, accel: np.ndarray) -> dict:
        summary = {
            "general": {
                "mean_speed": np.nanmean(speed, axis=1).tolist(),
                "mean_accel": np.nanmean(accel, axis=1).tolist(),
            }
        }
        for part, idxs in self.body_parts_map.items():
            summary[part] = {
                "mean_speed": np.nanmean(speed[:, idxs], axis=1).tolist(),
                "mean_accel": np.nanmean(accel[:, idxs], axis=1).tolist(),
            }
        return summary

    def extract(self) -> dict:
        motion_features = {}
        for artist in tqdm(os.listdir(self.dataset_dir), desc="Artists"):
            if self.artist_filter and artist != self.artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_dir, artist)
            if not os.path.isdir(artist_dir) or artist.startswith("."):
                continue
            motion_features.setdefault(artist, {})
            for song in tqdm(os.listdir(artist_dir), desc="Songs", leave=False):
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                motion_features[artist].setdefault(song, {})
                for inst in self.instruments:
                    inst_dir = os.path.join(song_dir, inst)
                    if not os.path.isdir(inst_dir):
                        continue
                    try:
                        kps = np.load(os.path.join(inst_dir, "keypoints.npy"))
                        scs = np.load(os.path.join(inst_dir, "keypoint_scores.npy"))
                        # drop lower body
                        kps = np.delete(kps, np.s_[11:23], axis=1)
                        scs = np.delete(scs, np.s_[11:23], axis=1)
                        kps = smooth_keypoints(kps=kps, smooth_poly=self.smooth_poly, smooth_win=self.smooth_win)
                        speed, accel = self._compute_speed_accel(kps, scs)
                        summary = self._summarize(speed, accel)
                        motion_features[artist][song][inst] = summary
                    except Exception as e:
                        print(f"Motion error {artist}/{song}/{inst}: {e}")
                        motion_features[artist][song][inst] = {}
        # save out
        for artist, songs in motion_features.items():
            for song, insts in songs.items():
                for inst, summary in insts.items():
                    if not summary:
                        continue
                    outp = os.path.join(
                        self.dataset_dir, artist, song, inst, "motion_features_3.json"
                    )
                    with open(outp, "w") as f:
                        json.dump(summary, f, indent=4)
        return motion_features


class AudioFeatureExtractor:
    def __init__(
        self,
        dataset_dir: str,
        instruments: list,
        artist_filter: str = None,
        sr: int = 48000,
        hop_length: int = 1024,
    ):
        load_dotenv()
        self.dataset_dir = dataset_dir
        self.instruments = instruments
        self.artist_filter = artist_filter
        self.sr = sr
        self.hop_length = hop_length

    def _merge(self, paths: list[str]) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(paths[0], sr=self.sr)
        for p in paths[1:]:
            y2, _ = librosa.load(p, sr=self.sr)
            y = (y + y2) / 2.0
        return y, sr

    def _resample(self, feat: np.ndarray, target: int) -> list:
        hop_t = self.hop_length / self.sr
        t_orig = np.arange(len(feat)) * hop_t
        t_tgt = np.linspace(0, t_orig[-1], target)
        f = interp1d(t_orig, feat, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        return f(t_tgt).tolist()

    def extract(self, motion_summary: dict) -> dict:
        audio_features = {}
        for artist in tqdm(os.listdir(self.dataset_dir), desc="Artists"):
            if self.artist_filter and artist != self.artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_dir, artist)
            if not os.path.isdir(artist_dir) or artist.startswith("."):
                continue
            audio_features.setdefault(artist, {})
            for song in tqdm(os.listdir(artist_dir), desc="Songs", leave=False):
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                audio_features[artist].setdefault(song, {})
                for inst in self.instruments:
                    if "general" not in motion_summary[artist][song].get(inst, {}):
                        continue
                    # find .wav files
                    if inst == "mridangam":
                        paths = [
                            os.path.join(song_dir, f)
                            for f in os.listdir(song_dir)
                            if f.lower().endswith(".wav") and "mri" in f.lower()
                        ]
                    else:
                        paths = [
                            os.path.join(song_dir, f)
                            for f in os.listdir(song_dir)
                            if f.lower().endswith(".wav") and inst in f.lower()
                        ]
                    if not paths:
                        continue
                    y, sr = self._merge(paths) if len(paths) > 1 else librosa.load(paths[0], sr=self.sr)
                    rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
                    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
                    tgt = len(motion_summary[artist][song][inst]["general"]["mean_speed"])
                    audio_features[artist][song][inst] = {
                        "rms": self._resample(rms, tgt),
                        "onset_env": self._resample(onset, tgt),
                    }
        # save out
        for artist, songs in audio_features.items():
            for song, insts in songs.items():
                for inst, feats in insts.items():
                    outp = os.path.join(
                        self.dataset_dir, artist, song, inst, "audio_features_2.json"
                    )
                    with open(outp, "w") as f:
                        json.dump(feats, f, indent=4)
        return audio_features


def main():
    load_dotenv()
    ds = os.getenv("DATASET_PATH")
    instruments = ["vocal", "mridangam", "violin"]
    artist = "Aditi Prahalad"

    motion_extractor = MotionFeatureExtractor(
        dataset_dir=ds,
        instruments=instruments,
        artist_filter=artist
    )
    motion_summary = motion_extractor.extract()

    audio_extractor = AudioFeatureExtractor(
        dataset_dir=ds,
        instruments=instruments,
        artist_filter=artist
    )
    audio_extractor.extract(motion_summary)


if __name__ == "__main__":
    main()