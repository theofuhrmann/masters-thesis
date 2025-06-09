import json
import os

import librosa
import numpy as np
from dotenv import load_dotenv
from scipy.interpolate import interp1d
from tqdm import tqdm


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
        f = interp1d(
            t_orig,
            feat,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return f(t_tgt).tolist()

    def extract(self) -> dict:
        audio_features = {}
        for artist in tqdm(os.listdir(self.dataset_dir), desc="Artists"):
            if self.artist_filter and artist != self.artist_filter:
                continue
            artist_dir = os.path.join(self.dataset_dir, artist)
            if not os.path.isdir(artist_dir) or artist.startswith("."):
                continue
            audio_features.setdefault(artist, {})
            for song in tqdm(
                os.listdir(artist_dir), desc="Songs", leave=False
            ):
                song_dir = os.path.join(artist_dir, song)
                if not os.path.isdir(song_dir) or song.startswith("."):
                    continue
                audio_features[artist].setdefault(song, {})
                for inst in self.instruments:
                    try:
                        with open(
                            os.path.join(
                                song_dir, inst, "motion_features.json"
                            ),
                            "r",
                        ) as f:
                            motion_features = json.load(f)
                    except FileNotFoundError:
                        print(
                            f"Skipping {artist}/{song}/{inst}: motion features not found."
                        )
                        continue
                    # find .wav files
                    if inst == "mridangam":
                        paths = [
                            os.path.join(song_dir, f)
                            for f in os.listdir(song_dir)
                            if f.lower().endswith(".wav")
                            and "mri" in f.lower()
                        ]
                    else:
                        paths = [
                            os.path.join(song_dir, f)
                            for f in os.listdir(song_dir)
                            if f.lower().endswith(".wav") and inst in f.lower()
                        ]
                    if not paths:
                        continue
                    y, sr = (
                        self._merge(paths)
                        if len(paths) > 1
                        else librosa.load(paths[0], sr=self.sr)
                    )
                    rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[
                        0
                    ]
                    onset = librosa.onset.onset_strength(
                        y=y, sr=sr, hop_length=self.hop_length
                    )
                    target_frames = len(
                        motion_features["general"]["mean_speed"]
                    )
                    audio_features[artist][song][inst] = {
                        "rms": self._resample(rms, target_frames),
                        "onset_env": self._resample(onset, target_frames),
                    }
        # save out
        for artist, songs in audio_features.items():
            for song, insts in songs.items():
                for inst, feats in insts.items():
                    outp = os.path.join(
                        self.dataset_dir,
                        artist,
                        song,
                        inst,
                        "audio_features.json",
                    )
                    with open(outp, "w") as f:
                        json.dump(feats, f, indent=4)
        return audio_features
