import argparse
import os
import json
from dotenv import load_dotenv
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute correlations over the dataset (or a single artist/song)."
    )
    parser.add_argument("--artist", type=str, help="Name of the artist to process")
    parser.add_argument(
        "--song", type=str, help="Name of the song to process (requires --artist)"
    )
    parser.add_argument(
        "--save_strong_windows",
        action="store_true",
        help="Save strong correlation windows to disk",
    )
    args = parser.parse_args()
    if args.song and not args.artist:
        parser.error("--song requires --artist")
    return args


def load_features(dataset_path, artist_filter=None, song_filter=None):
    instruments = ["vocal", "mridangam", "violin"]
    motion_features = {}
    audio_features = {}

    for artist in tqdm(os.listdir(dataset_path), desc="Artists"):
        if artist_filter and artist != artist_filter:
            continue
        artist_dir = os.path.join(dataset_path, artist)
        if not os.path.isdir(artist_dir) or artist.startswith("."):
            continue

        motion_features.setdefault(artist, {})
        audio_features.setdefault(artist, {})

        for song in tqdm(os.listdir(artist_dir), desc="Songs", leave=False):
            if song_filter and song != song_filter:
                continue
            song_dir = os.path.join(artist_dir, song)
            if not os.path.isdir(song_dir) or song.startswith("."):
                continue

            motion_features[artist].setdefault(song, {})
            audio_features[artist].setdefault(song, {})

            for instr in instruments:
                inst_dir = os.path.join(song_dir, instr)
                if not os.path.isdir(inst_dir):
                    continue

                try:
                    with open(os.path.join(inst_dir, "motion_features.json")) as f:
                        motion_features[artist][song][instr] = json.load(f)
                    with open(os.path.join(inst_dir, "audio_features.json")) as f:
                        audio_features[artist][song][instr] = json.load(f)

                    print(f"Loaded features for {artist}/{song}/{instr}")
                except FileNotFoundError:
                    print(f"Missing feature files for {artist}/{song}/{instr}, skipping")
    return motion_features, audio_features


def safe_pearsonr(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])


def compute_global_correlations(motion_features, audio_features,
                                corr_thresh=0.25, pval_thresh=0.05):
    speed_corrs = []
    accel_corrs = []

    for A in motion_features:
        for S in motion_features[A]:
            for instr in motion_features[A][S]:
                for part in motion_features[A][S][instr]:
                    x = motion_features[A][S][instr][part]["mean_speed"]
                    y = audio_features[A][S][instr]["onset_env"]
                    c1, p1 = safe_pearsonr(x, y)
                    if c1 > corr_thresh and p1 < pval_thresh:
                        speed_corrs.append(dict(artist=A, song=S,
                                                instrument=instr,
                                                body_part=part,
                                                corr=float(c1)))
                    x2 = motion_features[A][S][instr][part]["mean_acceleration"]
                    c2, p2 = safe_pearsonr(x2, y)
                    if c2 > corr_thresh and p2 < pval_thresh:
                        accel_corrs.append(dict(artist=A, song=S,
                                                instrument=instr,
                                                body_part=part,
                                                corr=float(c2)))
    return speed_corrs, accel_corrs


def sliding_correlation(signal1, signal2, window_size, step_size, threshold):
    assert len(signal1) == len(signal2)
    corrs, pvals, times = [], [], []
    for start in range(0, len(signal1) - window_size + 1, step_size):
        seg1 = signal1[start:start + window_size]
        seg2 = signal2[start:start + window_size]
        c, p = safe_pearsonr(seg1, seg2)
        corrs.append(c); pvals.append(p)
        times.append(start / 30)

    return [
        (t, c) for t, c, p in zip(times, corrs, pvals)
        if not np.isnan(c) and not np.isnan(p) and abs(c) > threshold and p < 0.05
    ]


def compute_strong_windows(motion_features, audio_features, dataset_path,
                           fps=30, win_dur=0.5, step_dur=0.1, thresh=0.75):
    win = int(win_dur * fps)
    step = int(step_dur * fps)

    for A in tqdm(motion_features, desc="Artists"):
        for S in tqdm(motion_features[A], desc="Songs", leave=False):
            for instr in motion_features[A][S]:
                file_name = f"{str(thresh).replace('.', '')}_correlation_{str(win_dur).replace('.', '')}s_windows.json"
                output_dir = os.path.join(dataset_path, A, S, instr)
                output_path = os.path.join(output_dir, file_name)
                if os.path.exists(output_path):
                    print(f"Strong windows already computed for {A}/{S}/{instr}, skipping")
                    continue
                strong_subset = {}
                for part in motion_features[A][S][instr]:
                    y = audio_features[A][S][instr]["onset_env"]
                    x_sp = motion_features[A][S][instr][part]["mean_speed"]
                    speed_strong_windows = sliding_correlation(x_sp, y, win, step, thresh)
                    x_ac = motion_features[A][S][instr][part]["mean_acceleration"]
                    acceleration_strong_windows = sliding_correlation(x_ac, y, win, step, thresh)
                    strong_subset[part] = {"speed": speed_strong_windows,
                                                 "accel": acceleration_strong_windows}
                with open(output_path, "w") as f:
                    json.dump(strong_subset, f, indent=4)
                print(f"Saved strong windows for {A}/{S}/{instr} to {output_path}")


def compute_number_of_strong_windows(strong_windows):
    num_windows = 0
    for A in strong_windows:
        for S in strong_windows[A]:
            for instr in strong_windows[A][S]:
                for part in strong_windows[A][S][instr]:
                    num_windows += len(strong_windows[A][S][instr][part]["speed"]) + \
                                   len(strong_windows[A][S][instr][part]["accel"])
    return num_windows


def main():
    args = parse_args()
    load_dotenv()
    path = os.getenv("DATASET_PATH")
    local_correlation_threshold = 0.0
    window_duration = 0.5
    motion_features, audio_features = load_features(path, args.artist, args.song)
    compute_strong_windows(motion_features, audio_features, path, win_dur=window_duration, thresh=local_correlation_threshold)

if __name__ == "__main__":
    main()