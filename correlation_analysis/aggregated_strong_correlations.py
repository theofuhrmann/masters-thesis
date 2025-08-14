import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm
from utils import sliding_correlation

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
instruments = os.getenv("INSTRUMENTS").split(",")


def filter_metadata(dataset_metadata: dict, instrument: str):
    dataset_metadata = dataset_metadata.copy()
    for artist, songs in list(dataset_metadata.items()):
        if args.artist and artist != args.artist:
            del dataset_metadata[artist]
            continue
        for song in list(songs.keys()):
            if args.song and song != args.song:
                del dataset_metadata[artist][song]
                continue
            if (
                dataset_metadata[artist][song]["layout"] is None
                or None in dataset_metadata[artist][song]["layout"]
            ):
                print(f"Removing {artist} - {song}: invalid layout")
                del dataset_metadata[artist][song]
                continue
            if dataset_metadata[artist][song]["moving_camera"]:
                print(f"Removing {artist} - {song}: moving camera")
                del dataset_metadata[artist][song]
                continue
            if instrument == "vocal":
                if sorted(dataset_metadata[artist][song]["layout"]) != sorted(
                    instruments
                ):
                    print(f"Removing {artist} - {song}: layout mismatch")
                    del dataset_metadata[artist][song]
            elif instrument == "violin":
                if dataset_metadata[artist][song]["layout"] != [
                    "violin",
                    "vocal",
                    "mridangam",
                ]:
                    print(f"Removing {artist} - {song}: layout mismatch")
                    del dataset_metadata[artist][song]
                elif (
                    dataset_metadata[artist][song]["correct_body_detection"]
                    is not True
                ):
                    print(f"Removing {artist} - {song}: no body detected")
                    del dataset_metadata[artist][song]

        if len(dataset_metadata[artist]) == 0:
            del dataset_metadata[artist]

    return dataset_metadata


def load_features(
    dataset_metadata: dict, instrument: str, motion_file: str = "instrument"
):
    """Load motion & audio features.

    motion_file options:
      - 'instrument': only instrument-specific motion features (raw names)
      - 'generic': only generic motion features (prefixed 'general::')
      - 'both': merge both sets (instrument features prefixed 'instrument::', generic prefixed 'general::')
    """
    motion_features = {}
    audio_features = {}

    for artist, songs in tqdm(
        dataset_metadata.items(), desc="Loading Artists"
    ):
        motion_features[artist] = {}
        audio_features[artist] = {}

        for song in tqdm(songs, desc="Loading Songs", leave=False):
            instrument_dir = os.path.join(
                dataset_path, artist, song, instrument
            )

            features_dict = {}
            had_any_motion = False
            # Attempt to load instrument-specific features if requested
            if motion_file in ("instrument", "both"):
                inst_file = os.path.join(
                    instrument_dir, f"{instrument}_motion_features.json"
                )
                if os.path.isfile(inst_file):
                    try:
                        with open(inst_file, "r") as f:
                            inst_motion_data = json.load(f)
                        if motion_file == "instrument":
                            # keep original names
                            for k, v in inst_motion_data.items():
                                features_dict[k] = v
                        else:  # both -> prefix
                            for k, v in inst_motion_data.items():
                                features_dict[f"instrument::{k}"] = v
                        had_any_motion = True
                    except Exception as e:
                        print(f"Failed reading {inst_file}: {e}")
                else:
                    if motion_file == "instrument":
                        print(
                            f"Instrument motion file missing for {artist}/{song}/{instrument}"
                        )
            # Attempt to load generic features if requested
            if motion_file in ("generic", "both"):
                gen_file = os.path.join(instrument_dir, "motion_features.json")
                if os.path.isfile(gen_file):
                    try:
                        with open(gen_file, "r") as f:
                            gen_motion_data = json.load(f)
                        general_block = gen_motion_data.get("general", {})
                        prefix = "general::" if motion_file == "both" else ""
                        for k, v in general_block.items():
                            features_dict[f"{prefix}{k}"] = v
                        had_any_motion = True
                    except Exception as e:
                        print(f"Failed reading {gen_file}: {e}")
                else:
                    if motion_file == "generic":
                        print(
                            f"Generic motion file missing for {artist}/{song}/{instrument}"
                        )

            if not had_any_motion:
                # Skip if neither file was available
                continue

            # Assign motion features collected
            motion_features[artist][song] = {instrument: features_dict}

            # Audio features (common path)
            audio_file = os.path.join(instrument_dir, "audio_features.json")
            try:
                with open(audio_file, "r") as f:
                    audio_features[artist][song] = {instrument: json.load(f)}
            except FileNotFoundError:
                print(
                    f"Audio feature file missing for {artist}/{song}/{instrument}. Skipping entry."
                )
                # Remove motion entry if audio missing to keep parity
                del motion_features[artist][song]
                if len(motion_features[artist]) == 0:
                    del motion_features[artist]
                continue

    return motion_features, audio_features


def calculate_strong_windows(
    dataset_metadata: dict,
    motion_features: dict,
    audio_features: dict,
    instrument: str = "vocal",
    threshold: float = 0.5,
):
    """Calculate strong correlation windows for each motion feature against specified audio features.

    Returns nested dict:
    total_strong_windows[artist][song][motion_feature_name][audio_feature] = list[(time, corr)]
    """
    audio_feature_names = ["onset_env", "rms"]

    total_strong_windows = {}
    for artist, songs in tqdm(
        dataset_metadata.items(), desc="Calculating Strong Windows"
    ):
        total_strong_windows.setdefault(artist, {})
        for song in tqdm(songs.keys(), desc="Loading Songs", leave=False):
            if (
                artist not in motion_features
                or song not in motion_features[artist]
                or artist not in audio_features
                or song not in audio_features[artist]
            ):
                continue

            total_strong_windows[artist].setdefault(song, {})
            fps = dataset_metadata[artist][song]["fps"]
            window_size = int(0.5 * fps)
            step_size = int(0.1 * fps)
            # Ensure audio feature exists
            available_audio = audio_features[artist][song][instrument]
            for motion_feature_name, motion_feature in motion_features[artist][
                song
            ][instrument].items():
                total_strong_windows[artist][song].setdefault(
                    motion_feature_name, {}
                )
                for af_name in audio_feature_names:
                    if af_name not in available_audio:
                        continue  # skip missing audio feature
                    try:
                        audio_feature = available_audio[af_name]
                        _, _, strong_windows = sliding_correlation(
                            motion_feature,
                            audio_feature,
                            window_size,
                            step_size,
                            fps,
                            threshold,
                        )
                        total_strong_windows[artist][song][
                            motion_feature_name
                        ][af_name] = strong_windows
                    except Exception as e:
                        print(
                            f"Error processing {artist}/{song}/{motion_feature_name} with {af_name}: {e}"
                        )
                        total_strong_windows[artist][song][
                            motion_feature_name
                        ][af_name] = []

    return total_strong_windows


def aggregate_strong_windows_per_chunk(
    total_strong_windows: dict, chunk_size: int = 4
):
    """Aggregate strong windows counts per time chunk for each artist/song.

    Returns structure:
      aggregated[artist][song][chunk_index] = count
    """
    aggregated_windows = {}
    for artist, songs in tqdm(
        total_strong_windows.items(), desc="Aggregating Strong Windows"
    ):
        aggregated_windows.setdefault(artist, {})
        for song, features in songs.items():
            chunks = {}
            for feature_name, audio_dict in features.items():
                for audio_feat_name, windows in audio_dict.items():
                    for time, _ in windows:
                        chunk_index = int(time // chunk_size)
                        chunks[chunk_index] = chunks.get(chunk_index, 0) + 1
            aggregated_windows[artist][song] = chunks

    return aggregated_windows


def main(args):
    with open(os.path.join(dataset_path, "dataset_metadata.json"), "r") as f:
        dataset_metadata = json.load(f)

    dataset_metadata = filter_metadata(
        dataset_metadata,
        args.instrument,
    )

    motion_features, audio_features = load_features(
        dataset_metadata, args.instrument, motion_file=args.motion_file
    )

    total_strong_windows = calculate_strong_windows(
        dataset_metadata,
        motion_features,
        audio_features,
        args.instrument,
        args.threshold,
    )

    aggregated_windows = aggregate_strong_windows_per_chunk(
        total_strong_windows
    )

    for artist, songs in aggregated_windows.items():
        for song, features in songs.items():
            output_filename = os.path.join(
                dataset_path,
                artist,
                song,
                args.instrument,
                "aggregated_strong_windows.json",
            )
            with open(output_filename, "w") as f:
                json.dump(features, f, indent=4)
            print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General correlation analysis script"
    )
    parser.add_argument(
        "--instrument",
        "-i",
        type=str,
        default="vocal",
        choices=instruments,
        help="Specify the instrument for correlation analysis",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="instrument",
        choices=["instrument", "generic", "both"],
        help=(
            "Which motion features to load: 'instrument' loads {instrument}_motion_features.json; "
            "'generic' loads motion_features.json (general block); 'both' merges both sets with prefixes."
        ),
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Threshold for strong correlation windows",
    )
    parser.add_argument(
        "--artist",
        "-a",
        type=str,
        default=None,
        help="Filter by artist name",
    )
    parser.add_argument(
        "--song",
        "-s",
        type=str,
        default=None,
        help="Filter by song name",
    )
    args = parser.parse_args()
    main(args)
