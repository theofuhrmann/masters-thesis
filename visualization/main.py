import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm
from Visualizer import GestureVisualizer

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
instruments = os.getenv("INSTRUMENTS").split(",")

parser = argparse.ArgumentParser(
    description="Run gesture visualization on dataset."
)
parser.add_argument(
    "-a", "--artist", type=str, default=None, help="Filter by artist name"
)
parser.add_argument(
    "-s", "--song", type=str, default=None, help="Filter by song name"
)
parser.add_argument(
    "-st",
    "--start_time",
    type=float,
    default=0.0,
    help="Start time for visualization in seconds",
)
parser.add_argument(
    "-et",
    "--end_time",
    type=float,
    default=20.0,
    help="End time for visualization in seconds",
    nargs="?",
    const=None,
)
parser.add_argument(
    "-aa",
    "--add_audio",
    action="store_true",
    help="Add audio to the visualization",
)
parser.add_argument(
    "-nf",
    "--no_features",
    action="store_true",
    help="Do not use features for visualization",
)

args = parser.parse_args()

metadata_file = os.path.join(dataset_path, "dataset_metadata.json")
with open(metadata_file, "r") as f:
    dataset_metadata = json.load(f)

if args.artist is None and args.song is None:
    for artist, songs in list(dataset_metadata.items()):
        for song in list(songs.keys()):
            if dataset_metadata[artist][song]["layout"] != instruments:
                del dataset_metadata[artist][song]

    for artist, songs in list(dataset_metadata.items()):
        if artist in dataset_metadata:
            if len(dataset_metadata[artist]) == 0:
                del dataset_metadata[artist]

for artist, songs in tqdm(sorted(dataset_metadata.items())):
    if args.artist and artist != args.artist:
        continue
    for song, metadata in songs.items():
        if args.song and song != args.song:
            continue
        print(f"Processing: {artist} - {song}")
        if not args.no_features:
            gesture_visualizer = GestureVisualizer(
                dataset_path=dataset_path,
                artist=artist,
                song=song,
                instruments=metadata["layout"],
                window_size=5,
                motion_features_filename="motion_features.json",
                audio_features_filename="audio_features.json",
                pca_components_filename="pca_components.json",
                correlation_windows_filename="00_correlation_05s_windows.json",
            )
        else:
            gesture_visualizer = GestureVisualizer(
                dataset_path=dataset_path,
                artist=artist,
                song=song,
                instruments=metadata["layout"],
                window_size=5,
            )
        output_file = os.path.join(
            dataset_path,
            artist,
            song,
            f"{args.start_time}_{args.end_time}_test.mp4",
        )
        gesture_visualizer.visualize(
            output_file=output_file,
            confidence_threshold=3,
            start_time=args.start_time,
            end_time=args.end_time,
            add_audio=args.add_audio,
        )
        print(
            f"Successfully generated video for {artist} - {song} at {output_file}"
        )
