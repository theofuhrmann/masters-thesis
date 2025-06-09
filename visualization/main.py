import os
import json
from Visualizer import GestureVisualizer
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
layout = ['mridangam', 'vocal', 'violin']

parser = argparse.ArgumentParser(description='Run gesture visualization on dataset.')
parser.add_argument('--artist', type=str, default=None, help='Filter by artist name')
parser.add_argument('--song', type=str, default=None, help='Filter by song name')
parser.add_argument('--start_time', type=float, default=0.0, help='Start time for visualization in seconds')
parser.add_argument('--end_time', type=float, default=20.0, help='End time for visualization in seconds')
parser.add_argument('--add_audio', action='store_true', help='Add audio to the visualization')
parser.add_argument('--no_features', action='store_true', help='Do not use features for visualization')

args = parser.parse_args()

metadata_file = os.path.join(dataset_path, "dataset_metadata.json")
with open(metadata_file, "r") as f:
    dataset_metadata = json.load(f)

for artist, songs in tqdm(sorted(dataset_metadata.items())):
    if args.artist and artist != args.artist:
        continue
    for song, metadata in songs.items():
        if args.song and song != args.song:
            continue
        if metadata["layout"] == layout:
            try:
                print(f"Processing: {artist} - {song}")
                if not args.no_features:
                    gesture_visualizer = GestureVisualizer(
                        dataset_path=dataset_path,
                        artist=artist,
                        song=song,
                        instruments=layout,
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
                        instruments=layout,
                        window_size=5,
                    )
                output_file = os.path.join(dataset_path, artist, song, 
                                           f"{args.start_time}_{args.end_time}_test.mp4")
                gesture_visualizer.visualize(
                    output_file=output_file,
                    confidence_threshold=3,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    add_audio=args.add_audio
                )
                print(f"Successfully generated video for {artist} - {song} at {output_file}")
            except Exception as e:
                print(f"Failed to process {artist} - {song}: {e}")
