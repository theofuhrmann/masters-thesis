import argparse
import os

from dotenv import load_dotenv
from PoseEstimationPostProcessor import PoseEstimationPostProcessor
from PoseEstimator import PoseEstimator

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
config_file = os.getenv("MMPOSE_CONFIG_PATH")
checkpoint_file = os.getenv("MMPOSE_CHECKPOINT_PATH")

parser = argparse.ArgumentParser(
    description="Run pose estimation and/or post-processing."
)
parser.add_argument("--pose", action="store_true", help="Run pose estimation")
parser.add_argument("--post", action="store_true", help="Run post-processing")
parser.add_argument(
    "--artist", "-a", type=str, default=None, help="Filter by artist name"
)
parser.add_argument(
    "--song", "-s", type=str, default=None, help="Filter by song name"
)
parser.add_argument(
    "--force",
    "-f",
    action="store_true",
    help="Force reprocessing of existing data",
)

args = parser.parse_args()

if args.pose:
    print("Processing dataset...")
    estimator = PoseEstimator(config_file, checkpoint_file, device="cpu")
    estimator.process_dataset(
        dataset_path=dataset_path,
        artist_filter=args.artist,
        song_filter=args.song,
        force=args.force,
    )

if args.post:
    print("Post-processing dataset...")
    pepp = PoseEstimationPostProcessor(dataset_path=dataset_path)
    pepp.run(
        artist_filter=args.artist, song_filter=args.song, force=args.force
    )
