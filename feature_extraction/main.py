import argparse
import os

from AudioFeatureExtractor import AudioFeatureExtractor
from dotenv import load_dotenv
from MotionFeatureExtractor import MotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = ["vocal", "mridangam", "violin"]

parser = argparse.ArgumentParser(description="Feature extraction script")
parser.add_argument(
    "--extract",
    choices=["motion", "audio", "both"],
    default="both",
    help="Specify which extraction to perform: motion, audio, or both",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=3.0,
    help="Confidence threshold for motion feature extraction",
)
parser.add_argument(
    "--ignore_occluded_parts",
    action="store_true",
    help="Ignore occluded body parts during motion feature extraction",
)
parser.add_argument(
    "--artist",
    type=str,
    default=None,
    help="Filter by artist name",
)

args = parser.parse_args()

motion_output_filename = "motion_features_normalized.json" if not args.ignore_occluded_parts else "motion_features_normalized_occluded.json"

if args.extract in ["motion", "both"]:
    motion_extractor = MotionFeatureExtractor(
        dataset_dir=ds,
        instruments=instruments,
        artist_filter=args.artist,
        conf_threshold=args.confidence_threshold,
        motion_output_filename=motion_output_filename,
        pca_output_filename=None,
    )
    motion_extractor.extract(ignore_occluded_parts=args.ignore_occluded_parts)

if args.extract in ["audio", "both"]:
    audio_extractor = AudioFeatureExtractor(
        dataset_dir=ds, instruments=instruments, motion_output_filename=motion_output_filename, artist_filter=args.artist
    )
    audio_extractor.extract()
