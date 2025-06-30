import argparse
import os

from audio.AudioFeatureExtractor import AudioFeatureExtractor
from dotenv import load_dotenv
from motion.GeneralMotionFeatureExtractor import GeneralMotionFeatureExtractor
from motion.ViolinMotionFeatureExtractor import ViolinMotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = ["vocal", "mridangam", "violin"]

parser = argparse.ArgumentParser(description="Feature extraction script")
parser.add_argument(
    "--extract",
    "-e",
    choices=["motion", "audio", "both", "motion-violin"],
    default="both",
    help="Specify which extraction to perform: motion, audio, or both",
)
parser.add_argument(
    "--confidence_threshold",
    "-ct",
    type=float,
    default=3.0,
    help="Confidence threshold for motion feature extraction",
)
parser.add_argument(
    "--ignore_occluded_parts",
    "-i",
    action="store_true",
    help="Ignore occluded body parts during motion feature extraction",
)
parser.add_argument(
    "--artist",
    "-a",
    type=str,
    default=None,
    help="Filter by artist name",
)

args = parser.parse_args()

motion_output_filename = (
    "motion_features_normalized.json"
    if not args.ignore_occluded_parts
    else "motion_features_normalized_occluded.json"
)

if args.extract in ["motion", "both"]:
    motion_extractor = GeneralMotionFeatureExtractor(
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
        dataset_dir=ds,
        instruments=instruments,
        motion_output_filename=motion_output_filename,
        artist_filter=args.artist,
    )
    audio_extractor.extract()

if args.extract == "motion-violin":
    violin_motion_extractor = ViolinMotionFeatureExtractor(
        dataset_dir=ds,
        artist_filter=args.artist,
        conf_threshold=args.confidence_threshold,
    )
    violin_motion_extractor.extract()
