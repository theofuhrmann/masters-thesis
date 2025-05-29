import argparse
import os

from AudioFeatureExtractor import AudioFeatureExtractor
from dotenv import load_dotenv
from MotionFeatureExtractor import MotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = ["vocal", "mridangam", "violin"]
artist = "Abhiram Bode"

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
    default=5.0,
    help="Confidence threshold for motion feature extraction",
)
args = parser.parse_args()

if args.extract in ["motion", "both"]:
    motion_extractor = MotionFeatureExtractor(
        dataset_dir=ds,
        instruments=instruments,
        artist_filter=artist,
        conf_threshold=args.confidence_threshold,
    )
    motion_summary = motion_extractor.extract()
else:
    motion_summary = None

if args.extract in ["audio", "both"]:
    audio_extractor = AudioFeatureExtractor(
        dataset_dir=ds, instruments=instruments, artist_filter=artist
    )
    audio_extractor.extract(motion_summary)
