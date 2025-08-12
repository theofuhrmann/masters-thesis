import argparse
import os

from audio.AudioFeatureExtractor import AudioFeatureExtractor
from dotenv import load_dotenv
from motion.GeneralMotionFeatureExtractor import GeneralMotionFeatureExtractor
from motion.ViolinMotionFeatureExtractor import ViolinMotionFeatureExtractor
from motion.VocalMotionFeatureExtractor import VocalMotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = os.getenv("INSTRUMENTS").split(",")

parser = argparse.ArgumentParser(description="Feature extraction script")
parser.add_argument(
    "--extract",
    "-e",
    choices=["motion", "audio", "both"],
    default="both",
    help="Specify which extraction to perform: motion, audio, or both",
)
parser.add_argument(
    "--motion_type",
    "-m",
    choices=["general", "vocal", "violin"],
    default="general",
    help="Specify the type of motion feature extraction: general or violin",
)
parser.add_argument(
    "--confidence_threshold",
    "-ct",
    type=float,
    default=3.0,
    help="Confidence threshold for motion feature extraction",
)
parser.add_argument(
    "--all_body_parts",
    "-all",
    action="store_true",
    help="Use all body parts during motion feature extraction, including occluded parts",
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
parser.add_argument(
    "--force",
    "-f",
    action="store_true",
    help="Force reprocessing of existing data",
)

args = parser.parse_args()

if args.extract in ["motion", "both"]:
    if args.motion_type == "violin":
        violin_motion_extractor = ViolinMotionFeatureExtractor(
            dataset_dir=ds,
            artist_filter=args.artist,
            song_filter=args.song,
            conf_threshold=args.confidence_threshold,
        )
        violin_motion_extractor.extract(force=args.force)
    elif args.motion_type == "vocal":
        vocal_motion_extractor = VocalMotionFeatureExtractor(
            dataset_dir=ds,
            artist_filter=args.artist,
            song_filter=args.song,
            conf_threshold=args.confidence_threshold,
        )
        vocal_motion_extractor.extract(force=args.force)
    elif args.motion_type == "general":
        motion_extractor = GeneralMotionFeatureExtractor(
            dataset_dir=ds,
            instruments=instruments,
            artist_filter=args.artist,
            song_filter=args.song,
            conf_threshold=args.confidence_threshold,
            pca_output_filename=None,
        )
        motion_extractor.extract(
            all_body_parts=args.all_body_parts, force=args.force
        )
    else:
        raise ValueError(
            "Invalid motion type specified. Choose 'general' or 'violin'."
        )

if args.extract in ["audio", "both"]:
    audio_extractor = AudioFeatureExtractor(
        dataset_dir=ds,
        instruments=instruments,
        artist_filter=args.artist,
    )
    audio_extractor.extract()
