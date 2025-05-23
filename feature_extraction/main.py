from dotenv import load_dotenv
import os
from feature_extraction.AudioFeatureExtractor import AudioFeatureExtractor
from feature_extraction.MotionFeatureExtractor import MotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = ["vocal", "mridangam", "violin"]
artist = "Aditi Prahalad"

motion_extractor = MotionFeatureExtractor(
    dataset_dir=ds,
    instruments=instruments,
    artist_filter=artist
)
motion_summary = motion_extractor.extract()

audio_extractor = AudioFeatureExtractor(
    dataset_dir=ds,
    instruments=instruments,
    artist_filter=artist
)
audio_extractor.extract(motion_summary)