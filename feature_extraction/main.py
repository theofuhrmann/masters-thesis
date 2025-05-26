import os

from AudioFeatureExtractor import AudioFeatureExtractor
from dotenv import load_dotenv
from MotionFeatureExtractor import MotionFeatureExtractor

load_dotenv()
ds = os.getenv("DATASET_PATH")
instruments = ["vocal", "mridangam", "violin"]
artist = "Abhiram Bode"

motion_extractor = MotionFeatureExtractor(
    dataset_dir=ds, instruments=instruments, artist_filter=artist
)
motion_summary = motion_extractor.extract()

audio_extractor = AudioFeatureExtractor(
    dataset_dir=ds, instruments=instruments, artist_filter=artist
)
audio_extractor.extract(motion_summary)
