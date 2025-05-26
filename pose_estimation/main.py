import os

from dotenv import load_dotenv
from PoseEstimationPostProcessor import PoseEstimationPostProcessor
from PoseEstimator import PoseEstimator

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
config_file = "../../thesis/mmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py"
checkpoint_file = "../../thesis/mmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

estimator = PoseEstimator(config_file, checkpoint_file, device="cuda:0")
avoid_artists = ["Abhiram Bode", "Aditi Prahalad", "Ameya Karthikeyan"]
print("Processing dataset...")
estimator.process_dataset(dataset_path, avoid_artists=avoid_artists)

print("Post-processing dataset...")
# define your left→right instruments here:
instruments = ["mridangam", "vocal", "violin"]
pepp = PoseEstimationPostProcessor(dataset_path, instruments)
pepp.run(avoid_artists)
