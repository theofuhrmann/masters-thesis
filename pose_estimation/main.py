import os
import argparse

from dotenv import load_dotenv
from PoseEstimationPostProcessor import PoseEstimationPostProcessor
from PoseEstimator import PoseEstimator

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
config_file = "../../thesis/mmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py"
checkpoint_file = "../../thesis/mmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

parser = argparse.ArgumentParser(description='Run pose estimation and/or post-processing.')
parser.add_argument('--pose', action='store_true', help='Run pose estimation')
parser.add_argument('--post', action='store_true', help='Run post-processing')

args = parser.parse_args()

avoid_artists = ["Abhiram Bode", "Aditi Prahalad", "Ameya Karthikeyan"]

if args.pose:
    print("Processing dataset...")
    estimator = PoseEstimator(config_file, checkpoint_file, device="cuda:0")
    estimator.process_dataset(dataset_path, avoid_artists=avoid_artists)

if args.post:
    print("Post-processing dataset...")
    # define your left→right instruments here:
    instruments = ["mridangam", "vocal", "violin"]
    pepp = PoseEstimationPostProcessor(dataset_path, instruments)
    pepp.run()
