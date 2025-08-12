# Code for my master's thesis

## Installation

To set up the environment, follow these steps:

1.  Create a conda environment with Python 3.10:

    ```bash
    conda create --name thesis python=3.10
    conda activate thesis
    ```
2.  Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
    
3.  Copy the `.env.example` file to `.env`:

    ```bash
    cp .env.example .env
    ```

4.  Fill in the values in the `.env` file with your actual configuration.

## Pose estimation

The pose estimation pipeline has two stages:

- Pose estimation: runs MMPose on the dataset and writes per-song pickles.
- Post-processing: reorders subjects, maps them to instruments using metadata, and writes NumPy arrays per instrument.

Prerequisites:

- Set DATASET_PATH in your .env to the absolute path of your dataset root. This folder should contain:
    - dataset_metadata.json
    - One subfolder per artist, and inside each, one subfolder per song containing the video file(s)
- Set CONFIG_FILE and CHECKPOINT_FILE in your .env to point to valid local MMPose config and checkpoint files.
    - By default they reference files under the mmpose folder in this repo. Adjust if your paths differ.
- Default device is CPU. To use a GPU, edit pose_estimation/main.py and set device="cuda:0" when creating PoseEstimator.

Run commands (from masters-thesis/):

```bash
# Run pose estimation for the whole dataset
python -m pose_estimation.main --pose

# Only for a specific artist
python -m pose_estimation.main --pose -a "Artist Name"

# Only for a specific song by an artist
python -m pose_estimation.main --pose -a "Artist Name" -s "Song Title"

# Force reprocessing even if outputs exist
python -m pose_estimation.main --pose -f

# Run post-processing (builds per-instrument NumPy arrays)
python -m pose_estimation.main --post

# Post-process a specific artist/song (with optional force)
python -m pose_estimation.main --post -a "Artist Name" -s "Song Title" -f

# Do both stages in one go
python -m pose_estimation.main --pose --post
```

Inputs and outputs:

- Input videos: the script picks the first .mov file in each song folder.
- Pose estimation writes pose_estimation_30fps.pkl to each song folder.
- Post-processing writes, for each instrument in the song layout:
  - <song>/<instrument>/keypoints.npy
  - <song>/<instrument>/keypoint_scores.npy
- Songs marked with "moving_camera" in dataset_metadata.json are skipped during post-processing.

## Feature extraction

Extract motion (general, vocal, violin) and audio features per instrument from the processed dataset.

Prerequisites:

- Ensure pose post-processing has been run so each song/instrument folder contains `keypoints.npy` and `keypoint_scores.npy` (for vocal motion, `vocal/face_keypoints.npy`).
- In your `.env`, set:
  - `DATASET_PATH` to the dataset root.
  - `INSTRUMENTS` as a comma-separated list in left-to-right stage order (must match each song's `layout` in `dataset_metadata.json`). Example: `INSTRUMENTS=violin,vocal,mridangam`.
- For audio features, each song folder should contain WAV files named with the instrument (e.g., `violin_*.wav`, `vocal_*.wav`; mridangam may contain `mri` in the filename).

Run commands (from `masters-thesis/`):

```bash
# Run both motion and audio feature extraction over the whole dataset
python -m feature_extraction.main --extract both

# Motion only (general per-instrument motion features)
python -m feature_extraction.main --extract motion --motion_type general

# Motion only for violin-specific features (bowing arm etc.)
python -m feature_extraction.main --extract motion --motion_type violin

# Motion only for vocal-specific facial features
python -m feature_extraction.main --extract motion --motion_type vocal

# Audio only
python -m feature_extraction.main --extract audio

# Filter by artist and song
python -m feature_extraction.main -e motion -m general -a "Artist Name" -s "Song Title"

# Force reprocessing even if outputs exist
python -m feature_extraction.main -e both -f

# Use all body parts (do not hide occluded arms for edge performers)
python -m feature_extraction.main -e motion -m general --all_body_parts

# Adjust keypoint confidence threshold (default 3.0)
python -m feature_extraction.main -e motion -m general -ct 4.0
```

Outputs (per song/instrument unless noted):

- General motion: `<song>/<instrument>/motion_features.json`
- Violin motion: `<song>/violin/violin_motion_features.json`
- Vocal motion: `<song>/vocal/vocal_motion_features.json`
- Audio features: `<song>/<instrument>/audio_features.json`

Notes:

- General motion requires `INSTRUMENTS` to exactly match the song's `layout`; mismatches are skipped.
- Audio features are time-aligned to video frames using `fps` and `duration` from `dataset_metadata.json`.
- Confidence threshold masks low-confidence keypoints before computing motion features.