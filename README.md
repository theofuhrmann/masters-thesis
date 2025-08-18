# Code for: Understanding Audio Source Separation in Carnatic Music with Multimodal Data

This repository contains the code used for all of the data processing and analysis used for my thesis for the Sound and Music Computing master's at Universitat Pompeu Fabra.

Disclaimer: The code regarding the source separation models is not included in this repository and is not public yet. Meaning that the code interacting with the models will not work.

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

## Dataset metadata

A reference metadata file is provided at `masters-thesis/dataset_analysis/dataset_metadata.json`. You can copy this file to your dataset root (as `dataset_metadata.json`) and modify it to match your media. Each top-level key is an artist; each nested key is a song.

Structure:

```
{
  "Artist Name": {
    "Song Title": {
      "layout": ["instrument1", "instrument2", ...],     # Left-to-right stage order (must match INSTRUMENTS)
      "moving_camera": true|false|null,                  # If true, some processing steps may skip the song
      "fps": 30.0,                                       # Video frame rate
      "duration": 1234.0333,                             # Duration in seconds (float)
      "correct_body_detection": true|false|null,         # QC flag (optional)
      "face_false_positive_frames": []                   # Frame indices to ignore for facial motion (optional)
    },
    ... more songs ...
  },
  ... more artists ...
}
```

Notes:
- Songs with null `layout`, `fps`, or `duration` are skipped.
- Optional flags (`correct_body_detection`, `face_false_positive_frames`) are used for filtering / cleaning but can be omitted.
- Additional custom fields are ignored by the current code.

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

## Visualization

The visualization module creates videos with pose estimation overlays to visually inspect the quality of pose detection and feature extraction.

Prerequisites:

- Ensure pose post-processing has been completed so keypoints are available for each instrument.
- For feature visualization, ensure motion and audio features have been extracted.
- FFmpeg must be installed on your system for audio processing.

Run commands (from `masters-thesis/`):

```bash
# Generate visualization for a specific artist and song (20-second clip by default)
python -m visualization.main -a "Artist Name" -s "Song Title"

# Specify custom time range (start and end times in seconds)
python -m visualization.main -a "Artist Name" -s "Song Title" -st 10.0 -et 30.0

# Generate visualization without audio
python -m visualization.main -a "Artist Name" -s "Song Title" -st 0 -et 20

# Generate visualization without motion/audio features overlay
python -m visualization.main -a "Artist Name" -s "Song Title" --no_features

# Add audio to the output video
python -m visualization.main -a "Artist Name" -s "Song Title" --add_audio
```

Outputs:

- Video file: `<song>/<start_time>_<end_time>_test.mp4` containing the original video with pose skeleton overlays

Features:

- **Skeleton overlay**: Shows detected body pose with colored lines connecting keypoints
- **Instrument-specific colors**: Each instrument gets a distinct color (vocal: green, violin: blue, mridangam: red)
- **Confidence filtering**: Only keypoints above the confidence threshold (default 3.0) are displayed
- **Occlusion handling**: Hides occluded body parts for edge performers (arms/hands)
- **Feature visualization**: When available, displays real-time motion and audio features as colored bars
- **Correlation highlighting**: Highlights body parts that correlate with audio during specific time windows
- **PCA contribution**: Keypoint brightness indicates contribution to the first principal component

Notes:

- Lower body keypoints (legs, feet) are hidden by default to focus on upper body performance gestures
- Face keypoints are shown for vocal performers when facial pose data is available
- The tool automatically finds the first .mov file in each song directory
- Output videos maintain the original frame rate and resolution

## Gradient Saliency Analysis

The gradient saliency module computes feature importance for VoViT models to understand which keypoints and audio features contribute most to source separation performance.

Prerequisites:

- Set `VOVIT_PATH` in your `.env` to point to the VoViT repository location
- Ensure the dataset has been processed with pose estimation and feature extraction
- CUDA-capable GPU recommended for faster processing

Available methods:

- **Vanilla gradients** (`gradient_saliency.py`): Standard gradient-based saliency
- **Gradient × Input** (`gradient_input_saliency.py`): Element-wise product of gradients and inputs
- **Integrated Gradients** (`integrated_gradients.py`): More robust attribution method using path integration

Run commands (from `masters-thesis/`):

```bash
# Integrated gradients (recommended) - face+body model
python -m gradient_saliency.integrated_gradients --model face_body

# Face-only model
python -m gradient_saliency.integrated_gradients --model face

# Body-only model  
python -m gradient_saliency.integrated_gradients --model body

# Filter to top 10% correlation chunks only
python -m gradient_saliency.integrated_gradients --model face_body --correlation-filter-percentage 0.1 --correlation-filter-top

# Filter to bottom 10% correlation chunks
python -m gradient_saliency.integrated_gradients --model face_body --correlation-filter-percentage 0.1 --correlation-filter-bottom

# Vanilla gradients (faster but less robust)
python -m gradient_saliency.gradient_saliency --model face_body

# Gradient × Input method
python -m gradient_saliency.gradient_input_saliency --model face_body
```

Outputs:

- Integrated gradients: `results/integrated_gradients_[top/bottom]_[percentage]_[model]/[artist]_[song].json`
- Other methods: `saliency_averages_[model].json` or `saliency_scores/[artist]_[song]_[model].json`

Features:

- **Multi-modal attribution**: Computes importance for both visual (face/body keypoints) and audio (mix) inputs
- **Correlation filtering**: Analyzes chunks with highest/lowest audio-visual correlation
- **Baseline selection**: Uses speech mean face for facial keypoints, zeros for other modalities
- **Per-keypoint granularity**: Individual importance scores for each of the 68 face or 55 body keypoints
- **Robustness**: Integrated gradients method reduces noise compared to vanilla gradients

Notes:

- Integrated gradients uses 25 integration steps by default; increase `--num_steps` for more precision
- Results are saved in JSON files with per-keypoint importance scores

## Attention Analysis

The attention analysis module extracts cross-modal attention weights from VoViT models to understand how audio and visual features interact during source separation.

Prerequisites:

- Set `VOVIT_PATH` in your `.env` to point to the VoViT repository location
- Ensure the dataset has been processed with pose estimation and feature extraction
- CUDA-capable GPU recommended for faster processing

Run commands (from `masters-thesis/`):

```bash
# Analyze vocal model attention (audio-to-face and face-to-audio)
python -m attention_analysis.attention_analysis --model vocal

# Analyze violin model attention (audio-to-body and body-to-audio)
python -m attention_analysis.attention_analysis --model violin
```

Outputs:

- Attention scores: `results/attention_scores_[model]_test_new.json`

Features:

- **Cross-modal attention**: Captures how much audio features attend to visual features and vice versa
- **Temporal alignment**: Provides attention scores at each time step in the 4-second chunks
- **Bidirectional analysis**: Computes both audio→video and video→audio attention weights
- **Per-song aggregation**: Collects all attention patterns for each artist-song combination

Data structure:

```json
{
  "Artist Name": {
    "Song Title": [
      {
        "audio_attention": [float, ...],   // How much each audio timestep attends to video
        "video_attention": [float, ...]    // How much audio attends to each video timestep
      },
      ... // One entry per 4-second chunk
    ]
  }
}
```

Notes:

- Vocal model uses facial keypoints; violin model uses body keypoints
- Attention weights are averaged across attention heads for interpretability
- Processing is done in 4-second chunks matching the model's training configuration
- Results can be used to identify moments of high audio-visual interaction

