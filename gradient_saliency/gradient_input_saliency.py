import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv()
vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from tools.body_parts_map import keypoint_map

from vovit import VoViT_b, VoViT_f, VoViT_fb # type: ignore # noqa: E402
from vovit.display.dataloaders_new import ( # noqa: E402 # type: ignore
    AudioType,
    ModelType,
    SaragaAudiovisualDataset,
)

torch.autograd.set_detect_anomaly(True)

AUDIO_RATE = 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION = 4.0
DATASET_PATH = os.getenv("DATASET_PATH")
DATASET_METADATA_PATH = os.path.join(DATASET_PATH, "dataset_metadata.json")


def calculate_saliency(model, sample, device="cpu"):
    """
    Calculates gradient saliency for a single data sample.
    Assumes model is already on the correct device and in eval mode.
    """
    mix = sample['mix'].float().requires_grad_(True).to(device)
    target = sample['target'].float().to(device)
    if 'face' in sample and sample['face'] is not None:
        face_tensor = torch.nan_to_num(sample['face'].float(), nan=0.0)
        face = face_tensor.requires_grad_(True).to(device)
        face.retain_grad()
    else:
        face = None

    if 'body' in sample and sample['body'] is not None:
        body_tensor = torch.nan_to_num(sample['body'].float(), nan=0.0)
        body = body_tensor.requires_grad_(True).to(device)
        body.retain_grad()
    else:
        body = None
    mix.retain_grad()

    model.zero_grad()

    with autocast():
        if isinstance(model, VoViT_f):
            out = model.forward(mix, face)
        elif isinstance(model, VoViT_b):
            out = model.forward(mix, body)
        elif isinstance(model, VoViT_fb):
            out = model.forward(mix, face, body)
        else:
            raise TypeError("Model must be an instance of VoViT_f, VoViT_b, or VoViT_fb")
        
        wav = out["estimated_wav"]
        loss = F.mse_loss(wav, target)

    scale_factor = 2**16
    (loss * scale_factor).float().backward()

    face_sal, body_sal, mix_sal = None, None, None
    if face is not None and face.grad is not None:
        face_sal = (face * face.grad).abs().mean(dim=(0, 1, 2))
    if body is not None and body.grad is not None:
        body_sal = (body * body.grad).abs().mean(dim=(0, 1, 2))
    if mix.grad is not None:
        mix_sal = (mix * mix.grad).abs().mean()

    return face_sal, body_sal, mix_sal


def save_song_results(artist, song, data, output_dir, model_type):
    """Formats and saves the saliency data for a single song to a JSON file."""
    if not any(data.values()):
        print(f"\nNo data collected for {artist} - {song}. Skipping save.")
        return

    print(f"\nFormatting results for {artist} - {song}...")
    final_data = {
        'face_saliency_per_keypoint': [t.tolist() for t in data['face_saliency']],
        'body_saliency_per_keypoint': [t.tolist() for t in data['body_saliency']],
        'mix_saliency': [t.item() for t in data['mix_saliency']]
    }

    # Sanitize artist and song names for use in filenames
    safe_artist = "".join(c for c in artist if c.isalnum() or c in " ._").rstrip()
    safe_song = "".join(c for c in song if c.isalnum() or c in " ._").rstrip()
    
    output_filename = f"{safe_artist}_{safe_song}_{model_type}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=4)


def main(args):
    model_type_map = {
        "face": ModelType.FACE,
        "body": ModelType.BODY,
        "face_body": ModelType.BODY_FACE,
    }
    model_type_key = args.model
    model_type = model_type_map[model_type_key]

    dataset = SaragaAudiovisualDataset(
        data_path=DATASET_PATH,
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=model_type,
        audio_type=AudioType.VOCAL,
        metadata_path=DATASET_METADATA_PATH,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    print(f"Loading {model_type_key} model to {DEVICE}...")
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(DEVICE)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(DEVICE)
    model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    current_artist, current_song = None, None
    current_song_data = defaultdict(list)

    print(f"Calculating gradient saliency for all {len(dataset)} samples...")
    for sample in tqdm(dataloader):
        artist_name = sample['artist'][0]
        song_name = sample['song'][0]

        # Initialize on the first sample
        if current_artist is None:
            current_artist, current_song = artist_name, song_name

        # If we've moved to a new song, save the results for the previous one
        if artist_name != current_artist or song_name != current_song:
            save_song_results(current_artist, current_song, current_song_data, args.output_dir, model_type_key)
            current_artist, current_song = artist_name, song_name
            current_song_data = defaultdict(list)

        face_sal, body_sal, mix_sal = calculate_saliency(model, sample, DEVICE)

        if face_sal is not None:
            current_song_data['face_saliency'].append(face_sal.detach().cpu())
        if body_sal is not None:
            current_song_data['body_saliency'].append(body_sal.detach().cpu())
        if mix_sal is not None:
            current_song_data['mix_saliency'].append(mix_sal.detach().cpu())

    # Save the data for the very last song after the loop finishes
    if current_artist is not None:
        save_song_results(current_artist, current_song, current_song_data, args.output_dir, model_type_key)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoViT Full Dataset Gradient Saliency")
    parser.add_argument(
        "--model",
        type=str,
        default="face_body",
        choices=["face", "body", "face_body"],
        help="Type of VoViT model to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saliency_scores",
        help="Directory to save the output JSON files",
    )
    args = parser.parse_args()
    main(args)
