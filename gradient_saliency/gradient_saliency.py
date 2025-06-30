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

from vovit import VoViT_b, VoViT_f, VoViT_fb  # type: ignore # noqa: E402
from vovit.display.dataloaders_new import (  # noqa: E402 # type: ignore
    AudioType,
    ModelType,
    SaragaAudiovisualDataset,
)

AUDIO_RATE = 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION = 4.0
DATASET_PATH = os.getenv("DATASET_PATH")
DATASET_METADATA_PATH = os.path.join(
    DATASET_PATH, "dataset_metadata_test.json"
)


def calculate_saliency(model, sample, device="cpu"):
    """
    Calculates gradient saliency for a single data sample.
    Assumes model is already on the correct device and in eval mode.
    """
    mix = sample["mix"].float().requires_grad_(True).to(device)
    target = sample["target"].float().to(device)
    if "face" in sample and sample["face"] is not None:
        face_tensor = torch.nan_to_num(sample["face"].float(), nan=0.0)
        face = face_tensor.requires_grad_(True).to(device)
        face.retain_grad()
    else:
        face = None

    if "body" in sample and sample["body"] is not None:
        body_tensor = torch.nan_to_num(sample["body"].float(), nan=0.0)
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
            raise TypeError(
                "Model must be an instance of VoViT_f, VoViT_b, or VoViT_fb"
            )

        wav = out["estimated_wav"]
        loss = F.mse_loss(wav, target)

    scale_factor = 2**16
    (loss * scale_factor).float().backward()

    face_sal, body_sal, mix_sal = None, None, None
    if face is not None and face.grad is not None:
        face_sal = face.grad.abs().mean(dim=(0, 1, 2))
    if body is not None and body.grad is not None:
        body_sal = body.grad.abs().mean(dim=(0, 1, 2))
    if mix.grad is not None:
        mix_sal = mix.grad.abs().mean()

    return face_sal, body_sal, mix_sal


def main(args):
    model_type_map = {
        "face": ModelType.FACE,
        "body": ModelType.BODY,
        "face_body": ModelType.BODY_FACE,
    }
    model_type = model_type_map[args.model]

    dataset = SaragaAudiovisualDataset(
        data_path=DATASET_PATH,
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=model_type,
        audio_type=AudioType.VOCAL,
        metadata_path=DATASET_METADATA_PATH,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
    )

    print(f"Loading {args.model} model to {DEVICE}...")
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(DEVICE)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(DEVICE)
    model.eval()

    results = defaultdict(
        lambda: defaultdict(
            lambda: {
                "face_saliency_sum": 0,
                "face_count": 0,
                "body_saliency_sum": 0,
                "body_count": 0,
                "mix_saliency_sum": 0,
                "mix_count": 0,
            }
        )
    )

    print(f"Calculating gradient saliency for all {len(dataset)} samples...")
    for sample in tqdm(dataloader):
        artist_name = sample["artist"][0]
        song_name = sample["song"][0]
        face_sal, body_sal, mix_sal = calculate_saliency(model, sample, DEVICE)

        if face_sal is not None:
            results[artist_name][song_name][
                "face_saliency_sum"
            ] += face_sal.detach().cpu()
            results[artist_name][song_name]["face_count"] += 1
        if body_sal is not None:
            results[artist_name][song_name][
                "body_saliency_sum"
            ] += body_sal.detach().cpu()
            results[artist_name][song_name]["body_count"] += 1
        if mix_sal is not None:
            results[artist_name][song_name][
                "mix_saliency_sum"
            ] += mix_sal.detach().cpu()
            results[artist_name][song_name]["mix_count"] += 1

    print("\nCalculating final per-keypoint averages...")
    final_averages = defaultdict(dict)
    for artist, songs in results.items():
        for song, data in songs.items():
            avg_face = (
                data["face_saliency_sum"] / data["face_count"]
                if data["face_count"] > 0
                else 0
            )
            avg_body = (
                data["body_saliency_sum"] / data["body_count"]
                if data["body_count"] > 0
                else 0
            )
            avg_mix = (
                data["mix_saliency_sum"] / data["mix_count"]
                if data["mix_count"] > 0
                else 0
            )

            final_averages[artist][song] = {
                "face_saliency_per_keypoint": (
                    avg_face.tolist()
                    if torch.is_tensor(avg_face)
                    else avg_face
                ),
                "body_saliency_per_keypoint": (
                    avg_body.tolist()
                    if torch.is_tensor(avg_body)
                    else avg_body
                ),
                "mix_saliency": (
                    avg_mix.item() if torch.is_tensor(avg_mix) else avg_mix
                ),
            }

    output_path = f"saliency_averages_{args.model}.json"
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(final_averages, f, indent=4)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VoViT Full Dataset Gradient Saliency"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="face_body",
        choices=["face", "body", "face_body"],
        help="Type of VoViT model to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "sdr", "sir", "sar"],
        help="Loss function to use for gradient saliency calculation",
    )
    args = parser.parse_args()
    main(args)
