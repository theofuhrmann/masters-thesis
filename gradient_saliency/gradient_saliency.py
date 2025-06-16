import argparse
import os
import sys

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

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
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
DURATION = 4.0
DATASET_PATH = os.getenv("DATASET_PATH")
DATASET_METADATA_PATH = os.path.join(DATASET_PATH, "dataset_metadata_test.json")


def gradient_saliency(dataset, sample, device="cpu"):
    if dataset.model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={})
    elif dataset.model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={})
    elif dataset.model_type == ModelType.BODY_FACE:
        model = VoViT_fb(pretrained=True, debug={})
    else:
        raise ValueError("Unsupported model type")

    model.eval()
    model.zero_grad()

    (mix, face, body), target = dataset[0]

    mix = mix.unsqueeze(0).float()
    target = target.unsqueeze(0).float()
    face = face.unsqueeze(0).float().requires_grad_(True)
    body = body.unsqueeze(0).float().requires_grad_(True)

    # run the official forward (4s only)
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

    # backprop
    loss = F.mse_loss(wav, target)
    loss.backward()

    # compute saliency: average abs‐grad over batch/time/coords
    face_sal = face.grad.abs().mean(dim=(0, 1, 2))  # → [68]
    body_sal = body.grad.abs().mean(dim=(0, 1, 2))  # → [55]

    # print top‐10
    face_imp, face_idx = torch.sort(face_sal, descending=True)
    body_imp, body_idx = torch.sort(body_sal, descending=True)

    return (
        face_idx.tolist(),
        face_imp.tolist(),
        body_idx.tolist(),
        body_imp.tolist(),
    )


def main(args):
    dataset = SaragaAudiovisualDataset(
        data_path=DATASET_PATH,
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=ModelType.BODY_FACE,
        audio_type=AudioType.VOCAL,
        metadata_path=DATASET_METADATA_PATH,
    )

    # pick a chunk
    sample = (args.artist, args.song)
    face_idx, face_imp, body_idx, body_imp = gradient_saliency(dataset, sample, DEVICE)
    print(f"Face landmarks: {args.artist} - {args.song}")
    for i in range(10):
        print(f"  {i+1:2d}: {face_idx[i]:3d} ({face_imp[i]})")
    print(f"Body landmarks: {args.artist} - {args.song}")
    for i in range(10):
        print(f"  {i+1:2d}: {body_idx[i]:3d} ({body_imp[i]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoViT Model Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="face_body",
        choices=["face", "body", "face_body"],
        help="Type of VoViT model to use (face, body, face_body)",
    )
    parser.add_argument(
        "--artist", type=str, default="Abhiram Bode", help="Artist name"
    )
    parser.add_argument(
        "--song", type=str, default="Devi Pavane", help="Song name"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time (seconds)"
    )
    parser.add_argument(
        "--duration", type=float, default=4.0, help="Duration (seconds)"
    )

    args = parser.parse_args()
    main(args)
