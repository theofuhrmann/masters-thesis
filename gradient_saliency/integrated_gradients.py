import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np

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

from vovit import VoViT_b, VoViT_f, VoViT_fb # type: ignore # noqa: E402
from vovit.display.dataloaders_new import ( # noqa: E402 # type: ignore
    AudioType,
    ModelType,
    SaragaAudiovisualDataset,
)

AUDIO_RATE = 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DURATION = 4.0
DATASET_PATH = os.getenv("DATASET_PATH")
DATASET_METADATA_PATH = os.path.join(DATASET_PATH, "dataset_metadata_test.json")

speech_mean_face = torch.from_numpy(np.load("speech_mean_face.npy")).float()


def calculate_saliency(model, sample, device="cpu"):
    mix = sample['mix'].float().to(device)
    face = torch.nan_to_num(sample['face'].float(), nan=0.0).to(device) if 'face' in sample and sample['face'] is not None else None
    body = torch.nan_to_num(sample['body'].float(), nan=0.0).to(device) if 'body' in sample and sample['body'] is not None else None

    face_sal, body_sal, mix_sal = None, None, None
    if face is not None:
        face_sal = integrated_gradients(face, 'face', model, sample, device=device)
    if body is not None:
        body_sal = integrated_gradients(body, 'body', model, sample, device=device)
    mix_sal = integrated_gradients(mix, 'mix', model, sample, device=device)

    return face_sal, body_sal, mix_sal

def integrated_gradients(input_tensor, input_name, model, sample, baseline=None, steps=25, device="cpu"):
    """
    Computes Integrated Gradients for a given input tensor.

    input_tensor: torch.Tensor (will have requires_grad set in this function)
    input_name: str, must be one of ['mix', 'face', 'body']
    model: the VoViT model
    sample: the input sample
    baseline: baseline tensor (same shape as input_tensor), defaults to appropriate baseline
    steps: number of integration steps
    """
    # Make sure input_tensor doesn't already have requires_grad to avoid issues
    input_tensor = input_tensor.detach().clone()
    
    if baseline is None:
        if input_name == 'face':
            baseline = speech_mean_face.unsqueeze(0).unsqueeze(0).expand_as(input_tensor).clone().detach().to(device)
            # Add a small amount of noise for gradient stability
            baseline += torch.randn_like(baseline).to(device) * 0.01
            print(f"face baseline shape: {baseline.shape}, mean: {baseline.mean().item()}, original mean: {input_tensor.mean().item()}")
        else:
            # For normalized inputs (body keypoints, audio), zero is appropriate
            baseline = torch.zeros_like(input_tensor).to(device)

    # Move sample tensors to device once
    mix_orig = sample['mix'].float().to(device)
    face_orig = torch.nan_to_num(sample['face'].float(), nan=0.0).to(device) if 'face' in sample and sample['face'] is not None else None
    body_orig = torch.nan_to_num(sample['body'].float(), nan=0.0).to(device) if 'body' in sample and sample['body'] is not None else None
    target_orig = sample["target"].float().to(device)

    # Interpolation coefficients
    alphas = torch.linspace(0, 1, steps).to(device)
    accumulated_gradients = torch.zeros_like(input_tensor).to(device)

    for alpha in tqdm(alphas, desc=f"Calculating IG for {input_name}"):
        # Interpolate between baseline and input
        interpolated_input = baseline + alpha * (input_tensor - baseline)
        interpolated_input.requires_grad_(True)

        # Prepare inputs for forward
        mix, face, body = mix_orig, face_orig, body_orig

        if input_name == 'mix':
            mix = interpolated_input
        elif input_name == 'face':
            face = interpolated_input
        elif input_name == 'body':
            body = interpolated_input

        model.zero_grad()

        with autocast():
            if isinstance(model, VoViT_f):
                out = model.forward(mix, face)
            elif isinstance(model, VoViT_b):
                out = model.forward(mix, body)
            elif isinstance(model, VoViT_fb):
                out = model.forward(mix, face, body)
            else:
                raise TypeError("Invalid model type")

            loss = F.mse_loss(out["estimated_wav"], target_orig)

        scale_factor = 2**16
        (loss * scale_factor).float().backward()

        grad = interpolated_input.grad
        if grad is not None:
            # Handle nan/inf values and ensure we don't have gradients that are exactly zero
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            # Add a very small epsilon to avoid complete zeros
            grad += torch.randn_like(grad).to(device) * 1e-6
            accumulated_gradients += grad.detach()
        else:
            # If grad is None, it's likely due to no gradient flow to this input
            # Log this but continue with the computation
            print(f"Warning: No gradient for {input_name} at alpha={alpha.item():.2f}")

        # Free up memory
        del grad, loss, out, interpolated_input
        if device == 'cuda':
            torch.cuda.empty_cache()

    avg_grad = accumulated_gradients / steps
    integrated_grad = (input_tensor - baseline) * avg_grad

    if input_name == 'mix':
        return integrated_grad.abs().mean()
    elif input_name == 'face':
        # For face keypoints, use a different aggregation method to preserve more information
        # Return the mean of absolute values across frames, keeping landmark dimensions
        return integrated_grad.abs().mean(dim=0)  # Only average across batch dimension
    else:
        return integrated_grad.abs().mean(dim=(0, 1, 2))

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
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    print(f"Loading {args.model} model to {DEVICE}...")
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(DEVICE)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(DEVICE)
    model.eval()

    results = defaultdict(lambda: defaultdict(lambda: {
        'face_saliency': [],
        'body_saliency': [],
        'mix_saliency': []
    }))

    print(f"Calculating gradient saliency for all {len(dataset)} samples...")
    for sample in tqdm(dataloader):
        artist_name = sample['artist'][0]
        song_name = sample['song'][0]
        face_sal, body_sal, mix_sal = calculate_saliency(model, sample, DEVICE)

        if face_sal is not None:
            results[artist_name][song_name]['face_saliency'].append(face_sal.detach().cpu().tolist())
        if body_sal is not None:
            results[artist_name][song_name]['body_saliency'].append(body_sal.detach().cpu().tolist())
        if mix_sal is not None:
            results[artist_name][song_name]['mix_saliency'].append(mix_sal.detach().cpu().item())

        counter += 1
        if counter == 2:
            break

    output_path = f"results/integrated_gradients_new_baseline_{args.model}.json"
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

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
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "sdr", "sir", "sar"],
        help="Loss function to use for gradient saliency calculation",
    )
    args = parser.parse_args()
    main(args)
