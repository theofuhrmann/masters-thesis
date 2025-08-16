import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
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
DATASET_METADATA_PATH = os.path.join(DATASET_PATH, "dataset_metadata.json")

speech_mean_face = torch.from_numpy(
    np.load("../tools/speech_mean_face.npy")
).float()


def calculate_saliency(model, sample, device="cpu"):
    mix = sample["mix"].float().to(device)
    face = (
        torch.nan_to_num(sample["face"].float(), nan=0.0).to(device)
        if "face" in sample and sample["face"] is not None
        else None
    )
    body = (
        torch.nan_to_num(sample["body"].float(), nan=0.0).to(device)
        if "body" in sample and sample["body"] is not None
        else None
    )

    face_sal, body_sal, mix_sal = None, None, None
    if face is not None:
        face_sal = integrated_gradients(
            face, "face", model, sample, device=device
        )
    if body is not None:
        body_sal = integrated_gradients(
            body, "body", model, sample, device=device
        )
    mix_sal = integrated_gradients(mix, "mix", model, sample, device=device)

    return face_sal, body_sal, mix_sal


def integrated_gradients(
    input_tensor,
    input_name,
    model,
    sample,
    baseline=None,
    steps=25,
    device="cpu",
):
    """
    Computes Integrated Gradients for a given input tensor.

    input_tensor: torch.Tensor (will have requires_grad set in this function)
    input_name: str, must be one of ['mix', 'face', 'body']
    model: the VoViT model
    sample: the input sample
    baseline: baseline tensor (same shape as input_tensor), defaults to appropriate baseline
    steps: number of integration steps
    """
    input_tensor = input_tensor.detach().clone()

    if baseline is None:
        if input_name == "face":
            baseline = (
                speech_mean_face.unsqueeze(0)
                .unsqueeze(0)
                .expand_as(input_tensor)
                .clone()
                .detach()
                .to(device)
            )
        else:
            baseline = torch.zeros_like(input_tensor).to(device)

    mix_orig = sample["mix"].float().to(device)
    face_orig = (
        torch.nan_to_num(sample["face"].float(), nan=0.0).to(device)
        if "face" in sample and sample["face"] is not None
        else None
    )
    body_orig = (
        torch.nan_to_num(sample["body"].float(), nan=0.0).to(device)
        if "body" in sample and sample["body"] is not None
        else None
    )
    target_orig = sample["target"].float().to(device)

    alphas = torch.linspace(0, 1, steps).to(device)
    accumulated_gradients = torch.zeros_like(input_tensor).to(device)

    for alpha in alphas:
        interpolated_input = baseline + alpha * (input_tensor - baseline)
        interpolated_input.requires_grad_(True)

        mix, face, body = mix_orig, face_orig, body_orig

        if input_name == "mix":
            mix = interpolated_input
        elif input_name == "face":
            face = interpolated_input
        elif input_name == "body":
            body = interpolated_input

        model.zero_grad()

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
            grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            accumulated_gradients += grad.detach()
        else:
            print(
                f"Warning: No gradient for {input_name} at alpha={alpha.item():.2f}"
            )

        del grad, loss, out, interpolated_input
        if device == "cuda":
            torch.cuda.empty_cache()

    avg_grad = accumulated_gradients / steps
    integrated_grad = (input_tensor - baseline) * avg_grad

    if input_name == "mix":
        return integrated_grad.abs().mean()
    elif input_name == "face":
        return integrated_grad.abs().mean(dim=(0, 1, 2))
    else:
        return integrated_grad.abs().mean(dim=(0, 1, 2))


def main(args):
    model_type_map = {
        "face": ModelType.FACE,
        "body": ModelType.BODY,
        "face_body": ModelType.BODY_FACE,
    }
    model_type = model_type_map[args.model]

    results_dir = f"results/integrated_gradients_new_baseline_{args.model}"
    os.makedirs(results_dir, exist_ok=True)

    with open(DATASET_METADATA_PATH, "r") as f:
        full_metadata = json.load(f)

    print(f"Loading {args.model} model to {DEVICE}...")
    if model_type == ModelType.FACE:
        model = VoViT_f(pretrained=True, debug={}).to(DEVICE)
    elif model_type == ModelType.BODY:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        model = VoViT_fb(pretrained=True, debug={}).to(DEVICE)
    model.eval()

    total_combinations = sum(len(songs) for songs in full_metadata.values())
    processed = 0
    
    temp_metadata_path = "temp_metadata_single_song.json"
    
    for artist_name, songs in full_metadata.items():
        for song_name in songs:
            processed += 1
            
            safe_artist_name = artist_name.replace("/", "_").replace(" ", "_")
            safe_song_name = song_name.replace("/", "_").replace(" ", "_")
            output_path = os.path.join(results_dir, f"{safe_artist_name}_{safe_song_name}.json")
            
            if os.path.exists(output_path):
                print(f"\nSkipping {processed}/{total_combinations}: {artist_name} - {song_name} (already processed)")
                continue
            
            print(f"\nProcessing {processed}/{total_combinations}: {artist_name} - {song_name}")
            
            temp_metadata = {artist_name: {song_name: songs[song_name]}}
            
            with open(temp_metadata_path, "w") as f:
                json.dump(temp_metadata, f)
            
            try:
                dataset = SaragaAudiovisualDataset(
                    data_path=DATASET_PATH,
                    audiorate=AUDIO_RATE,
                    chunk_duration=DURATION,
                    model_type=model_type,
                    audio_type=AudioType.VOCAL,
                    metadata_path=temp_metadata_path,
                    layout=(
                        ["violin", "vocal", "mridangam"]
                        if model_type == ModelType.BODY_VIOLIN_FUSION
                        else None
                    ),
                    keep_highest_correlation_chunks=True,
                    correlation_filter_percentage=args.correlation_filter_percentage,
                    correlation_filter_top=args.correlation_filter_top,
                )
                
                if len(dataset.items) == 0:
                    print(f"  Skipping {artist_name} - {song_name}: no valid chunks")
                    continue
                
                use_pin_memory = DEVICE == "cuda"
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=use_pin_memory,
                )
                
                face_saliencies = []
                body_saliencies = []
                mix_saliencies = []
                
                print(f"  Processing {len(dataset)} chunks...")
                for i, sample in enumerate(tqdm(dataloader, desc=f"  Chunks")):
                    try:
                        face_sal, body_sal, mix_sal = calculate_saliency(model, sample, DEVICE)
                        
                        if face_sal is not None:
                            face_saliencies.append(face_sal.detach().cpu())
                        if body_sal is not None:
                            body_saliencies.append(body_sal.detach().cpu())
                        if mix_sal is not None:
                            mix_saliencies.append(mix_sal.detach().cpu().item())
                        
                        # Clear CUDA cache after each chunk
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                            
                    except torch.cuda.OutOfMemoryError:
                        print(f"    CUDA OOM error on chunk {i}, skipping...")
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        print(f"    Error processing chunk {i}: {e}")
                        continue
                
                song_results = {
                    "face_saliency": None,
                    "body_saliency": None,
                    "mix_saliency": None,
                }
                
                if face_saliencies:
                    stacked_face = torch.stack(face_saliencies, dim=0)  # Shape: (num_chunks, 68)
                    averaged_face = stacked_face.mean(dim=0)  # Shape: (68,)
                    song_results["face_saliency"] = averaged_face.tolist()
                
                if body_saliencies:
                    stacked_body = torch.stack(body_saliencies, dim=0)  # Shape: (num_chunks, 55)
                    averaged_body = stacked_body.mean(dim=0)  # Shape: (55,)
                    song_results["body_saliency"] = averaged_body.tolist()
                
                if mix_saliencies:
                    averaged_mix = sum(mix_saliencies) / len(mix_saliencies)
                    song_results["mix_saliency"] = averaged_mix
                
                with open(output_path, "w") as f:
                    json.dump({
                        "artist": artist_name,
                        "song": song_name,
                        "results": song_results
                    }, f, indent=4)
                
                print(f"  Saved results to {output_path}")
                
            except Exception as e:
                print(f"  Error processing {artist_name} - {song_name}: {e}")
                continue
            finally:
                if 'dataset' in locals():
                    del dataset
                if 'dataloader' in locals():
                    del dataloader
                if 'song_results' in locals():
                    del song_results
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    
    if os.path.exists(temp_metadata_path):
        os.remove(temp_metadata_path)

    print(f"\nDone. Results saved in {results_dir}/")
    print("To combine all results into a single file, you can run a separate script.")


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
        "--correlation-filter-percentage",
        type=float,
        default=0.1,
        help="Percentage of chunks to keep when filtering by correlation (0.0-1.0, default: 0.1 for 10%%)",
    )
    parser.add_argument(
        "--correlation-filter-top",
        action="store_true",
        default=True,
        help="Keep top correlation chunks (default: True)",
    )
    args = parser.parse_args()
    main(args)
