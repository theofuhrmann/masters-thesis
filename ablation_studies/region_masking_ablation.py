import argparse
import json
import os
import sys
from typing import Dict, Any, Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv()
vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from vovit.display.dataloaders_new import (  # type: ignore  # noqa: E402
    SaragaAudiovisualDataset,
    AudioType,
    ModelType,
)
from ablation_studies.ablation_utils import si_sdr, load_model, forward_model  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_RATE = 16384
DURATION = 4.0

# Assumed landmark index groupings (modify if you have a mapping file)
MOUTH_IDXS = list(range(48, 68))  # 68-point face (last 20 are mouth)
NOSE_IDXS = list(range(27, 36))   # upper + lower nose region
BODY_IDXS = None  # We'll zero ALL body keypoints instead of subset

@torch.no_grad()
def apply_mask(sample: Dict[str, Any], mask_type: str, fill_method: str, mean_face: torch.Tensor | None) -> Dict[str, Any]:
    """Return a masked copy of sample.

    mask_type: 'mouth' | 'nose' | 'body'
    fill_method: 'zero' | 'mean_face'
    mean_face: (C,K) tensor if fill_method == 'mean_face'
    """
    new_sample = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
    if mask_type in ("mouth", "nose") and "face" in new_sample and new_sample["face"] is not None:
        face = new_sample["face"]  # (B,T,C,K)
        idxs = MOUTH_IDXS if mask_type == "mouth" else NOSE_IDXS
        if fill_method == "zero":
            face[:, :, :, idxs] = 0.0
        elif fill_method == "mean_face" and mean_face is not None:
            # mean_face shape (C,K); broadcast to (1,1,C,len(idxs))
            mf = mean_face.to(face.device)
            face[:, :, :, idxs] = mf[:, idxs].view(1, 1, mf.shape[0], len(idxs))
        else:
            face[:, :, :, idxs] = 0.0  # fallback
    elif mask_type == "body" and "body" in new_sample and new_sample["body"] is not None:
        new_sample["body"] = torch.zeros_like(new_sample["body"])
    return new_sample

def run(args):
    dataset = SaragaAudiovisualDataset(
        data_path=os.getenv("DATASET_PATH"),
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=ModelType.BODY_FACE if args.model == "face_body" else (ModelType.FACE if args.model=="face" else ModelType.BODY),
        audio_type=AudioType.VOCAL,
        metadata_path=args.metadata_path,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=DEVICE=="cuda")

    model, model_type = load_model(args.model)

    mean_face = None
    if args.fill_method == "mean_face":
        mean_face_path = os.path.join(project_root, "tools", "speech_mean_face.npy")
        if os.path.exists(mean_face_path):
            mean_face = torch.from_numpy(np.load(mean_face_path)).float()
        else:
            print(f"Warning: mean face file not found at {mean_face_path}; falling back to zeros.")

    # Accumulators for global averaging
    baseline_scores: List[float] = []
    mouth_scores: List[float] = []
    nose_scores: List[float] = []
    body_scores: List[float] = []
    skipped_samples = 0
    skipped_existing = 0

    # Per (artist,song) accumulation: store lists of values and chunkwise deltas
    song_acc: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    summary_dir = os.path.join(args.output_dir, f"region_masking_{args.model}")
    os.makedirs(summary_dir, exist_ok=True)
    per_song_dir = os.path.join(summary_dir, "per_song")
    os.makedirs(per_song_dir, exist_ok=True)

    def avg(lst: List[float]):
        return sum(lst)/len(lst) if lst else None

    def build_song_entry(artist: str, song: str, vals: Dict[str, List[float]]):
        base_mean = avg(vals["baseline"]) if vals["baseline"] else None
        mouth_mean = avg(vals["mouth"]) if vals["mouth"] else None
        nose_mean = avg(vals["nose"]) if vals["nose"] else None
        body_mean = avg(vals["body"]) if vals["body"] else None
        entry = {
            "artist": artist,
            "song": song,
            "n_chunks": len(vals["baseline"]),
            "baseline_mean": base_mean,
            "mouth_mean": mouth_mean,
            "nose_mean": nose_mean,
            "body_mean": body_mean,
            "delta_mouth_mean": (base_mean - mouth_mean) if (base_mean is not None and mouth_mean is not None) else None,
            "delta_nose_mean": (base_mean - nose_mean) if (base_mean is not None and nose_mean is not None) else None,
            "delta_body_mean": (base_mean - body_mean) if (base_mean is not None and body_mean is not None) else None,
            "delta_mouth_mean_chunkwise": avg(vals["delta_mouth"]) if vals["delta_mouth"] else None,
            "delta_nose_mean_chunkwise": avg(vals["delta_nose"]) if vals["delta_nose"] else None,
            "delta_body_mean_chunkwise": avg(vals["delta_body"]) if vals["delta_body"] else None,
        }
        return entry

    for sample in tqdm(dataloader, desc="Samples"):
        artist = sample.get("artist", ["unknown"])[0]
        song = sample.get("song", ["unknown"])[0]
        key = (artist, song)
        safe_artist = artist.replace("/", "_").replace(" ", "_") if isinstance(artist, str) else str(artist)
        safe_song = song.replace("/", "_").replace(" ", "_") if isinstance(song, str) else str(song)
        song_entry_path = os.path.join(per_song_dir, f"{safe_artist}__{safe_song}.json")
        # Skip recomputation if per-song JSON already exists and not forcing
        if os.path.exists(song_entry_path) and not args.force:
            skipped_existing += 1
            continue
        if key not in song_acc:
            song_acc[key] = {
                "baseline": [],
                "mouth": [],
                "nose": [],
                "body": [],
                "delta_mouth": [],
                "delta_nose": [],
                "delta_body": [],
            }

        target = sample["target"].float().squeeze(0)
        est = forward_model(model, sample)
        if est is None:
            skipped_samples += 1
            continue
        est = est.squeeze(0)
        base_val = si_sdr(est, target).item()
        baseline_scores.append(base_val)
        song_acc[key]["baseline"].append(base_val)

        # Conditional masks depending on model modalities
        mouth_val = None
        nose_val = None
        body_val = None

        has_face = (model_type in [ModelType.FACE, ModelType.BODY_FACE]) and (sample.get("face") is not None)
        has_body = (model_type in [ModelType.BODY, ModelType.BODY_FACE]) and (sample.get("body") is not None)

        if has_face:
            est_mouth = forward_model(model, apply_mask(sample, "mouth", args.fill_method, mean_face))
            if est_mouth is not None:
                mouth_val = si_sdr(est_mouth.squeeze(0), target).item()
                mouth_scores.append(mouth_val)
                song_acc[key]["mouth"].append(mouth_val)
                song_acc[key]["delta_mouth"].append(base_val - mouth_val)
            est_nose = forward_model(model, apply_mask(sample, "nose", args.fill_method, mean_face))
            if est_nose is not None:
                nose_val = si_sdr(est_nose.squeeze(0), target).item()
                nose_scores.append(nose_val)
                song_acc[key]["nose"].append(nose_val)
                song_acc[key]["delta_nose"].append(base_val - nose_val)
        if has_body:
            est_body = forward_model(model, apply_mask(sample, "body", args.fill_method, mean_face))
            if est_body is not None:
                body_val = si_sdr(est_body.squeeze(0), target).item()
                body_scores.append(body_val)
                song_acc[key]["body"].append(body_val)
                song_acc[key]["delta_body"].append(base_val - body_val)

        # After updating song_acc for this (artist,song), write/update its per-song JSON
        with open(song_entry_path, "w") as f_song:
            json.dump(build_song_entry(artist, song, song_acc[key]), f_song, indent=2)

    if len(baseline_scores) == 0:
        print("No valid samples processed.")
        return

    # Build final per-song aggregated list
    per_song_entries = [build_song_entry(a, s, vals) for (a, s), vals in song_acc.items()]
    aggregated_per_song_path = os.path.join(per_song_dir, "aggregated_per_song.json")
    with open(aggregated_per_song_path, "w") as f_all:
        json.dump(per_song_entries, f_all, indent=2)
    print(f"Saved per-song aggregation to {aggregated_per_song_path}")

    # Global summary across songs (song-level averaging)
    def collect(field: str):
        vals = [e[field] for e in per_song_entries if e.get(field) is not None]
        return sum(vals)/len(vals) if vals else None

    baseline_song_mean = collect("baseline_mean")
    mouth_song_mean = collect("mouth_mean")
    nose_song_mean = collect("nose_mean")
    body_song_mean = collect("body_mean")

    summary = {
        "model": args.model,
        "fill_method": args.fill_method,
        "n_songs": len(per_song_entries),
        "skipped_samples": skipped_samples,
        "skipped_existing_songs": skipped_existing,
        "force_recompute": args.force,
        "baseline_SI_SDR_song_mean": baseline_song_mean,
        "mouth_mask_SI_SDR_song_mean": mouth_song_mean,
        "nose_mask_SI_SDR_song_mean": nose_song_mean,
        "body_mask_SI_SDR_song_mean": body_song_mean,
        "delta_mouth_song_mean_difference_of_means": (baseline_song_mean - mouth_song_mean) if (baseline_song_mean is not None and mouth_song_mean is not None) else None,
        "delta_nose_song_mean_difference_of_means": (baseline_song_mean - nose_song_mean) if (baseline_song_mean is not None and nose_song_mean is not None) else None,
        "delta_body_song_mean_difference_of_means": (baseline_song_mean - body_song_mean) if (baseline_song_mean is not None and body_song_mean is not None) else None,
        "delta_mouth_song_mean_chunkwise": collect("delta_mouth_mean_chunkwise"),
        "delta_nose_song_mean_chunkwise": collect("delta_nose_mean_chunkwise"),
        "delta_body_song_mean_chunkwise": collect("delta_body_mean_chunkwise"),
        "aggregation_level": "per_song",
    }

    summary_path = os.path.join(summary_dir, "summary.json")
    with open(summary_path, "w") as f_sum:
        json.dump(summary, f_sum, indent=4)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Region masking ablation (mouth vs nose vs body)")
    parser.add_argument("--model", type=str, default="face_body", choices=["face", "body", "face_body"], help="Model variant")
    parser.add_argument("--metadata-path", type=str, default=os.path.join(os.getenv("DATASET_PATH", "dataset"), "dataset_metadata.json"))
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--fill-method", type=str, default="mean_face", choices=["zero", "mean_face"], help="How to fill masked face landmarks")
    parser.add_argument("--force", action="store_true", help="Recompute songs even if per-song JSON exists")
    args = parser.parse_args()
    run(args)
