import argparse
import json
import os
import sys
import random
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

# Project + external (vovit) paths
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

def temporal_shuffle(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied sample with face/body time dimension shuffled.
    Assumes tensors shaped (B, T, C, K)."""
    new_sample = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
    for key in ["face", "body"]:
        if key in new_sample and new_sample[key] is not None:
            tensor = new_sample[key]
            # Expect shape (B,T,C,K)
            if tensor.ndim == 4:
                T = tensor.shape[1]
                perm = torch.randperm(T)
                new_sample[key] = tensor[:, perm, :, :]
    return new_sample

def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = SaragaAudiovisualDataset(
        data_path=os.getenv("DATASET_PATH"),
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=ModelType.BODY_FACE if args.model == "face_body" else (ModelType.FACE if args.model=="face" else ModelType.BODY),
        audio_type=AudioType.VOCAL,
        metadata_path=args.metadata_path,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=DEVICE=="cuda")

    model, _ = load_model(args.model)

    baseline_scores: List[float] = []
    shuffled_scores: List[float] = []
    skipped_samples = 0
    skipped_existing = 0

    # Accumulate per (artist, song)
    song_acc: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    summary_dir = os.path.join(args.output_dir, f"temporal_shuffle_{args.model}")
    os.makedirs(summary_dir, exist_ok=True)
    per_song_dir = os.path.join(summary_dir, "per_song")
    os.makedirs(per_song_dir, exist_ok=True)

    def avg(lst: List[float]):
        return sum(lst)/len(lst) if lst else None

    def build_song_entry(artist: str, song: str, vals: Dict[str, List[float]]):
        b_mean = avg(vals["baseline"]) if vals["baseline"] else None
        s_mean = avg(vals["shuffled"]) if vals["shuffled"] else None
        entry = {
            "artist": artist,
            "song": song,
            "n_chunks": len(vals["baseline"]),
            "baseline_mean": b_mean,
            "shuffled_mean": s_mean,
            "delta_mean": (b_mean - s_mean) if (b_mean is not None and s_mean is not None) else None,
            "delta_mean_chunkwise": avg(vals["delta"]) if vals["delta"] else None,
        }
        return entry

    for sample in tqdm(dataloader, desc="Samples"):
        artist = sample.get("artist", ["unknown"])[0]
        song = sample.get("song", ["unknown"])[0]
        safe_artist = artist.replace("/", "_").replace(" ", "_") if isinstance(artist, str) else str(artist)
        safe_song = song.replace("/", "_").replace(" ", "_") if isinstance(song, str) else str(song)
        song_entry_path = os.path.join(per_song_dir, f"{safe_artist}__{safe_song}.json")
        # Skip if already computed unless force
        if os.path.exists(song_entry_path) and not args.force:
            skipped_existing += 1
            continue
        key = (artist, song)
        if key not in song_acc:
            song_acc[key] = {"baseline": [], "shuffled": [], "delta": []}

        target = sample["target"].float().squeeze(0)
        est = forward_model(model, sample)
        if est is None:
            skipped_samples += 1
            continue
        base_metric = si_sdr(est.squeeze(0), target).item()

        shuffled_sample = temporal_shuffle(sample)
        est_shuf = forward_model(model, shuffled_sample)
        if est_shuf is None:
            skipped_samples += 1
            continue
        shuf_metric = si_sdr(est_shuf.squeeze(0), target).item()

        baseline_scores.append(base_metric)
        shuffled_scores.append(shuf_metric)
        song_acc[key]["baseline"].append(base_metric)
        song_acc[key]["shuffled"].append(shuf_metric)
        song_acc[key]["delta"].append(base_metric - shuf_metric)

        # Replace incremental write block
        with open(song_entry_path, "w") as f_song:
            json.dump(build_song_entry(artist, song, song_acc[key]), f_song, indent=2)

    if len(baseline_scores) == 0:
        print("No valid samples after skipping missing modality cases.")
        return
    
    # Removed in-script summary; use aggregate_ablation_results.py with --per-sample-subdir per_song
    # Example:
    # python ablation_studies/aggregate_ablation_results.py \
    #   --input-root "${args.output_dir}/temporal_shuffle_{args.model}" \
    #   --per-sample-subdir per_song

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal shuffle ablation (destroy AV synchrony)")
    parser.add_argument("--model", type=str, default="face_body", choices=["face", "body", "face_body"], help="Model variant")
    parser.add_argument("--metadata-path", type=str, default=os.path.join(os.getenv("DATASET_PATH", "dataset"), "dataset_metadata.json"))
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recompute songs even if per-song JSON exists")
    args = parser.parse_args()
    run(args)
