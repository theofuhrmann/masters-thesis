import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv()
vovit_path = os.path.abspath(os.getenv("VOVIT_PATH"))
sys.path.insert(0, vovit_path)

from vovit import VoViT_b, VoViT_f  # type: ignore # noqa: E402
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


def analyze_film_modulation(model, sample, device, debug=False, meta=None, debug_file_handle=None):
    """
    Analyzes FiLM modulation.
    If debug=True, logs raw vs interpreted stats per-sample to a JSONL file.
    Adds gamma distribution percentiles, feature-level std, and optional
    correlation between |gamma| time profile and attention focus (if available).
    """
    audio = sample["mix"].to(device)
    face = sample.get("face")
    body = sample.get("body")
    if face is not None:
        face = face.to(device)
    if body is not None:
        body = body.to(device)

    with torch.no_grad():
        if face is not None and body is not None:
            output = model(audio, face, body)
        elif face is not None:
            output = model(audio, face)
        elif body is not None:
            output = model(audio, body)
        else:
            output = model(audio, None)

    fusion_module = model.vovit.avse.audio_network.fusion_module
    film_gamma = getattr(fusion_module, "last_film_gamma", None)
    film_beta = getattr(fusion_module, "last_film_beta", None)
    audio_in = getattr(fusion_module, "last_audio_in", None)
    film_query = getattr(fusion_module, "last_film_query", None)
    attn_weights = getattr(fusion_module, "last_attn_weights", None)

    if film_gamma is None or film_beta is None:
        return None

    raw_gamma_t = film_gamma.detach().float().cpu()
    raw_beta_t = film_beta.detach().float().cpu()
    raw_gamma_np = raw_gamma_t.numpy()
    raw_beta_np = raw_beta_t.numpy()

    gamma = raw_gamma_np.copy()
    beta = raw_beta_np.copy()

    def sanitize(arr, name):
        finite_mask = np.isfinite(arr)
        frac_finite = finite_mask.mean()
        if frac_finite < 0.5:
            return None, {f"{name}_finite_fraction": float(frac_finite), f"{name}_skipped": True}
        if not np.all(finite_mask):
            med = np.median(arr[finite_mask]) if finite_mask.any() else 0.0
            arr = arr.copy(); arr[~finite_mask] = med
        stats = {f"{name}_finite_fraction": float(frac_finite), f"{name}_max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0}
        return arr, stats

    gamma, gamma_diag = sanitize(gamma, "gamma")
    beta, beta_diag = sanitize(beta, "beta")
    if gamma is None or beta is None:
        return None

    raw_gamma = gamma
    scale = 1.0 + raw_gamma
    bias = beta

    gamma_abs = np.abs(raw_gamma)
    gamma_percentiles = {p: float(np.percentile(gamma_abs, p)) for p in [50, 75, 90, 95, 99]}
    gamma_abs_mean = float(gamma_abs.mean())
    gamma_abs_std = float(gamma_abs.std())
    feature_gamma_std = np.std(raw_gamma, axis=(0, 1))
    feature_gamma_abs_mean = np.mean(gamma_abs, axis=(0, 1))

    rel_change_stats = {}
    try:
        if audio_in is not None and film_query is not None:
            a_in = audio_in.detach().float().cpu().numpy()
            q = film_query.detach().float().cpu().numpy()
            if a_in.shape == q.shape:
                denom = np.maximum(np.abs(a_in), 1e-2)
                rel = (q - a_in) / denom
                rel = np.clip(rel, -50, 50)
                rel_change_stats = {
                    "relative_change_mean_abs": float(np.mean(np.abs(rel))),
                    "relative_change_p95_abs": float(np.percentile(np.abs(rel), 95)),
                    "relative_change_inactive_frac": float(np.mean(np.abs(rel) < 0.01)),
                }
    except Exception:
        pass

    feature_mean_scale = np.nanmean(scale, axis=(0, 1))
    feature_mean_bias = np.nanmean(bias, axis=(0, 1))
    delta = scale - 1.0
    delta_abs = np.abs(delta)
    scale_dev_from_unity = float(delta_abs.mean())
    bias_mag = float(np.mean(np.abs(bias)))

    recomputed = float(np.abs(scale - 1.0).mean())
    consistent = abs(recomputed - scale_dev_from_unity) < 1e-8

    boosted = np.where(feature_mean_scale > feature_mean_scale.mean() + feature_mean_scale.std())[0].tolist()
    suppressed = np.where(feature_mean_scale < feature_mean_scale.mean() - feature_mean_scale.std())[0].tolist()
    pos_bias = np.where(feature_mean_bias > feature_mean_bias.mean() + feature_mean_bias.std())[0].tolist()
    neg_bias = np.where(feature_mean_bias < feature_mean_bias.mean() - feature_mean_bias.std())[0].tolist()

    gamma_attention_focus_corr = None
    if attn_weights is not None:
        try:
            attn = attn_weights.detach().float().cpu().numpy()
            if attn.ndim == 4 and attn.shape[0] == raw_gamma.shape[0] and attn.shape[2] == raw_gamma.shape[1]:
                attn_mean = attn.mean(axis=1)
                p = np.clip(attn_mean, 1e-9, 1.0)
                entropy = -np.sum(p * np.log(p), axis=-1)
                gamma_t_mean = gamma_abs.mean(axis=-1)
                ent_vec = entropy.flatten()
                gamma_vec = gamma_t_mean.flatten()
                if np.std(ent_vec) > 0 and np.std(gamma_vec) > 0:
                    corr = np.corrcoef(gamma_vec, -ent_vec)[0, 1]
                    gamma_attention_focus_corr = float(corr)
        except Exception:
            gamma_attention_focus_corr = None

    # --- Temporal correlation with audio RMS ---
    gamma_audio_corr = None
    if audio is not None:
        hop_length = max(1, audio.shape[-1] // gamma_t.shape[1])
        audio_np = audio.detach().cpu().numpy().squeeze()
        rms = librosa.feature.rms(y=audio_np, hop_length=hop_length, frame_length=hop_length*2).squeeze()
        min_len = min(len(rms), gamma_t.shape[1])
        gamma_mean = np.abs(raw_gamma_t.numpy()[0, :min_len, :]).mean(axis=-1)
        if np.std(gamma_mean) > 0 and np.std(rms[:min_len]) > 0:
            gamma_audio_corr = float(np.corrcoef(gamma_mean, rms[:min_len])[0, 1])

    # --- FiLM knock-out ΔSI-SDR ---
    delta_sisdr = None
    try:
        # store original gamma/beta
        orig_gamma = fusion_module.last_film_gamma.clone()
        orig_beta = fusion_module.last_film_beta.clone()
        # knockout
        fusion_module.last_film_gamma[:] = 0.0
        fusion_module.last_film_beta[:] = 0.0
        with torch.no_grad():
            if face is not None and body is not None:
                output_ko = model(audio, face, body)
            elif face is not None:
                output_ko = model(audio, face)
            elif body is not None:
                output_ko = model(audio, body)
            else:
                output_ko = model(audio, None)
        # restore gamma/beta
        fusion_module.last_film_gamma[:] = orig_gamma
        fusion_module.last_film_beta[:] = orig_beta
        # SI-SDR delta: placeholder (replace with your SI-SDR function)
        if hasattr(output, "sisdr") and hasattr(output_ko, "sisdr"):
            delta_sisdr = float(output.sisdr - output_ko.sisdr)
    except Exception:
        delta_sisdr = None

    analysis = {
        "diagnostics": {**gamma_diag, **beta_diag},
        "scale_stats": {
            "mean": float(np.mean(scale)),
            "std": float(np.std(scale)),
            "min": float(np.min(scale)),
            "max": float(np.max(scale)),
            "median": float(np.median(scale)),
            "feature_mean": feature_mean_scale.tolist(),
        },
        "bias_stats": {
            "mean": float(np.mean(bias)),
            "std": float(np.std(bias)),
            "min": float(np.min(bias)),
            "max": float(np.max(bias)),
            "median": float(np.median(bias)),
            "feature_mean": feature_mean_bias.tolist(),
        },
        "gamma_stats": {
            "abs_mean": gamma_abs_mean,
            "abs_std": gamma_abs_std,
            "percentiles": gamma_percentiles,
            "feature_abs_mean": feature_gamma_abs_mean.tolist(),
            "feature_std": feature_gamma_std.tolist(),
            "attention_focus_corr": gamma_attention_focus_corr,
            "audio_corr": gamma_audio_corr,
        },
        "modulation_effects": {
            "highly_boosted_features": boosted,
            "suppressed_features": suppressed,
            "positive_bias_features": pos_bias,
            "negative_bias_features": neg_bias,
        },
        "modulation_intensity": {
            "scale_deviation_from_unity": scale_dev_from_unity,
            "bias_magnitude": bias_mag,
            "total_modulation": scale_dev_from_unity + bias_mag,
        },
        "relative_change": rel_change_stats,
        "film_knockout_delta_sisdr": delta_sisdr,
    }

    if debug and debug_file_handle is not None:
        dbg = {
            **(meta or {}),
            "gamma_audio_corr": gamma_audio_corr,
            "delta_sisdr": delta_sisdr,
        }
        debug_file_handle.write(json.dumps(dbg) + "\n")
        debug_file_handle.flush()

    return analysis


# --- Rest of your print_film_summary and main() code remains unchanged ---
# Add a print section for temporal correlation and ΔSI-SDR:

def print_film_summary(results, model_type):
    print(f"\n{'='*60}")
    print(f"FiLM MODULATION ANALYSIS SUMMARY - {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    gamma_audio_corrs = []
    delta_sisdr_list = []

    for artist_data in results.values():
        for song_data in artist_data.values():
            for analysis in song_data:
                if analysis:
                    if analysis["gamma_stats"].get("audio_corr") is not None:
                        gamma_audio_corrs.append(analysis["gamma_stats"]["audio_corr"])
                    if analysis.get("film_knockout_delta_sisdr") is not None:
                        delta_sisdr_list.append(analysis["film_knockout_delta_sisdr"])

    if gamma_audio_corrs:
        print(f"\nTemporal correlation |γ| vs audio RMS:")
        print(f"  Mean: {np.mean(gamma_audio_corrs):.4f} ± {np.std(gamma_audio_corrs):.4f}")
        print(f"  Range: [{np.min(gamma_audio_corrs):.4f}, {np.max(gamma_audio_corrs):.4f}]")

    if delta_sisdr_list:
        print(f"\nFiLM Knock-out ΔSI-SDR:")
        print(f"  Mean: {np.mean(delta_sisdr_list):.4f} ± {np.std(delta_sisdr_list):.4f}")
        print(f"  Range: [{np.min(delta_sisdr_list):.4f}, {np.max(delta_sisdr_list):.4f}]")

# --- main() remains mostly unchanged, calling analyze_film_modulation in the loop ---



def main(args):
    model_type_map = {
        "vocal": ModelType.FACE_VOCAL_FUSION,
        "violin": ModelType.BODY_VIOLIN_FUSION,
    }
    model_type = model_type_map[args.model]

    audio_type_map = {
        "vocal": AudioType.VOCAL,
        "violin": AudioType.VIOLIN,
    }
    audio_type = audio_type_map[args.model]

    dataset = SaragaAudiovisualDataset(
        data_path=DATASET_PATH,
        audiorate=AUDIO_RATE,
        chunk_duration=DURATION,
        model_type=model_type,
        audio_type=audio_type,
        metadata_path=DATASET_METADATA_PATH,
        face_keypoints_source="mmpose",
        layout=(
            ["violin", "vocal", "mridangam"]
            if args.model == "violin"
            else None
        )
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Loading {args.model} model to {DEVICE}...")
    if model_type == ModelType.FACE_VOCAL_FUSION:
        model = VoViT_f(pretrained=True, debug={}, face_keypoints_source="mmpose").to(DEVICE)
    elif model_type == ModelType.BODY_VIOLIN_FUSION:
        model = VoViT_b(pretrained=True, debug={}).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.eval()

    results = defaultdict(lambda: defaultdict(list))
    print(f"Analyzing FiLM modulation for all {len(dataset)} samples...")
    
    debug_fh = None
    if args.debug_film:
        os.makedirs("results", exist_ok=True)
        debug_fh = open(f"results/film_debug_{args.model}.jsonl", "w")

    for idx, sample in enumerate(tqdm(dataloader)):
        artist_name = sample["artist"][0]
        song_name = sample["song"][0]
        try:
            analysis = analyze_film_modulation(
                model,
                sample,
                DEVICE,
                debug=args.debug_film,
                meta={"artist": artist_name, "song": song_name, "idx": idx},
                debug_file_handle=debug_fh,
            )
        except RuntimeError as e:
            print(f"Error processing sample for artist: {artist_name}, song: {song_name}")
            print(f"Sample shape: {sample['mix'].shape}")
            print(e)
            continue
        results[artist_name][song_name].append(analysis)

    if debug_fh is not None:
        debug_fh.close()

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/film_analysis_{args.model}_mha_film.json"
    print(f"Saving results to {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print comprehensive summary
    print_film_summary(results, args.model)

    print(f"\nDetailed results saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VoViT FiLM Modulation Analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vocal",
        choices=["vocal", "violin"],
        help="Type of VoViT model to use",
    )
    parser.add_argument(
        "--debug_film", action="store_true", help="Enable per-sample FiLM debug logging"
    )

    args = parser.parse_args()
    main(args)
