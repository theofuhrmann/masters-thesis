import argparse
import json
import os
import sys
import numpy as np

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# --- Added utility for safe filenames ---
import re

def _sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "-")
    name = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]  # limit length

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est.float().flatten()
    ref = ref.float().flatten()
    ref_zm = ref - ref.mean()
    est_zm = est - est.mean()
    s_target = (torch.dot(est_zm, ref_zm) / (ref_zm.pow(2).sum() + eps)) * ref_zm
    e_noise = est_zm - s_target
    ratio = (s_target.pow(2).sum() + eps) / (e_noise.pow(2).sum() + eps)
    return 10 * torch.log10(ratio + eps)


# Cache for precomputed RMS values
SONG_RMS_CACHE = {}
SONG_ONSET_ENV_CACHE = {}

def _load_precomputed_rms(artist: str, song: str):
    """Load precomputed RMS (array of frame-wise values) with no fallback logic."""
    key = (artist, song)
    if key in SONG_RMS_CACHE:
        return SONG_RMS_CACHE[key]
    features_path = os.path.join(DATASET_PATH, artist, song, args.model, "audio_features.json")
    if not os.path.isfile(features_path):
        SONG_RMS_CACHE[key] = None
        return None
    with open(features_path, "r") as f:
        data = json.load(f)
    rms_vals = data.get("rms")  # expected to be a direct list/array
    arr = np.asarray(rms_vals, dtype=np.float32) if rms_vals is not None else None
    SONG_RMS_CACHE[key] = arr
    return arr

def _load_precomputed_onset_env(artist: str, song: str):
    """Load precomputed onset_env (array of frame-wise values)."""
    key = (artist, song)
    if key in SONG_ONSET_ENV_CACHE:
        return SONG_ONSET_ENV_CACHE[key]
    features_path = os.path.join(DATASET_PATH, artist, song, args.model, "audio_features.json")
    if not os.path.isfile(features_path):
        SONG_ONSET_ENV_CACHE[key] = None
        return None
    with open(features_path, "r") as f:
        data = json.load(f)
    onset_vals = data.get("onset_env")
    arr = np.asarray(onset_vals, dtype=np.float32) if onset_vals is not None else None
    SONG_ONSET_ENV_CACHE[key] = arr
    return arr

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

    # --- Temporal correlation with audio RMS & onset_env ---
    gamma_rms_corr = None
    gamma_onset_env_corr = None
    rms_source = None
    onset_env_source = None
    artist_meta = (meta or {}).get("artist")
    song_meta = (meta or {}).get("song")
    pre_rms = None
    pre_onset = None
    if artist_meta and song_meta:
        pre_rms = _load_precomputed_rms(artist_meta, song_meta)
        pre_onset = _load_precomputed_onset_env(artist_meta, song_meta)
    T_gamma = raw_gamma_t.shape[1]
    # Precompute gamma mean over features per time step (for correlations)
    gamma_time_mean = np.abs(raw_gamma_t.numpy()[0, :, :]).mean(axis=-1)
    if pre_rms is not None and pre_rms.size > 1:
        T_rms = pre_rms.shape[0]
        if T_rms != T_gamma and T_rms > 1:
            x_old = np.linspace(0, 1, T_rms)
            x_new = np.linspace(0, 1, T_gamma)
            rms_interp = np.interp(x_new, x_old, pre_rms)
        else:
            rms_interp = pre_rms[:T_gamma]
        if np.std(gamma_time_mean[:len(rms_interp)]) > 0 and np.std(rms_interp) > 0:
            gamma_rms_corr = float(np.corrcoef(gamma_time_mean[:len(rms_interp)], rms_interp)[0, 1])
            rms_source = "precomputed"
    if pre_onset is not None and pre_onset.size > 1:
        T_on = pre_onset.shape[0]
        if T_on != T_gamma and T_on > 1:
            x_old = np.linspace(0, 1, T_on)
            x_new = np.linspace(0, 1, T_gamma)
            onset_interp = np.interp(x_new, x_old, pre_onset)
        else:
            onset_interp = pre_onset[:T_gamma]
        if np.std(gamma_time_mean[:len(onset_interp)]) > 0 and np.std(onset_interp) > 0:
            gamma_onset_env_corr = float(np.corrcoef(gamma_time_mean[:len(onset_interp)], onset_interp)[0, 1])
            onset_env_source = "precomputed"

    # --- FiLM knock-out ΔSI-SDR (compute baseline + disable FiLM via flag if available) ---
    target = sample.get("target")
    est = output["estimated_wav"].detach().float().to(device).squeeze(0)
    target_t = target.to(device).float().squeeze(0)
    base_sisdr = si_sdr(est, target_t).item()
    ko_sisdr = None
    prev_flag = fusion_module.film_cross_attention.use_film
    fusion_module.film_cross_attention.use_film = False
    with torch.no_grad():
        if face is not None and body is not None:
            out_ko = model(audio, face, body)
        elif face is not None:
            out_ko = model(audio, face)
        elif body is not None:
            out_ko = model(audio, body)
        else:
            out_ko = model(audio, None)
    fusion_module.film_cross_attention.use_film = prev_flag
    est_ko = out_ko["estimated_wav"].detach().float().to(device).squeeze(0)
    ko_sisdr = si_sdr(est_ko, target_t).item()
        
    delta_sisdr = base_sisdr - ko_sisdr
    
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
            "rms_corr": gamma_rms_corr,
            "rms_corr_source": rms_source,
            "onset_env_corr": gamma_onset_env_corr,
            "onset_env_corr_source": onset_env_source,
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
            "gamma_rms_corr": gamma_rms_corr,
            "gamma_rms_corr_source": rms_source,
            "gamma_onset_env_corr": gamma_onset_env_corr,
            "gamma_onset_env_corr_source": onset_env_source,
            "delta_sisdr": delta_sisdr,
        }
        debug_file_handle.write(json.dumps(dbg) + "\n")
        debug_file_handle.flush()

    return analysis

def print_film_summary(results, model_type):
    print(f"\n{'='*60}")
    print(f"FiLM MODULATION ANALYSIS SUMMARY - {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    gamma_rms_corrs = []
    delta_sisdr_list = []

    for artist_data in results.values():
        for song_data in artist_data.values():
            for analysis in song_data:
                if analysis:
                    if analysis["gamma_stats"].get("rms_corr") is not None:
                        gamma_rms_corrs.append(analysis["gamma_stats"]["rms_corr"])
                    if analysis.get("film_knockout_delta_sisdr") is not None:
                        delta_sisdr_list.append(analysis["film_knockout_delta_sisdr"])

    if gamma_rms_corrs:
        print(f"\nTemporal correlation |γ| vs audio RMS:")
        print(f"  Mean: {np.mean(gamma_rms_corrs):.4f} ± {np.std(gamma_rms_corrs):.4f}")
        print(f"  Range: [{np.min(gamma_rms_corrs):.4f}, {np.max(gamma_rms_corrs):.4f}]")

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

    # Output dirs (temporal_shuffle style)
    root_dir = os.path.join(args.output_dir, f"film_analysis_{args.model}")
    per_song_dir = os.path.join(root_dir, "per_song")
    os.makedirs(per_song_dir, exist_ok=True)
    print(f"Per-song outputs -> {per_song_dir}")

    debug_fh = None
    if args.debug_film:
        debug_fh = open(os.path.join(root_dir, f"film_debug_{args.model}.jsonl"), "w")

    # Accumulator similar to song_acc in temporal shuffle script
    # key: (artist, song) -> dict of metric lists
    song_acc = {}

    def _avg(lst):
        vals = [v for v in lst if v is not None]
        return float(np.mean(vals)) if vals else None

    def _std(lst):
        vals = [v for v in lst if v is not None]
        return float(np.std(vals)) if vals else None

    def build_song_entry(artist, song, acc):
        return {
            "artist": artist,
            "song": song,
            "scale_mean_mean": _avg(acc["scale_mean"]),
            "scale_mean_std": _std(acc["scale_mean"]),
            "bias_mean_mean": _avg(acc["bias_mean"]),
            "bias_mean_std": _std(acc["bias_mean"]),
            "gamma_abs_mean_mean": _avg(acc["gamma_abs_mean"]),
            "gamma_abs_mean_std": _std(acc["gamma_abs_mean"]),
            "total_modulation_mean": _avg(acc["total_modulation"]),
            "total_modulation_std": _std(acc["total_modulation"]),
            "rms_corr_mean": _avg(acc["rms_corr"]),
            "rms_corr_std": _std(acc["rms_corr"]),
            "onset_env_corr_mean": _avg(acc["onset_env_corr"]),
            "onset_env_corr_std": _std(acc["onset_env_corr"]),
            "delta_sisdr_mean": _avg(acc["delta_sisdr"]),
            "delta_sisdr_std": _std(acc["delta_sisdr"]),
            "boosted_count_mean": _avg(acc["boosted_count"]),
            "suppressed_count_mean": _avg(acc["suppressed_count"]),
        }

    for sample in tqdm(dataloader, desc="Samples"):
        artist = sample.get("artist", ["unknown"])[0]
        song = sample.get("song", ["unknown"])[0]
        key = (artist, song)
        safe_artist = _sanitize_name(artist)
        safe_song = _sanitize_name(song)
        song_entry_path = os.path.join(per_song_dir, f"{safe_artist}__{safe_song}.json")

        # Skip existing unless force
        if os.path.exists(song_entry_path) and not args.force:
            continue

        analysis = analyze_film_modulation(
            model,
            sample,
            DEVICE,
            debug=args.debug_film,
            meta={"artist": artist, "song": song},
            debug_file_handle=debug_fh,
        )
        if analysis is None:
            continue

        if key not in song_acc:
            song_acc[key] = {
                "scale_mean": [],
                "bias_mean": [],
                "gamma_abs_mean": [],
                "total_modulation": [],
                "rms_corr": [],
                "onset_env_corr": [],
                "delta_sisdr": [],
                "boosted_count": [],
                "suppressed_count": [],
            }
        acc = song_acc[key]
        acc["scale_mean"].append(analysis["scale_stats"]["mean"])  # per-chunk values
        acc["bias_mean"].append(analysis["bias_stats"]["mean"])  # per-chunk values
        acc["gamma_abs_mean"].append(analysis["gamma_stats"]["abs_mean"])
        acc["total_modulation"].append(analysis["modulation_intensity"]["total_modulation"])
        acc["rms_corr"].append(analysis["gamma_stats"].get("rms_corr"))
        acc["onset_env_corr"].append(analysis["gamma_stats"].get("onset_env_corr"))
        acc["delta_sisdr"].append(analysis.get("film_knockout_delta_sisdr"))
        acc["boosted_count"].append(len(analysis["modulation_effects"]["highly_boosted_features"]))
        acc["suppressed_count"].append(len(analysis["modulation_effects"]["suppressed_features"]))

        # Write / overwrite aggregated per-song JSON
        with open(song_entry_path, "w") as f_song:
            json.dump(build_song_entry(artist, song, acc), f_song, indent=2)

    if debug_fh is not None:
        debug_fh.close()

    print("Done. Per-song aggregated files written.")


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
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Root output directory"
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute songs even if per-song JSON exists"
    )

    args = parser.parse_args()
    main(args)
