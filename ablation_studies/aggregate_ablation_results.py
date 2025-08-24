import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any

NUMERIC_EXCLUDE = {"model", "artist", "song"}


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def collect_sample_files(per_sample_dir: str) -> List[str]:
    files = []
    if not os.path.isdir(per_sample_dir):
        raise FileNotFoundError(f"Per-sample directory not found: {per_sample_dir}")
    for fn in os.listdir(per_sample_dir):
        if not fn.endswith('.json'):
            continue
        if fn.startswith('aggregated_'):
            # Skip previously aggregated summary files so they are not double-counted
            continue
        files.append(os.path.join(per_sample_dir, fn))
    if not files:
        raise RuntimeError(f"No JSON sample files found in {per_sample_dir}")
    return sorted(files)


def aggregate_values(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in records:
        for k, v in r.items():
            if k in NUMERIC_EXCLUDE:
                continue
            if is_number(v) and v is not None:
                sums[k] += v
                counts[k] += 1
    agg = {}
    for k, total in sums.items():
        agg[k + '_mean'] = total / counts[k] if counts[k] > 0 else None
    return agg


def aggregate_per_artist(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_artist: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_artist[s.get('artist', 'unknown')].append(s)
    result = {}
    for artist, recs in by_artist.items():
        result[artist] = aggregate_values(recs)
        result[artist]['n_samples'] = len(recs)
    return result


def load_samples(files: List[str]) -> List[Dict[str, Any]]:
    out = []
    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            out.append(data)
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")
    return out


def detect_study_type(samples: List[Dict[str, Any]]) -> str:
    keys = set().union(*[s.keys() for s in samples])
    # Support both per-chunk and per-song key conventions
    if 'shuffled_SI_SDR' in keys or 'shuffled_mean' in keys:
        return 'temporal_shuffle'
    if any(k in keys for k in ['mouth_mask_SI_SDR', 'nose_mask_SI_SDR', 'mouth_mean', 'nose_mean']):
        return 'region_masking'
    if any(k.startswith('SI_SDR_per_p') or k == 'proportions' for k in keys):
        return 'landmark_dropout'
    return 'unknown'


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = aggregate_values(samples)
    overall['n_samples'] = len(samples)
    return overall


def build_output_structure(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    study_type = detect_study_type(samples)
    # Determine level (per_song vs per_chunk) heuristically
    example_keys = set(samples[0].keys()) if samples else set()
    level = 'per_song' if any(k.endswith('_mean') for k in example_keys) or 'n_chunks' in example_keys else 'per_chunk'
    return {
        'study_type': study_type,
        'aggregation_level': level,
        'model': samples[0].get('model') if samples else None,
        'overall': summarize(samples),
        'per_artist': aggregate_per_artist(samples),
        'sample_keys': sorted(list(set().union(*[s.keys() for s in samples])))
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-sample ablation results by artist and overall.")
    parser.add_argument('--input-root', required=True, help='Root folder for a specific ablation (e.g. results/temporal_shuffle_face_body) OR the parent containing per_song/')
    parser.add_argument('--per-song-subdir', default='per_song', help='Subdirectory name housing per-sample JSON files')
    parser.add_argument('--output', default=None, help='Output JSON path (default: <input-root>/aggregate.json)')
    args = parser.parse_args()

    # Resolve per-sample directory
    if os.path.isdir(os.path.join(args.input_root, args.per_song_subdir)):
        per_song_dir = os.path.join(args.input_root, args.per_song_subdir)
    else:
        # Maybe input_root already is the per_song directory
        if os.path.basename(args.input_root) == args.per_song_subdir:
            per_song_dir = args.input_root
        else:
            raise FileNotFoundError(f"Could not locate per-sample directory under {args.input_root}")

    files = collect_sample_files(per_song_dir)
    samples = load_samples(files)
    if not samples:
        raise RuntimeError("No samples loaded.")

    output_struct = build_output_structure(samples)

    out_path = args.output or os.path.join(os.path.dirname(per_song_dir), 'aggregate.json')
    with open(out_path, 'w') as f:
        json.dump(output_struct, f, indent=2)
    print(f"Wrote aggregate to {out_path}")
    print(f"Study: {output_struct['study_type']} | Model: {output_struct['model']} | Samples: {output_struct['overall']['n_samples']}")

if __name__ == '__main__':
    main()
