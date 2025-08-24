import argparse
import json
import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict

NUMERIC_EXCLUDE = {"artist", "song"}
CHUNK_COUNT_KEY = "n_chunks"


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_per_song_records(root_dir: str) -> List[Dict[str, Any]]:
    per_song_dir = os.path.join(root_dir, "per_song")
    if not os.path.isdir(per_song_dir):
        raise FileNotFoundError(f"per_song directory not found: {per_song_dir}")
    records: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(per_song_dir)):
        if not fn.endswith('.json'):
            continue
        fp = os.path.join(per_song_dir, fn)
        try:
            with open(fp, 'r') as f:
                rec = json.load(f)
            # Basic validation
            if 'artist' in rec and 'song' in rec:
                records.append(rec)
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}")
    if not records:
        raise RuntimeError(f"No valid per-song JSON files parsed in {per_song_dir}")
    return records


def collect_numeric_keys(records: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in records:
        for k, v in r.items():
            if k in NUMERIC_EXCLUDE:
                continue
            if is_number(v):
                keys.add(k)
    return sorted(keys)


def aggregate_overall(records: List[Dict[str, Any]], numeric_keys: List[str]) -> Dict[str, Any]:
    overall: Dict[str, Dict[str, float]] = {}
    # Pre-compute weights (n_chunks) for weighted means of *_mean metrics
    weights = [max(1, int(r.get(CHUNK_COUNT_KEY, 1))) for r in records]
    for k in numeric_keys:
        vals = [r[k] for r in records if is_number(r.get(k))]
        if not vals:
            continue
        overall[k] = {
            'mean': float(sum(vals) / len(vals)),
            'std': float((sum((v - (sum(vals)/len(vals)))**2 for v in vals) / (len(vals)-1))**0.5) if len(vals) > 1 else 0.0,
            'min': float(min(vals)),
            'max': float(max(vals)),
            'n': len(vals),
        }
        # Weighted mean (if key looks like an averaged metric)
        if k.endswith('_mean'):
            num = 0.0
            denom = 0.0
            for r, w in zip(records, weights):
                v = r.get(k)
                if is_number(v):
                    num += v * w
                    denom += w
            if denom > 0:
                overall[k]['weighted_mean_chunks'] = float(num / denom)
    return overall


def aggregate_per_artist(records: List[Dict[str, Any]], numeric_keys: List[str]) -> Dict[str, Any]:
    by_artist: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_artist[r.get('artist', 'unknown')].append(r)
    out: Dict[str, Any] = {}
    for artist, recs in by_artist.items():
        weights = [max(1, int(r.get(CHUNK_COUNT_KEY, 1))) for r in recs]
        artist_entry: Dict[str, Any] = {'n_songs': len(recs)}
        for k in numeric_keys:
            vals = [r[k] for r in recs if is_number(r.get(k))]
            if not vals:
                continue
            artist_entry[k + '_mean'] = float(sum(vals) / len(vals))
            if len(vals) > 1:
                m = artist_entry[k + '_mean']
                artist_entry[k + '_std'] = float((sum((v - m)**2 for v in vals) / (len(vals)-1))**0.5)
            else:
                artist_entry[k + '_std'] = 0.0
            if k.endswith('_mean'):
                num = 0.0
                denom = 0.0
                for r, w in zip(recs, weights):
                    v = r.get(k)
                    if is_number(v):
                        num += v * w
                        denom += w
                if denom > 0:
                    artist_entry[k + '_weighted_mean_chunks'] = float(num / denom)
        out[artist] = artist_entry
    return out


def build_summary(model: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = collect_numeric_keys(records)
    overall = aggregate_overall(records, numeric_keys)
    per_artist = aggregate_per_artist(records, numeric_keys)
    # Compute percentage of songs with positive delta_sisdr_mean
    delta_vals = [r.get('delta_sisdr_mean') for r in records if isinstance(r.get('delta_sisdr_mean'), (int, float))]
    positive_delta = sum(1 for v in delta_vals if v is not None and v > 0)
    pct_positive_delta = (positive_delta / len(delta_vals) * 100.0) if delta_vals else None
    return {
        'model': model,
        'n_songs': len(records),
        'numeric_keys': numeric_keys,
        'overall': overall,
        'per_artist': per_artist,
        'delta_sisdr_mean_positive_percentage': pct_positive_delta,
        'delta_sisdr_mean_positive_count': positive_delta,
        'delta_sisdr_mean_count': len(delta_vals),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-song FiLM analysis results into summary.json")
    parser.add_argument('--model', required=True, choices=['vocal', 'violin'], help='Model name used in film_analysis_<model> directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Root output directory (default: results)')
    parser.add_argument('--outfile', type=str, default=None, help='Optional explicit output file path')
    args = parser.parse_args()

    root_dir = os.path.join(args.output_dir, f'film_analysis_{args.model}')
    records = load_per_song_records(root_dir)
    summary = build_summary(args.model, records)

    out_path = args.outfile or os.path.join(root_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote summary to {out_path} (songs={summary["n_songs"]})')


if __name__ == '__main__':
    main()
