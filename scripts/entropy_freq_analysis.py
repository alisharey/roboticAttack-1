#!/usr/bin/env python3
"""
Analyze entropy oscillations using frequency-domain features and plot histograms.

Example:
  python scripts/entropy_freq_analysis.py \
    --input_dir /path/to/rollouts/origin/libero_10/2026_01_05-14_42_16 \
    --output /path/to/entropy_freq_summary.png \
    --save_csv /path/to/entropy_freq_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_entropy(path: Path, skip: int) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    dims = [row["dims"] for row in data]
    arr = np.asarray(dims, dtype=np.float64)
    if skip > 0:
        arr = arr[skip:]
    return arr


def _fft_features(series: np.ndarray, high_freq_frac: float) -> dict:
    series = series.astype(np.float64)
    series = series - series.mean()
    n = series.shape[0]
    if n < 4 or np.allclose(series, 0):
        return {
            "high_freq_ratio": 0.0,
            "spectral_centroid": 0.0,
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "signal_std": float(series.std()),
        }

    fft = np.fft.rfft(series)
    power = (np.abs(fft) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power[0] = 0.0  # remove DC
    total_power = power.sum()
    if total_power <= 0:
        return {
            "high_freq_ratio": 0.0,
            "spectral_centroid": 0.0,
            "dominant_freq": 0.0,
            "spectral_entropy": 0.0,
            "signal_std": float(series.std()),
        }

    nyquist = 0.5
    high_freq_thresh = nyquist * high_freq_frac
    high_power = power[freqs >= high_freq_thresh].sum()
    high_freq_ratio = float(high_power / total_power)

    spectral_centroid = float((freqs * power).sum() / total_power)
    dominant_freq = float(freqs[np.argmax(power)])

    p = power / total_power
    p = np.clip(p, 1e-12, 1.0)
    spectral_entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))

    return {
        "high_freq_ratio": high_freq_ratio,
        "spectral_centroid": spectral_centroid,
        "dominant_freq": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "signal_std": float(series.std()),
    }


def _aggregate_metrics(metrics_list, mode: str) -> dict:
    keys = metrics_list[0].keys()
    if mode == "dim_mean":
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    if mode == "dim_max":
        return {k: float(np.max([m[k] for m in metrics_list])) for k in keys}
    raise ValueError(f"Unknown mode: {mode}")


def compute_features(entropy: np.ndarray, mode: str, high_freq_frac: float) -> dict:
    if entropy.ndim != 2:
        raise ValueError(f"Expected entropy to be (T, K), got {entropy.shape}")
    if mode == "avg":
        series = entropy.mean(axis=1)
        return _fft_features(series, high_freq_frac)
    metrics_list = [
        _fft_features(entropy[:, i], high_freq_frac) for i in range(entropy.shape[1])
    ]
    return _aggregate_metrics(metrics_list, mode)


def _label_from_name(name: str):
    if "success=True" in name:
        return "success"
    if "success=False" in name:
        return "failure"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip_initial", type=int, default=0)
    parser.add_argument("--high_freq_frac", type=float, default=0.3)
    parser.add_argument(
        "--mode",
        type=str,
        default="per_dim",
        choices=["per_dim", "avg", "dim_mean", "dim_max"],
    )
    parser.add_argument("--save_csv", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*_entropy.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_entropy.json files under {input_dir}")

    rows = []
    for path in json_files:
        label = _label_from_name(path.name)
        if label is None:
            continue
        entropy = _load_entropy(path, args.skip_initial)
        if entropy.shape[0] < 4:
            continue
        if args.mode == "per_dim":
            for dim in range(entropy.shape[1]):
                features = _fft_features(entropy[:, dim], args.high_freq_frac)
                features["label"] = label
                features["path"] = str(path)
                features["dim"] = dim
                rows.append(features)
        else:
            features = compute_features(entropy, args.mode, args.high_freq_frac)
            features["label"] = label
            features["path"] = str(path)
            rows.append(features)

    if not rows:
        raise RuntimeError("No labeled episodes found.")

    if args.save_csv:
        fieldnames = list(rows[0].keys())
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    metrics = ["high_freq_ratio", "spectral_centroid", "dominant_freq", "spectral_entropy", "signal_std"]
    success = [r for r in rows if r["label"] == "success"]
    failure = [r for r in rows if r["label"] == "failure"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.hist([r[metric] for r in success], bins=20, alpha=0.6, label="success")
        ax.hist([r[metric] for r in failure], bins=20, alpha=0.6, label="failure")
        ax.set_title(metric)
        ax.legend(fontsize="small")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Entropy Frequency Features (mode={args.mode}, skip={args.skip_initial})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = args.output or str(input_dir / "entropy_freq_summary.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote plot to {out_path}")
    if args.save_csv:
        print(f"Wrote CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
