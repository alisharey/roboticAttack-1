#!/usr/bin/env python3
"""
Compare detection rates for current per-dim threshold logic vs EMA-based logic.

Example:
  python scripts/entropy_detector_compare.py \
    --input_dir /path/to/rollouts/attack/libero_10/2026_01_05-19_29_17 \
    --threshold 0.75 \
    --window 17
"""

import argparse
import json
from pathlib import Path


def _label_from_name(name: str):
    if "success=True" in name:
        return "success"
    if "success=False" in name:
        return "failure"
    return None


def _load_entropy(path: Path):
    with path.open() as f:
        data = json.load(f)
    return [row["dims"] for row in data]


def _detect_current(entropy, threshold, window, gap_allow):
    if not entropy:
        return False
    num_dims = len(entropy[0])
    det_counts = [0] * num_dims
    det_gaps = [0] * num_dims
    for dims in entropy:
        high_mask = [d > threshold for d in dims]
        for i, is_high in enumerate(high_mask):
            if is_high:
                det_counts[i] += 1
                det_gaps[i] = 0
            else:
                if det_gaps[i] < gap_allow:
                    det_gaps[i] += 1
                else:
                    det_counts[i] = 0
                    det_gaps[i] = 0
        if any(c >= window for c in det_counts):
            return True
    return False


def _detect_ema(entropy, threshold, window, gap_allow, alpha):
    if not entropy:
        return False
    num_dims = len(entropy[0])
    ema = list(entropy[0])
    det_counts = [0] * num_dims
    det_gaps = [0] * num_dims
    for dims in entropy:
        high_mask = []
        for i, val in enumerate(dims):
            ema[i] = alpha * val + (1.0 - alpha) * ema[i]
            high_mask.append(ema[i] > threshold)
        for i, is_high in enumerate(high_mask):
            if is_high:
                det_counts[i] += 1
                det_gaps[i] = 0
            else:
                if det_gaps[i] < gap_allow:
                    det_gaps[i] += 1
                else:
                    det_counts[i] = 0
                    det_gaps[i] = 0
        if any(c >= window for c in det_counts):
            return True
    return False


def _rate(detected, total):
    return detected / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--gap_allow", type=int, default=1)
    parser.add_argument("--ema_alpha", type=float, default=None)
    args = parser.parse_args()

    alpha = args.ema_alpha
    if alpha is None:
        alpha = 2.0 / (args.window + 1)

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.rglob("*_entropy.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_entropy.json files under {input_dir}")

    stats = {
        "current": {"success": 0, "failure": 0, "succ_detect": 0, "fail_detect": 0},
        "ema": {"success": 0, "failure": 0, "succ_detect": 0, "fail_detect": 0},
    }

    for path in json_files:
        label = _label_from_name(path.name)
        if label is None:
            continue
        entropy = _load_entropy(path)
        detected_current = _detect_current(entropy, args.threshold, args.window, args.gap_allow)
        detected_ema = _detect_ema(entropy, args.threshold, args.window, args.gap_allow, alpha)

        stats["current"][label] += 1
        stats["ema"][label] += 1
        if label == "success":
            if detected_current:
                stats["current"]["succ_detect"] += 1
            if detected_ema:
                stats["ema"]["succ_detect"] += 1
        else:
            if detected_current:
                stats["current"]["fail_detect"] += 1
            if detected_ema:
                stats["ema"]["fail_detect"] += 1

    for key in ("current", "ema"):
        succ = stats[key]["success"]
        fail = stats[key]["failure"]
        fp = stats[key]["succ_detect"]
        tp = stats[key]["fail_detect"]
        tpr = _rate(tp, fail)
        fpr = _rate(fp, succ)
        print(
            f"{key}: thr={args.threshold} window={args.window} gap={args.gap_allow} alpha={alpha:.4f} "
            f"TPR={tpr:.3f} FPR={fpr:.3f} "
            f"(fail_detect={tp}/{fail}, succ_detect={fp}/{succ})"
        )


if __name__ == "__main__":
    main()
