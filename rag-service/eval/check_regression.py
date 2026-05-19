# CI regression gate: compares latest eval metrics against the stored baseline and
# exits with code 1 if any metric dropped more than the given threshold, blocking deploy.

from __future__ import annotations

import argparse
import json
import sys


def check(threshold: float) -> None:
    # Compares eval/latest_metrics.json against eval/baseline_metrics.json.
    # Exits with code 1 (blocks CI deploy) if any metric dropped more than threshold.
    with open("eval/latest_metrics.json") as f:
        latest = json.load(f)
    with open("eval/baseline_metrics.json") as f:
        baseline = json.load(f)

    failures = [
        f"{metric}: {baseline[metric]:.3f} → {latest[metric]:.3f}"
        for metric in baseline
        if (baseline[metric] - latest[metric]) / baseline[metric] > threshold
    ]

    if failures:
        print("Regression gate FAILED:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)

    print("Regression gate passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    check(args.threshold)
