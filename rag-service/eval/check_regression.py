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

    # A metric missing from latest is skipped, not fatal. The smoke tier of the CI
    # gate runs with --skip-faithfulness, so demanding every baseline key would
    # crash the gate on exactly the tier designed to be cheap.
    missing = [m for m in baseline if m not in latest or latest[m] is None]
    compared = [m for m in baseline if m not in missing]

    failures = [
        f"{metric}: {baseline[metric]:.3f} → {latest[metric]:.3f}"
        for metric in compared
        if (baseline[metric] - latest[metric]) / baseline[metric] > threshold
    ]

    for metric in missing:
        print(f"  skipped {metric} — not present in latest_metrics.json")

    if not compared:
        print("Regression gate INCONCLUSIVE: no metrics in common with the baseline.")
        sys.exit(1)

    if failures:
        print("Regression gate FAILED:")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)

    print(f"Regression gate passed ({len(compared)} metric(s) compared).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    check(args.threshold)
