#!/usr/bin/env python3
"""Regression gate: compare latest_metrics.json against the locked baseline.

    python eval/check_regression.py --threshold 0.05
    python eval/check_regression.py --lock          # write a new baseline

Adapted from rag-service/eval/check_regression.py, which compares a flat
{metric: value} map. This one walks arms x groups, and it has to know three
things that a flat comparison does not.

**Direction.** Gini is better when it falls. A gate that flags "dropped more
than the threshold" would report every improvement in exposure concentration as
a regression.

**Which metrics are quality at all.** Diversity, coverage and novelty trade
against accuracy: gating them demands that both improve together, which forbids
the trade-offs the beyond-accuracy metrics exist to make visible. They are
reported as movements, never as failures. Ceilings, cohort sizes and the ILD
companion figures are constraints and context, not quality, and are never
compared.

**Whether the two runs describe the same thing.** The evaluation cohort is
sampled, so a baseline taken over one cohort and a run taken over another are
not comparable in either direction -- that is INCONCLUSIVE, not a pass and not a
failure. Same for a run marked degraded, which by construction may carry
silently depressed numbers.

The metric classification is deliberately fixed here rather than derived from a
run. Choosing which metrics gate *after* seeing the numbers invites picking the
set that passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
LATEST = EVAL_DIR / "latest_metrics.json"
BASELINE = EVAL_DIR / "baseline_metrics.json"

# Single-directional quality. A material drop is a regression.
GATED_HIGHER_IS_BETTER = (
    "ndcg@10", "recall@10", "recall@20", "hit_rate@10", "mrr",
)

# Real signals, reported every run, never a pass/fail. Raising diversity or
# coverage usually costs accuracy; gating both directions at once would forbid
# the trade-off rather than surface it.
REPORTED_NOT_GATED = (
    "catalog_coverage", "gini", "novelty", "ild",
)

# Lower is better. Listed so a future move of any of these into the gated set
# cannot silently invert its meaning.
LOWER_IS_BETTER = frozenset({"gini"})

# Constraints and context. Comparing them says nothing about quality.
NEVER_COMPARED = (
    "recall@10_ceiling", "recall@20_ceiling", "n_users", "catalog_size",
    "mean_relevant", "ild_item_coverage", "ild_pairs_scored",
    "ild_labels_per_item",
)

# Below this, relative change is noise from a handful of users rather than a
# signal. Absolute movements smaller than this are reported but never failed.
ABS_FLOOR = 1e-4


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run eval/run_eval.py first")
    return json.loads(path.read_text())


def _groups(payload: dict) -> dict[tuple[str, str], dict]:
    return {(arm["arm"], group): metrics
            for arm in payload.get("arms", [])
            for group, metrics in arm.get("groups", {}).items()}


def _comparable(baseline: dict, latest: dict) -> list[str]:
    """Reasons the two runs cannot be compared at all."""
    blockers = []
    if latest.get("degraded"):
        blockers.append("latest run is marked degraded -- a precondition failed, "
                        "so its numbers may be silently depressed")
    if baseline.get("degraded"):
        blockers.append("baseline is marked degraded and should never have been locked")

    b_cohort, l_cohort = baseline.get("cohort", {}), latest.get("cohort", {})
    if b_cohort.get("evaluable_users") != l_cohort.get("evaluable_users"):
        blockers.append(
            f"cohort changed: baseline scored "
            f"{b_cohort.get('evaluable_users')} users, latest scored "
            f"{l_cohort.get('evaluable_users')}. Metrics over different "
            f"populations are not comparable in either direction")

    b_den, l_den = baseline.get("denominators", {}), latest.get("denominators", {})
    if b_den.get("catalogue_size") != l_den.get("catalogue_size"):
        blockers.append(
            f"catalogue changed: {b_den.get('catalogue_size')} -> "
            f"{l_den.get('catalogue_size')}. Coverage and novelty share this "
            f"denominator")
    return blockers


def _relative_drop(metric: str, base: float, new: float) -> float:
    """Fractional worsening, sign-corrected for direction. Negative = improved."""
    if metric in LOWER_IS_BETTER:
        base, new = -base, -new
    if base == 0:
        return 0.0 if new >= base else 1.0
    return (base - new) / abs(base)


def check(threshold: float, latest_path: Path, baseline_path: Path) -> int:
    latest, baseline = _load(latest_path), _load(baseline_path)

    blockers = _comparable(baseline, latest)
    if blockers:
        print("Regression gate INCONCLUSIVE:")
        for b in blockers:
            print(f"  {b}")
        return 1

    b_groups, l_groups = _groups(baseline), _groups(latest)
    shared = sorted(set(b_groups) & set(l_groups))
    if not shared:
        print("Regression gate INCONCLUSIVE: no arm/group pairs in common.")
        return 1

    failures: list[str] = []
    movements: list[str] = []
    compared = 0

    for key in shared:
        arm, group = key
        b, l = b_groups[key], l_groups[key]

        for metric in GATED_HIGHER_IS_BETTER:
            if metric not in b or metric not in l or l[metric] is None:
                continue
            compared += 1
            drop = _relative_drop(metric, b[metric], l[metric])
            if drop > threshold and abs(b[metric] - l[metric]) > ABS_FLOOR:
                failures.append(
                    f"{arm}/{group} {metric}: {b[metric]:.4f} -> {l[metric]:.4f} "
                    f"({drop:+.1%})")

        for metric in REPORTED_NOT_GATED:
            if metric not in b or metric not in l or l[metric] is None:
                continue
            delta = l[metric] - b[metric]
            if abs(delta) > ABS_FLOOR:
                arrow = "better" if _relative_drop(metric, b[metric], l[metric]) < 0 else "worse"
                movements.append(
                    f"{arm}/{group} {metric}: {b[metric]:.4f} -> {l[metric]:.4f} "
                    f"({delta:+.4f}, {arrow})")

    only_latest = sorted(set(l_groups) - set(b_groups))
    only_base = sorted(set(b_groups) - set(l_groups))
    for arm, group in only_base:
        print(f"  skipped {arm}/{group} -- absent from latest")
    for arm, group in only_latest:
        print(f"  new     {arm}/{group} -- not in the baseline, not gated")

    if movements:
        print(f"\nBeyond-accuracy movements ({len(movements)}), reported not gated:")
        for m in movements[:20]:
            print(f"  {m}")
        if len(movements) > 20:
            print(f"  ... {len(movements) - 20} more")

    if failures:
        print(f"\nRegression gate FAILED ({len(failures)} of {compared} comparisons):")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"\nRegression gate passed ({compared} comparisons across "
          f"{len(shared)} arm/group pairs, threshold {threshold:.0%}).")
    return 0


def lock(latest_path: Path, baseline_path: Path, force: bool) -> int:
    """Promote the latest run to the baseline.

    Refuses a degraded run. A baseline is the reference every future comparison
    is made against, so locking one whose preconditions failed poisons every
    later result and cannot be detected afterwards.
    """
    latest = _load(latest_path)
    if latest.get("degraded") and not force:
        print("Refusing to lock: the latest run is marked degraded.", file=sys.stderr)
        for p in latest.get("preconditions", {}).get("problems", []):
            print(f"  {p}", file=sys.stderr)
        print("Fix the precondition and re-run. --force overrides, and should "
              "not be used for a baseline anyone will compare against.",
              file=sys.stderr)
        return 2

    if baseline_path.exists():
        prev = json.loads(baseline_path.read_text())
        print(f"replacing baseline locked at {prev.get('locked_at', 'unknown')}")

    payload = dict(latest)
    payload["locked_at"] = latest.get("generated_at")
    baseline_path.write_text(json.dumps(payload, indent=2))
    c = latest.get("cohort", {})
    print(f"locked {baseline_path}")
    print(f"  cohort: {c.get('evaluable_users')} evaluable users "
          f"{c.get('by_cohort')}, warm strata {c.get('warm_strata')}")
    print(f"  arms:   {[a['arm'] for a in latest.get('arms', [])]}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.05,
                   help="maximum tolerated relative drop in a gated metric "
                        "(default 0.05)")
    p.add_argument("--latest", type=Path, default=LATEST)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--lock", action="store_true",
                   help="promote the latest run to the baseline")
    p.add_argument("--force", action="store_true",
                   help="with --lock: lock even a degraded run")
    args = p.parse_args()

    if args.lock:
        sys.exit(lock(args.latest, args.baseline, args.force))
    sys.exit(check(args.threshold, args.latest, args.baseline))


if __name__ == "__main__":
    main()
