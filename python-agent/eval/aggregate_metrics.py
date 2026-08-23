"""Aggregate the per-turn JSONL written by agent/metrics.py.

Usage:
    python eval/aggregate_metrics.py                          # metrics/turns.jsonl
    python eval/aggregate_metrics.py --input metrics/arm_guard_on.jsonl
    python eval/aggregate_metrics.py --json                   # machine-readable
    python eval/aggregate_metrics.py --turn-type chat         # exclude resume turns

Reports, in order of how much they can be argued with:

  1. Latency percentiles, end-to-end and per node.        Measured. Hard fact.
  2. Token split by node and by model.                    Measured. Hard fact.
  3. Cost, and the single-model counterfactual.           Measured tokens x an
                                                          ASSUMED price table.

The token split is the honest headline for the two-model architecture. The
dollar saving depends entirely on what you put in pricing.json — with the
finalizer priced at 0 (free tier) the "saving" is tautological, so the script
prints the token share next to it and refuses to print a cost when any model in
the run is unpriced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.metrics import Pricing  # noqa: E402

DEFAULT_INPUT = _REPO_ROOT / "metrics" / "turns.jsonl"


# ---------------------------------------------------------------------------
# Stats helpers (no numpy — python-agent has no scientific stack)
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method.

    q is a fraction in [0, 1]. Returns 0.0 for an empty input so callers can
    print a table without special-casing every cell.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_turns(path: Path, turn_type: Optional[str] = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"No metrics file at {path}.\n"
            "Run the agent with AGENT_METRICS_PATH set (main.py defaults to "
            "metrics/turns.jsonl) and make at least one request first."
        )
    turns: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # A truncated final line is normal if the process was killed mid-write.
            print(f"warning: skipping malformed line {lineno}", file=sys.stderr)
            continue
        if turn_type and rec.get("turn_type") != turn_type:
            continue
        turns.append(rec)
    return turns


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(turns: list[dict[str, Any]], pricing: Pricing) -> dict[str, Any]:
    if not turns:
        raise SystemExit("No turns matched the filter — nothing to aggregate.")

    e2e = [t["total_latency_ms"] for t in turns]

    # ---- per-node latency ----
    node_lat: dict[str, list[float]] = defaultdict(list)
    node_runs: dict[str, int] = defaultdict(int)
    for t in turns:
        for n in t.get("nodes") or []:
            node_lat[n["node"]].append(n["latency_ms"])
            node_runs[n["node"]] += 1

    total_node_ms = sum(sum(v) for v in node_lat.values())
    nodes_summary = {
        name: {
            "runs": node_runs[name],
            "runs_per_turn": round(node_runs[name] / len(turns), 2),
            "p50_ms": round(percentile(vals, 0.50), 1),
            "p95_ms": round(percentile(vals, 0.95), 1),
            "mean_ms": round(mean(vals), 1),
            "total_ms": round(sum(vals), 1),
            "share_of_node_time": round(sum(vals) / total_node_ms, 4) if total_node_ms else 0.0,
        }
        for name, vals in sorted(node_lat.items(), key=lambda kv: -sum(kv[1]))
    }

    # ---- tokens by node and by model ----
    tok_by_node: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
    tok_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
    llm_calls = 0
    llm_calls_with_usage = 0
    for t in turns:
        for n in t.get("nodes") or []:
            if not n.get("model"):
                continue
            llm_calls += 1
            if not n.get("usage_available"):
                continue
            llm_calls_with_usage += 1
            for bucket, key in ((tok_by_node, n["node"]), (tok_by_model, n["model"])):
                bucket[key]["input"] += n["input_tokens"]
                bucket[key]["output"] += n["output_tokens"]
                bucket[key]["total"] += n["total_tokens"]

    grand_total = sum(v["total"] for v in tok_by_node.values())
    for bucket in (tok_by_node, tok_by_model):
        for v in bucket.values():
            v["share"] = round(v["total"] / grand_total, 4) if grand_total else 0.0

    # ---- cost ----
    # A single unpriced model makes every aggregate cost meaningless, so the
    # whole cost block collapses to None rather than under-reporting.
    unpriced = [m for m in tok_by_model if pricing.cost(m, 1, 1) is None]
    if unpriced:
        cost_block: dict[str, Any] = {"available": False, "unpriced_models": sorted(unpriced)}
    else:
        per_model = {
            m: pricing.cost(m, v["input"], v["output"]) or 0.0
            for m, v in tok_by_model.items()
        }
        actual = sum(per_model.values())
        cf_model = pricing.counterfactual_model
        cf_cost = None
        if cf_model and pricing.cost(cf_model, 1, 1) is not None:
            cf_cost = pricing.cost(
                cf_model,
                sum(v["input"] for v in tok_by_model.values()),
                sum(v["output"] for v in tok_by_model.values()),
            )
        cost_block = {
            "available": True,
            "pricing_as_of": pricing.as_of,
            "total_usd": actual,
            "per_turn_usd": actual / len(turns),
            "by_model_usd": per_model,
            "counterfactual_model": cf_model,
            "counterfactual_total_usd": cf_cost,
            "saving_vs_counterfactual": (
                (cf_cost - actual) / cf_cost if cf_cost else None
            ),
        }

    paths: dict[str, int] = defaultdict(int)
    for t in turns:
        paths[t.get("path") or "-"] += 1

    return {
        "turns": len(turns),
        "latency": {
            "p50_ms": round(percentile(e2e, 0.50), 1),
            "p95_ms": round(percentile(e2e, 0.95), 1),
            "p99_ms": round(percentile(e2e, 0.99), 1),
            "mean_ms": round(mean(e2e), 1),
            "max_ms": round(max(e2e), 1),
            # Turn time not attributed to any node = LangGraph + checkpointer
            # overhead. A large number here means the framework, not the models.
            "unattributed_ms_total": round(sum(e2e) - total_node_ms, 1),
            "unattributed_share": round(1 - total_node_ms / sum(e2e), 4) if sum(e2e) else 0.0,
        },
        "nodes": nodes_summary,
        "tokens": {
            "grand_total": grand_total,
            "per_turn": round(grand_total / len(turns), 1),
            "by_node": dict(tok_by_node),
            "by_model": dict(tok_by_model),
            "llm_calls": llm_calls,
            "llm_calls_with_usage": llm_calls_with_usage,
            "usage_coverage": round(llm_calls_with_usage / llm_calls, 4) if llm_calls else 0.0,
        },
        "cost": cost_block,
        "paths": dict(paths),
        "iterations_per_turn": round(mean([t.get("iterations", 0) for t in turns]), 2),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(agg: dict[str, Any]) -> None:
    lat = agg["latency"]
    print(f"\n=== turns: {agg['turns']}  |  mean iterations/turn: {agg['iterations_per_turn']} ===\n")

    print("-- end-to-end latency --")
    print(f"  p50 {lat['p50_ms']:>9.1f} ms")
    print(f"  p95 {lat['p95_ms']:>9.1f} ms")
    print(f"  p99 {lat['p99_ms']:>9.1f} ms")
    print(f"  max {lat['max_ms']:>9.1f} ms   mean {lat['mean_ms']:.1f} ms")
    print(f"  unattributed (framework overhead): {lat['unattributed_share'] * 100:.1f}%")

    print("\n-- per-node latency --")
    print(f"  {'node':<14}{'runs/turn':>10}{'p50 ms':>10}{'p95 ms':>10}{'share':>9}")
    for name, s in agg["nodes"].items():
        print(
            f"  {name:<14}{s['runs_per_turn']:>10.2f}{s['p50_ms']:>10.1f}"
            f"{s['p95_ms']:>10.1f}{s['share_of_node_time'] * 100:>8.1f}%"
        )

    tok = agg["tokens"]
    print(f"\n-- tokens ({tok['grand_total']:,} total, {tok['per_turn']:.0f}/turn) --")
    if tok["usage_coverage"] < 1.0:
        print(
            f"  WARNING: usage reported for only {tok['llm_calls_with_usage']}/"
            f"{tok['llm_calls']} LLM calls ({tok['usage_coverage'] * 100:.0f}%). "
            "Totals below understate real usage."
        )
    print(f"  {'by node':<16}{'input':>10}{'output':>10}{'total':>10}{'share':>9}")
    for name, v in sorted(tok["by_node"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {name:<16}{v['input']:>10,}{v['output']:>10,}{v['total']:>10,}{v['share'] * 100:>8.1f}%")
    print(f"  {'by model':<16}{'input':>10}{'output':>10}{'total':>10}{'share':>9}")
    for name, v in sorted(tok["by_model"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {name:<16}{v['input']:>10,}{v['output']:>10,}{v['total']:>10,}{v['share'] * 100:>8.1f}%")

    cost = agg["cost"]
    print("\n-- cost --")
    if not cost["available"]:
        print(f"  unavailable: no price for {', '.join(cost['unpriced_models'])}")
        print("  add them to agent/pricing.json to enable cost reporting.")
    else:
        print(f"  price table as of {cost['pricing_as_of']} (operator-supplied assumption)")
        print(f"  total     ${cost['total_usd']:.6f}")
        print(f"  per turn  ${cost['per_turn_usd']:.6f}")
        for m, c in sorted(cost["by_model_usd"].items(), key=lambda kv: -kv[1]):
            print(f"    {m:<24} ${c:.6f}")
        if cost["counterfactual_total_usd"] is not None:
            print(
                f"  single-model counterfactual ({cost['counterfactual_model']}): "
                f"${cost['counterfactual_total_usd']:.6f}"
            )
            print(f"  saving vs counterfactual: {cost['saving_vs_counterfactual'] * 100:.1f}%")
            print(
                "  NOTE: this saving is only as real as pricing.json. Quote the "
                "token share above as the primary claim."
            )

    print(f"\n-- answer path --")
    for p, n in sorted(agg["paths"].items(), key=lambda kv: -kv[1]):
        print(f"  {p:<14}{n:>6}  ({n / agg['turns'] * 100:.1f}%)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL written by MetricsSink")
    ap.add_argument("--turn-type", choices=["chat", "resume"], help="only aggregate this turn type")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    turns = load_turns(args.input, args.turn_type)
    agg = aggregate(turns, Pricing.load())

    if args.json:
        print(json.dumps(agg, indent=2))
    else:
        render(agg)


if __name__ == "__main__":
    main()
