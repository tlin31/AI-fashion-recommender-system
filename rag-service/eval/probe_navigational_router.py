"""Offline test: can BM25 score concentration route navigational queries past the generator?

Motivation: generation is 69% of request latency (~920 ms TTFT + 8.3 ms/token). Navigational
queries score NDCG@10 = 0.9359 — the ranked products already are the answer — so skipping
generation for them would remove ~1817 ms outright rather than hiding it behind streaming.
The service does not know a query's type at request time, so it needs a cheap proxy.

Hypothesis: a navigational query names one specific product, so BM25 scores should be
concentrated on a few documents; an exploratory query matches many products weakly.

Ground truth: the `type` field in eval/golden_queries.json (20 navigational / 80 other).
Costs nothing to run — BM25 over rag_products only, no OpenAI and no Milvus.

VERDICT: rejected. See README § "What didn't work".

Run:  python eval/test_navigational_router.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.retrieval import _build_bm25_sync, _tokenize  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_queries.json"
OUT = Path(__file__).parent / "navigational_router_results.json"

# Same candidate pool the sparse path in retrieval.py uses, so the features are
# computed over exactly the scores a live request would have available.
POOL = 50

# Measured on the 30-query latency run: mean generation time on the /query path.
_GENERATION_MS = 1817


def features(scores: np.ndarray) -> dict[str, float]:
    """Concentration features over the top-POOL BM25 scores for one query."""
    top = np.sort(scores)[::-1][:POOL]
    s1 = float(top[0])
    if s1 <= 0:
        return {k: 0.0 for k in
                ("max_score", "top1_ratio", "gap_ratio", "top1_over_next9", "entropy")}

    total = float(top.sum())
    p = top / total
    p = p[p > 0]
    next9 = float(top[1:10].mean()) if len(top) > 1 else 0.0

    return {
        "max_score":       s1,
        "top1_ratio":      s1 / total,
        "gap_ratio":       (s1 - float(top[1])) / s1 if len(top) > 1 else 1.0,
        "top1_over_next9": s1 / next9 if next9 > 0 else float("inf"),
        "entropy":         float(-(p * np.log(p)).sum()),
    }


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney U / |pos||neg|. Ties count 0.5. No sklearn dependency."""
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def boot_ci(pos: list[float], neg: list[float], rng: random.Random,
            n: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap CI on AUC — same resampling scheme as calibrate_crag.py.
    With only 20 positives a point estimate says very little on its own.
    """
    draws = [auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg])
             for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def best_threshold(pos: list[float], neg: list[float], min_precision: float):
    """Highest-recall threshold (score >= t predicts navigational) meeting min_precision.

    Precision is the metric that matters, not accuracy: a false positive silently
    drops the prose from a query that needed it, while a false negative only forfeits
    a latency saving. The two errors are not symmetric.
    """
    best = None
    for t in sorted(set(pos + neg)):
        tp = sum(1 for x in pos if x >= t)
        fp = sum(1 for x in neg if x >= t)
        if tp == 0:
            continue
        precision, recall = tp / (tp + fp), tp / len(pos)
        if precision >= min_precision and (best is None or recall > best["recall"]):
            best = {"threshold": t, "precision": precision, "recall": recall,
                    "tp": tp, "fp": fp}
    return best


def main() -> None:
    queries = json.loads(GOLDEN.read_text())
    bm25, product_ids = _build_bm25_sync()
    print(f"BM25 index: {len(product_ids)} products\n", file=sys.stderr)

    rows = []
    for q in queries:
        scores = np.asarray(bm25.get_scores(_tokenize(q["query"])))
        rows.append({"id": q["id"], "type": q["type"], "query": q["query"],
                     "n_tokens": len(q["query"].split()), **features(scores)})

    nav = [r for r in rows if r["type"] == "navigational"]
    other = [r for r in rows if r["type"] != "navigational"]
    rng = random.Random(42)

    # n_tokens is a negative control: BM25 sums over query terms, so a longer query
    # scores higher mechanically. If it separated the classes as well as the real
    # features, the signal would be an artefact of query length rather than intent.
    feat_names = ["max_score", "top1_ratio", "gap_ratio", "top1_over_next9",
                  "entropy", "n_tokens"]

    report = {"n_navigational": len(nav), "n_other": len(other), "features": {}}
    print(f"{'feature':<18} {'AUC':>6}  {'95% CI':>16}  {'nav med':>9}  {'other med':>9}")
    for f in feat_names:
        pos = [r[f] for r in nav if math.isfinite(r[f])]
        neg = [r[f] for r in other if math.isfinite(r[f])]
        # Lower entropy means more concentrated — negate so "higher = navigational"
        # holds for every feature and the AUCs stay comparable.
        if f == "entropy":
            pos, neg = [-x for x in pos], [-x for x in neg]

        a = auc(pos, neg)
        lo, hi = boot_ci(pos, neg, rng)
        report["features"][f] = {
            "auc": a, "ci_low": lo, "ci_high": hi,
            "nav_median": float(np.median(pos)),
            "other_median": float(np.median(neg)),
            "at_precision_0.90": best_threshold(pos, neg, 0.90),
            "at_precision_0.80": best_threshold(pos, neg, 0.80),
        }
        tag = "  (negative control)" if f == "n_tokens" else ""
        print(f"{f:<18} {a:>6.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"{np.median(pos):>9.3f}  {np.median(neg):>9.3f}{tag}")

    print("\noperating points — predict navigational when feature >= threshold:")
    for f in feat_names:
        for key in ("at_precision_0.90", "at_precision_0.80"):
            b = report["features"][f][key]
            label = f"{f:<18} P>={key[-4:]}"
            if b is None:
                print(f"  {label}  unreachable at any threshold")
            else:
                print(f"  {label}  t={b['threshold']:>8.3f}  precision={b['precision']:.2f}"
                      f"  recall={b['recall']:.2f}  (tp={b['tp']}, fp={b['fp']})")

    best = report["features"]["max_score"]["at_precision_0.80"]
    if best:
        skipped = best["tp"] + best["fp"]
        report["expected_value"] = {
            "queries_skipped_per_100": skipped,
            "wrong_skips_per_100": best["fp"],
            "mean_ms_saved_across_all_traffic": skipped / len(rows) * _GENERATION_MS,
        }
        print(f"\nbest deployable point (max_score >= {best['threshold']:.2f}): "
              f"skips generation on {skipped}/100 queries, {best['fp']} of them wrong; "
              f"mean saving across all traffic "
              f"{skipped / len(rows) * _GENERATION_MS:.0f} ms")

    print("\nnon-navigational queries scoring above that threshold:")
    for x in sorted(other, key=lambda r: -r["max_score"]):
        if best and x["max_score"] >= best["threshold"]:
            print(f"  {x['max_score']:7.1f}  {x['type']:<12} {x['query'][:70]}")

    report["rows"] = rows
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
