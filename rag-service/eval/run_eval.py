# Full eval harness runner. Loads golden_queries.json, queries the live /query endpoint
# for each entry, and computes Recall@10, NDCG@10, and Ragas faithfulness. Writes
# results to eval/latest_metrics.json for the regression gate to compare against baseline.
#
# Usage:
#   python eval/run_eval.py --golden-set eval/golden_queries.json
#   python eval/run_eval.py --golden-set eval/golden_queries.json --endpoint http://localhost:8002
#   python eval/run_eval.py --golden-set eval/golden_queries.json --skip-faithfulness
#
# Exit criteria (checked automatically after each run):
#   NDCG@10     >= 0.50
#   Recall@10   >= 0.70
#   Faithfulness >= 0.85

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import httpx

# Support both invocation styles:
#   python eval/run_eval.py ...   (adds eval/ to sys.path — package prefix breaks)
#   python -m eval.run_eval ...   (adds rag-service/ to sys.path — package prefix works)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from eval.ragas_judge import faithfulness_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Exit criteria thresholds
_NDCG_TARGET = 0.50
_RECALL_TARGET = 0.70
_FAITH_TARGET = 0.85

# How many products to request from /query (must cover the full NDCG/recall window)
_TOP_K = 10

# Per-query HTTP timeout; the reranker is the slowest stage (~800 ms on CPU)
_REQUEST_TIMEOUT_S = 15.0


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _query_endpoint(
    client: httpx.AsyncClient,
    query: str,
    endpoint: str,
    query_id: str,
) -> dict | None:
    """POST /query and return the parsed JSON response, or None on failure."""
    try:
        resp = await client.post(
            f"{endpoint}/query",
            json={"query": query, "top_k": _TOP_K},
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("%s: HTTP %s — %s", query_id, exc.response.status_code, exc)
    except httpx.RequestError as exc:
        logger.warning("%s: request error — %s", query_id, exc)
    return None


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

async def _eval_one(
    entry: dict,
    response: dict,
    skip_faithfulness: bool,
) -> dict:
    """Compute all metrics for a single (golden_query, pipeline_response) pair."""
    relevance: dict[str, int] = entry.get("relevance", {})
    relevant_ids: set[str] = {pid for pid, score in relevance.items() if score > 0}

    retrieved_ids: list[str] = [
        p["product_id"] for p in response.get("products", [])
    ]

    recall = recall_at_k(retrieved_ids, relevant_ids, k=_TOP_K)
    ndcg = ndcg_at_k(retrieved_ids, relevance, k=_TOP_K)

    if skip_faithfulness:
        faith = None
    else:
        context_chunks = [p.get("chunk_text", "") for p in response.get("products", [])]
        faith = await faithfulness_score(
            entry["query"], response.get("answer", ""), context_chunks
        )

    return {
        "id": entry["id"],
        "query": entry["query"],
        "type": entry.get("type", "unknown"),
        "dense_only": entry.get("dense_only", False),
        "retrieval_path": response.get("retrieval_path", "unknown"),
        "recall_at_10": round(recall, 4),
        "ndcg_at_10": round(ndcg, 4),
        "faithfulness": round(faith, 4) if faith is not None else None,
        "retrieved_ids": retrieved_ids,
        "labeled_relevant": list(relevant_ids),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run(
    golden_set_path: str,
    endpoint: str = "http://localhost:8002",
    skip_faithfulness: bool = False,
    concurrency: int = 4,
) -> dict:
    """Evaluate the live /query endpoint against the golden query set.

    Args:
        golden_set_path:   Path to golden_queries.json.
        endpoint:          Base URL of the running RAG service.
        skip_faithfulness: Skip Ragas faithfulness scoring (faster, no OpenAI cost).
        concurrency:       Max simultaneous /query requests.

    Returns:
        Full results dict (also written to eval/results_YYYYMMDD.json and
        eval/latest_metrics.json).
    """
    golden_path = Path(golden_set_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    with open(golden_path) as f:
        queries: list[dict] = json.load(f)

    logger.info("Loaded %d golden queries from %s", len(queries), golden_path)
    logger.info("Endpoint: %s  |  top_k=%d  |  skip_faithfulness=%s",
                endpoint, _TOP_K, skip_faithfulness)

    # Fetch all responses, bounded by a semaphore to avoid overloading the service
    semaphore = asyncio.Semaphore(concurrency)
    responses: dict[str, dict | None] = {}

    async def fetch_with_limit(entry: dict) -> None:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                resp = await _query_endpoint(client, entry["query"], endpoint, entry["id"])
            responses[entry["id"]] = resp

    logger.info("Querying %d entries (concurrency=%d) …", len(queries), concurrency)
    await asyncio.gather(*(fetch_with_limit(q) for q in queries))

    # Evaluate each response
    per_query_results: list[dict] = []
    skipped = 0
    for entry in queries:
        resp = responses.get(entry["id"])
        if resp is None:
            logger.warning("Skipping %s — no response from endpoint", entry["id"])
            skipped += 1
            continue
        result = await _eval_one(entry, resp, skip_faithfulness)
        per_query_results.append(result)

    if not per_query_results:
        raise RuntimeError("No queries completed successfully; is the service running?")

    # ---------------------------------------------------------------------------
    # Aggregate: overall
    # ---------------------------------------------------------------------------
    def _avg(key: str) -> float:
        vals = [r[key] for r in per_query_results if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    overall = {
        "recall_at_10": _avg("recall_at_10"),
        "ndcg_at_10": _avg("ndcg_at_10"),
        "faithfulness": _avg("faithfulness") if not skip_faithfulness else None,
        "n_evaluated": len(per_query_results),
        "n_skipped": skipped,
    }

    # ---------------------------------------------------------------------------
    # Aggregate: per query type (navigational / exploratory / attribute / edge)
    # ---------------------------------------------------------------------------
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in per_query_results:
        by_type[r["type"]].append(r)

    type_breakdown = {
        qtype: {
            "recall_at_10": round(
                sum(r["recall_at_10"] for r in items) / len(items), 4
            ),
            "ndcg_at_10": round(
                sum(r["ndcg_at_10"] for r in items) / len(items), 4
            ),
            "faithfulness": (
                round(
                    sum(r["faithfulness"] for r in items if r["faithfulness"] is not None)
                    / max(1, sum(1 for r in items if r["faithfulness"] is not None)),
                    4,
                )
                if not skip_faithfulness
                else None
            ),
            "count": len(items),
        }
        for qtype, items in sorted(by_type.items())
    }

    # ---------------------------------------------------------------------------
    # Dense-only subset — sanity check on dense retrieval quality
    # ---------------------------------------------------------------------------
    dense_only = [r for r in per_query_results if r.get("dense_only")]
    dense_summary: dict = {}
    if dense_only:
        dense_summary = {
            "recall_at_10": round(
                sum(r["recall_at_10"] for r in dense_only) / len(dense_only), 4
            ),
            "ndcg_at_10": round(
                sum(r["ndcg_at_10"] for r in dense_only) / len(dense_only), 4
            ),
            "count": len(dense_only),
        }

    # ---------------------------------------------------------------------------
    # Retrieval path distribution
    # ---------------------------------------------------------------------------
    path_counts: dict[str, int] = defaultdict(int)
    for r in per_query_results:
        path_counts[r["retrieval_path"]] += 1

    # ---------------------------------------------------------------------------
    # Assemble full result
    # ---------------------------------------------------------------------------
    run_at = datetime.datetime.utcnow().isoformat() + "Z"
    full_result = {
        "run_at": run_at,
        "endpoint": endpoint,
        "golden_set": str(golden_path),
        "overall": overall,
        "by_type": type_breakdown,
        "dense_only_subset": dense_summary,
        "retrieval_path_distribution": dict(path_counts),
        "per_query": per_query_results,
    }

    # ---------------------------------------------------------------------------
    # Write output files
    # ---------------------------------------------------------------------------
    eval_dir = Path("eval")
    date_str = datetime.date.today().strftime("%Y%m%d")
    results_path = eval_dir / f"results_{date_str}.json"
    latest_path = eval_dir / "latest_metrics.json"

    with open(results_path, "w") as f:
        json.dump(full_result, f, indent=2)

    # latest_metrics.json contains only the scalar metrics the regression gate reads
    latest_metrics = {
        "recall_at_10": overall["recall_at_10"],
        "ndcg_at_10": overall["ndcg_at_10"],
    }
    if overall["faithfulness"] is not None:
        latest_metrics["faithfulness"] = overall["faithfulness"]

    with open(latest_path, "w") as f:
        json.dump(latest_metrics, f, indent=2)

    # ---------------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Eval complete  ({len(per_query_results)}/{len(queries)} queries)")
    print(f"{'='*60}")
    print(f"  NDCG@10:       {overall['ndcg_at_10']:.4f}  "
          f"{'✓' if overall['ndcg_at_10'] >= _NDCG_TARGET else '✗'} "
          f"(target ≥ {_NDCG_TARGET})")
    print(f"  Recall@10:     {overall['recall_at_10']:.4f}  "
          f"{'✓' if overall['recall_at_10'] >= _RECALL_TARGET else '✗'} "
          f"(target ≥ {_RECALL_TARGET})")
    if overall["faithfulness"] is not None:
        print(f"  Faithfulness:  {overall['faithfulness']:.4f}  "
              f"{'✓' if overall['faithfulness'] >= _FAITH_TARGET else '✗'} "
              f"(target ≥ {_FAITH_TARGET})")
    else:
        print("  Faithfulness:  (skipped)")

    print(f"\nPer-type breakdown:")
    for qtype, stats in type_breakdown.items():
        faith_str = (
            f"  faith={stats['faithfulness']:.4f}"
            if stats["faithfulness"] is not None
            else ""
        )
        print(
            f"  {qtype:<14} n={stats['count']:3d}  "
            f"NDCG={stats['ndcg_at_10']:.4f}  "
            f"Recall={stats['recall_at_10']:.4f}"
            f"{faith_str}"
        )

    if dense_summary:
        print(f"\nDense-only subset (n={dense_summary['count']}):")
        print(f"  NDCG={dense_summary['ndcg_at_10']:.4f}  "
              f"Recall={dense_summary['recall_at_10']:.4f}")

    print(f"\nRetrieval path distribution: {dict(path_counts)}")
    print(f"\nResults written to: {results_path}")
    print(f"Latest metrics:     {latest_path}")
    print(f"{'='*60}\n")

    # Diagnose misses (NDCG < target) by showing worst-performing queries
    if overall["ndcg_at_10"] < _NDCG_TARGET:
        worst = sorted(per_query_results, key=lambda r: r["ndcg_at_10"])[:5]
        print("Diagnosis — 5 lowest NDCG queries:")
        for r in worst:
            print(f"  [{r['id']}] {r['query'][:60]}…  "
                  f"NDCG={r['ndcg_at_10']:.4f}  path={r['retrieval_path']}")
        print()

    return full_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG /query endpoint against the golden query set."
    )
    parser.add_argument(
        "--golden-set",
        required=True,
        help="Path to golden_queries.json",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8002",
        help="Base URL of the running RAG service (default: http://localhost:8002)",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip Ragas faithfulness scoring (faster, avoids OpenAI cost)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max simultaneous /query requests (default: 4)",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run(
            golden_set_path=args.golden_set,
            endpoint=args.endpoint,
            skip_faithfulness=args.skip_faithfulness,
            concurrency=args.concurrency,
        )
    )
    sys.exit(0)
