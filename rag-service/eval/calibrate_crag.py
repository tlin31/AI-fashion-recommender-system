# CRAG threshold calibration via cache-and-replay.
#
# CRAG thresholds affect routing only — never retrieval. Given a query, the
# candidates hybrid_search returns and their milvus_score are identical whether
# the threshold is 0.65 or 0.80. So the expensive half (embed + retrieve +
# rewrite + retrieve again) runs ONCE, and every threshold combination is
# replayed against that cache offline. 1.2 eval runs of API spend instead of 12.
#
# Same principle as sweeping a classifier's decision threshold over cached
# scores to draw an ROC curve: score once, threshold many times.
#
# The cache is only valid while the cached stage is independent of what you're
# sweeping. Changing RRF k, the candidate pool, or the embedding model
# invalidates it — hence the fingerprint, which --grid verifies before running.
#
# Usage:
#   python eval/calibrate_crag.py --build-cache               # needs Postgres + Milvus + OpenAI
#   python eval/calibrate_crag.py --build-cache --limit 3     # cheap smoke test first
#
# Deliberately bypasses the HTTP API: no Redis response cache, no guardrail
# call, and no dependency on a running uvicorn.

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env", override=True)

from pymilvus import MilvusClient  # noqa: E402

from pipeline.crag import _grade, _rewrite_query  # noqa: E402
from pipeline.retrieval import (  # noqa: E402
    _BM25_MIN_SCORE_RATIO,
    _CANDIDATE_POOL,
    _MAX_CHUNKS_PER_PRODUCT,
    _RRF_K,
    build_bm25_index,
    hybrid_search,
)
from pipeline.utils import _EMBED_MODEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Must match api/routes.py — the grid replays what production would have seen.
_RETRIEVAL_TOP_K = 20

_DEFAULT_CACHE = _REPO_ROOT / "eval" / "calibration_cache.json"
_DEFAULT_RESULTS = _REPO_ROOT / "eval" / "calibration_results.json"


# ---------------------------------------------------------------------------
# Fingerprint — guards against silently replaying a stale cache
# ---------------------------------------------------------------------------

def _fingerprint(milvus_client: MilvusClient, collection_name: str) -> dict:
    """Capture everything the cached candidates depend on.

    --grid refuses to run when this differs from the live configuration.
    A stale cache is worse than no cache: it fails silently and produces
    confident, wrong thresholds.
    """
    try:
        row_count = milvus_client.get_collection_stats(collection_name).get("row_count")
    except Exception as exc:
        logger.warning("Could not read collection stats: %s", exc)
        row_count = None

    return {
        "milvus_collection":      collection_name,
        "milvus_row_count":       row_count,
        "embed_model":            _EMBED_MODEL,
        "rrf_k":                  _RRF_K,
        "candidate_pool":         _CANDIDATE_POOL,
        "max_chunks_per_product": _MAX_CHUNKS_PER_PRODUCT,
        "bm25_min_score_ratio":   _BM25_MIN_SCORE_RATIO,
        "retrieval_top_k":        _RETRIEVAL_TOP_K,
    }


# ---------------------------------------------------------------------------
# Milvus connection (mirrors main.py, without importing the FastAPI app)
# ---------------------------------------------------------------------------

def _connect_milvus() -> tuple[MilvusClient, str]:
    host = os.environ.get("MILVUS_HOST", "localhost")
    port = int(os.environ.get("MILVUS_PORT", 19530))
    collection_name = os.environ.get("MILVUS_COLLECTION", "fashion_rag")

    client = MilvusClient(uri=f"http://{host}:{port}")
    client.load_collection(collection_name)
    logger.info("Milvus '%s' loaded from %s:%s", collection_name, host, port)
    return client, collection_name


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------

def _slim(chunk: dict) -> dict:
    """Keep only the fields the offline replay needs.

    text is required — the grid runs the cross-encoder over (query, text) pairs.
    milvus_score is required — it is what _grade() averages.
    metadata is kept whole; it is small and the reranker reads chunk_type from it.
    """
    return {
        "chunk_id":     chunk.get("chunk_id"),
        "product_id":   chunk.get("product_id"),
        "text":         chunk.get("text", ""),
        "milvus_score": chunk.get("milvus_score", 0.0),
        "score":        chunk.get("score", 0.0),
        "metadata":     chunk.get("metadata", {}),
    }


async def _build_one(
    entry: dict,
    *,
    bm25_index,
    milvus_client,
    collection_name: str,
) -> dict:
    """Retrieve, force one rewrite+retry, and time each stage.

    The rewrite runs for EVERY query regardless of which band its initial score
    falls in. A combo with a lower HIGH threshold routes different queries down
    the retry path, and the replay can only simulate that if the retry result
    already exists for all of them.
    """
    query = entry["query"]

    t0 = time.monotonic()
    initial = await hybrid_search(
        query=query,
        filters=None,
        top_k=_RETRIEVAL_TOP_K,
        bm25_index=bm25_index,
        milvus_client=milvus_client,
        collection_name=collection_name,
    )
    initial_ms = (time.monotonic() - t0) * 1000

    t1 = time.monotonic()
    rewritten_query = await _rewrite_query(query)
    rewrite_ms = (time.monotonic() - t1) * 1000

    t2 = time.monotonic()
    rewritten = await hybrid_search(
        query=rewritten_query,
        filters=None,
        top_k=_RETRIEVAL_TOP_K,
        bm25_index=bm25_index,
        milvus_client=milvus_client,
        collection_name=collection_name,
    )
    retry_retrieval_ms = (time.monotonic() - t2) * 1000

    return {
        "id":                   entry["id"],
        "query":                query,
        "type":                 entry.get("type", "unknown"),
        "dense_only":           entry.get("dense_only", False),
        "initial_score":        round(_grade(initial), 6),
        "initial_candidates":   [_slim(c) for c in initial],
        "rewritten_query":      rewritten_query,
        # _rewrite_query falls back to the original query on API failure, so a
        # no-op rewrite is a signal the retry path was never really exercised.
        "rewrite_changed":      rewritten_query.strip() != query.strip(),
        "rewritten_score":      round(_grade(rewritten), 6),
        "rewritten_candidates": [_slim(c) for c in rewritten],
        "initial_retrieval_ms": round(initial_ms, 1),
        "rewrite_ms":           round(rewrite_ms, 1),
        "retry_retrieval_ms":   round(retry_retrieval_ms, 1),
    }


async def build_cache(
    golden_set_path: str,
    out_path: Path,
    limit: int | None = None,
    concurrency: int = 4,
) -> dict:
    golden_path = Path(golden_set_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    queries: list[dict] = json.loads(golden_path.read_text())
    if limit:
        queries = queries[:limit]
        logger.info("--limit %d: smoke run over the first %d queries", limit, len(queries))

    logger.info("Building BM25 index from Postgres …")
    bm25_index = await build_bm25_index()
    milvus_client, collection_name = _connect_milvus()

    fingerprint = _fingerprint(milvus_client, collection_name)
    logger.info("Fingerprint: %s", fingerprint)

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    total = len(queries)

    async def with_limit(entry: dict) -> dict:
        nonlocal done
        async with semaphore:
            result = await _build_one(
                entry,
                bm25_index=bm25_index,
                milvus_client=milvus_client,
                collection_name=collection_name,
            )
        done += 1
        if done % 10 == 0 or done == total:
            logger.info("  %d/%d queries cached", done, total)
        return result

    started = time.monotonic()
    entries = await asyncio.gather(*(with_limit(q) for q in queries))
    elapsed = time.monotonic() - started

    cache = {
        "built_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "golden_set":  str(golden_path),
        "fingerprint": fingerprint,
        "n_queries":   len(entries),
        "entries":     entries,
    }

    out_path.write_text(json.dumps(cache, indent=2))

    # ---- summary ----------------------------------------------------------
    scores = [e["initial_score"] for e in entries]
    unchanged = sum(1 for e in entries if not e["rewrite_changed"])
    empty_initial = sum(1 for e in entries if not e["initial_candidates"])
    improved = sum(1 for e in entries if e["rewritten_score"] > e["initial_score"])

    print()
    print("=" * 60)
    print(f"Calibration cache built  ({len(entries)} queries in {elapsed:.1f}s)")
    print("=" * 60)
    print(f"  initial_score  min={min(scores):.4f}  "
          f"mean={sum(scores)/len(scores):.4f}  max={max(scores):.4f}")
    print(f"  rewrite improved score:     {improved}/{len(entries)}")
    print(f"  rewrite returned unchanged: {unchanged}/{len(entries)}")
    print(f"  empty initial retrieval:    {empty_initial}/{len(entries)}")
    print(f"  mean rewrite_ms:            "
          f"{sum(e['rewrite_ms'] for e in entries)/len(entries):.0f}")
    print(f"  mean retry_retrieval_ms:    "
          f"{sum(e['retry_retrieval_ms'] for e in entries)/len(entries):.0f}")
    print(f"  written to: {out_path}  "
          f"({out_path.stat().st_size / 1_048_576:.1f} MB)")
    print("=" * 60)

    return cache


# ---------------------------------------------------------------------------
# Graders — all are pure functions of the candidates' milvus_scores, so any of
# them can be evaluated offline from the cache with no new API calls.
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _grade_mean20(chunks: list[dict]) -> float:
    """Production grader: mean cosine over all candidates, weak tail included."""
    return _mean([c.get("milvus_score", 0.0) for c in chunks])


def _grade_topn(chunks: list[dict], n: int) -> float:
    scores = sorted((c.get("milvus_score", 0.0) for c in chunks), reverse=True)
    return _mean(scores[:n])


_GRADERS = {
    "mean20": _grade_mean20,                          # incumbent
    "max":    lambda c: _grade_topn(c, 1),
    "top3":   lambda c: _grade_topn(c, 3),
}

# Locked for the incumbent grader; other graders reuse the same PERCENTILE
# positions within their own distribution, so combos are compared at equal
# routing rates rather than at arbitrary absolute cutoffs.
_HIGH_ANCHORS = [0.45, 0.50, 0.55, 0.60, 0.65]
_LOW_ANCHORS  = [0.10, 0.32, 0.38, 0.43]

_INCUMBENT = ("mean20", 0.45, 0.10)
_NDCG_K = 10


def _percentile_of(value: float, population: list[float]) -> float:
    return sum(1 for v in population if v < value) / len(population)


def _value_at(pct: float, population: list[float]) -> float:
    s = sorted(population)
    idx = min(int(round(pct * (len(s) - 1))), len(s) - 1)
    return round(s[idx], 4)


def _threshold_grid(grader_name: str, scores: list[float],
                    incumbent_scores: list[float]) -> tuple[list[float], list[float]]:
    if grader_name == _INCUMBENT[0]:
        return list(_HIGH_ANCHORS), list(_LOW_ANCHORS)
    highs = [_value_at(_percentile_of(a, incumbent_scores), scores) for a in _HIGH_ANCHORS]
    lows  = [_value_at(_percentile_of(a, incumbent_scores), scores) for a in _LOW_ANCHORS]
    return sorted(set(highs)), sorted(set(lows))


# ---------------------------------------------------------------------------
# Replay — mirrors run_crag()'s branching against cached candidates
# ---------------------------------------------------------------------------

def _replay(entry: dict, grader, high: float, low: float) -> tuple[str, str]:
    """Return (which_candidate_set, path) for one query under one config.

    Only two candidate sets can ever be selected, which is why reranking is
    computed once per set rather than once per combo.

    Note on retries: run_crag() rewrites the ORIGINAL query on every attempt and
    never updates it, so with temperature=0 attempt 2 reproduces attempt 1
    exactly. One cached rewrite is therefore sufficient to simulate the loop.
    """
    initial = entry["initial_candidates"]
    if not initial:
        return "initial", "fallback"

    score = grader(initial)
    if score >= high:
        return "initial", "synthesize"
    if score < low:
        return "initial", "best_effort"

    rewritten = entry["rewritten_candidates"]
    if not rewritten:
        return "initial", "best_effort"

    new_score = grader(rewritten)
    if new_score >= high:
        return "rewritten", "retry"
    if new_score < low:
        return "initial", "best_effort"
    # Still borderline after every attempt — run_crag keeps the better set.
    return ("rewritten" if new_score > score else "initial"), "best_effort"


# ---------------------------------------------------------------------------
# Bootstrap — does the winning combo survive resampling the query set?
# ---------------------------------------------------------------------------

def _bootstrap(per_query: dict[tuple, list[float]], combos: list[tuple],
               incumbent: tuple, n_iter: int, seed: int) -> dict:
    import random

    rng = random.Random(seed)
    n = len(next(iter(per_query.values())))
    winners: dict[tuple, int] = {c: 0 for c in combos}
    deltas: list[float] = []

    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        best, best_score = None, float("-inf")
        for combo in combos:
            vals = per_query[combo]
            m = sum(vals[i] for i in idx) / n
            if m > best_score:
                best, best_score = combo, m
        winners[best] += 1
        inc = per_query[incumbent]
        deltas.append(best_score - sum(inc[i] for i in idx) / n)

    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
    ranked = sorted(winners.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "n_iter": n_iter,
        "selection_stability": [
            {"combo": f"{g}/{h}/{l}", "won_pct": round(100 * c / n_iter, 1)}
            for (g, h, l), c in ranked if c
        ],
        "improvement_vs_incumbent": {
            "mean":   round(sum(deltas) / len(deltas), 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "excludes_zero": bool(lo > 0),
        },
    }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def run_grid(cache_path: Path, golden_set_path: str, out_path: Path,
             n_bootstrap: int, seed: int) -> dict:
    from eval.metrics import ndcg_at_k, recall_at_k
    from pipeline.reranker import Reranker

    cache = json.loads(cache_path.read_text())
    entries = cache["entries"]
    golden = {q["id"]: q for q in json.loads(Path(golden_set_path).read_text())}

    # ---- staleness check --------------------------------------------------
    fp = cache["fingerprint"]
    live = {
        "embed_model":            _EMBED_MODEL,
        "rrf_k":                  _RRF_K,
        "candidate_pool":         _CANDIDATE_POOL,
        "max_chunks_per_product": _MAX_CHUNKS_PER_PRODUCT,
        "bm25_min_score_ratio":   _BM25_MIN_SCORE_RATIO,
        "retrieval_top_k":        _RETRIEVAL_TOP_K,
    }
    drift = {k: (fp.get(k), v) for k, v in live.items() if fp.get(k) != v}
    if drift:
        raise SystemExit(
            "Calibration cache is stale — retrieval config changed since it was "
            f"built: {drift}. Rebuild with --build-cache."
        )
    logger.info("Fingerprint OK (retrieval config unchanged since cache was built)")

    # ---- rerank each candidate set ONCE -----------------------------------
    # The chosen set is always either "initial" or "rewritten", so 2 rerank
    # passes per query cover every combo in the grid.
    logger.info("Loading cross-encoder …")
    reranker = Reranker()
    logger.info("Reranking %d candidate sets …", 2 * len(entries))

    ranked: dict[tuple[str, str], list[str]] = {}
    for i, e in enumerate(entries, 1):
        for which in ("initial", "rewritten"):
            chunks = e[f"{which}_candidates"]
            top = reranker.rerank(e["query"], chunks, top_k=_NDCG_K) if chunks else []
            ranked[(e["id"], which)] = [c["product_id"] for c in top]
        if i % 25 == 0:
            logger.info("  %d/%d queries reranked", i, len(entries))

    # ---- evaluate every combo --------------------------------------------
    incumbent_scores = [_GRADERS["mean20"](e["initial_candidates"]) for e in entries]
    rows: list[dict] = []
    per_query_ndcg: dict[tuple, list[float]] = {}

    for gname, grader in _GRADERS.items():
        scores = [grader(e["initial_candidates"]) for e in entries]
        highs, lows = _threshold_grid(gname, scores, incumbent_scores)
        for high in highs:
            for low in lows:
                if low >= high:
                    continue
                ndcgs, recalls, paths, retried = [], [], {}, 0
                for e in entries:
                    which, path = _replay(e, grader, high, low)
                    paths[path] = paths.get(path, 0) + 1
                    if path in ("retry", "best_effort"):
                        # entered the retry band → paid the rewrite round-trip
                        s = grader(e["initial_candidates"])
                        if s >= low:
                            retried += 1
                    ids = ranked[(e["id"], which)]
                    rel = golden[e["id"]]["relevance"]
                    ndcgs.append(ndcg_at_k(ids, rel, _NDCG_K))
                    recalls.append(recall_at_k(
                        ids, {p for p, v in rel.items() if v > 0}, _NDCG_K))

                combo = (gname, high, low)
                per_query_ndcg[combo] = ndcgs
                retry_rate = retried / len(entries)
                mean_retry_ms = _mean(
                    [e["rewrite_ms"] + e["retry_retrieval_ms"] for e in entries])
                rows.append({
                    "grader": gname, "high": high, "low": low,
                    "ndcg_at_10":   round(_mean(ndcgs), 4),
                    "recall_at_10": round(_mean(recalls), 4),
                    "retry_rate":   round(retry_rate, 3),
                    "expected_latency_delta_ms": round(retry_rate * mean_retry_ms, 1),
                    "paths": paths,
                })

    rows.sort(key=lambda r: r["ndcg_at_10"], reverse=True)

    # ---- constrained selection + bootstrap --------------------------------
    feasible = [r for r in rows if r["retry_rate"] < 0.20]
    winner = feasible[0] if feasible else rows[0]
    combos = list(per_query_ndcg.keys())
    incumbent = _INCUMBENT if _INCUMBENT in per_query_ndcg else combos[0]
    boot = _bootstrap(per_query_ndcg, combos, incumbent, n_bootstrap, seed)

    result = {
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache":         str(cache_path),
        "n_queries":     len(entries),
        "incumbent":     {"grader": incumbent[0], "high": incumbent[1], "low": incumbent[2]},
        "constraint":    "retry_rate < 0.20",
        "selected":      {k: winner[k] for k in ("grader", "high", "low", "ndcg_at_10",
                                                 "recall_at_10", "retry_rate")},
        "bootstrap":     boot,
        "score_distribution": {
            g: {
                "min":  round(min(v), 4), "p50": round(_value_at(0.50, v), 4),
                "max":  round(max(v), 4),
            }
            for g, v in ((g, [f(e["initial_candidates"]) for e in entries])
                         for g, f in _GRADERS.items())
        },
        "grid": rows,
    }
    out_path.write_text(json.dumps(result, indent=2))

    # ---- report -----------------------------------------------------------
    inc_row = next(r for r in rows if (r["grader"], r["high"], r["low"]) == incumbent)
    print()
    print("=" * 86)
    print(f"CRAG threshold grid — {len(rows)} combos over {len(entries)} queries")
    print("=" * 86)
    print(f"{'grader':<8} {'HIGH':>6} {'LOW':>6} {'NDCG@10':>9} {'Recall@10':>10} "
          f"{'retry':>7} {'+lat ms':>8}")
    print("-" * 86)
    for r in rows[:12]:
        mark = "  <- incumbent" if (r["grader"], r["high"], r["low"]) == incumbent else ""
        print(f"{r['grader']:<8} {r['high']:>6.3f} {r['low']:>6.3f} "
              f"{r['ndcg_at_10']:>9.4f} {r['recall_at_10']:>10.4f} "
              f"{r['retry_rate']:>7.2f} {r['expected_latency_delta_ms']:>8.1f}{mark}")
    if (inc_row["grader"], inc_row["high"], inc_row["low"]) not in [
            (r["grader"], r["high"], r["low"]) for r in rows[:12]]:
        print("...")
        print(f"{inc_row['grader']:<8} {inc_row['high']:>6.3f} {inc_row['low']:>6.3f} "
              f"{inc_row['ndcg_at_10']:>9.4f} {inc_row['recall_at_10']:>10.4f} "
              f"{inc_row['retry_rate']:>7.2f} "
              f"{inc_row['expected_latency_delta_ms']:>8.1f}  <- incumbent")
    print("-" * 86)
    b = boot["improvement_vs_incumbent"]
    print(f"Selected (retry_rate < 0.20): {winner['grader']} "
          f"HIGH={winner['high']} LOW={winner['low']}  NDCG={winner['ndcg_at_10']}")
    print(f"Incumbent:                    {incumbent[0]} "
          f"HIGH={incumbent[1]} LOW={incumbent[2]}  NDCG={inc_row['ndcg_at_10']}")
    print()
    print(f"Bootstrap ({boot['n_iter']} resamples):")
    print(f"  improvement vs incumbent: {b['mean']:+.4f}  "
          f"95% CI [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")
    print(f"  CI excludes zero: {b['excludes_zero']}"
          f"{'' if b['excludes_zero'] else '  -> calibration CONFIRMS the incumbent'}")
    print("  selection stability (top 5):")
    for s in boot["selection_stability"][:5]:
        print(f"    {s['combo']:<24} won {s['won_pct']:>5.1f}% of resamples")
    print("=" * 86)
    print(f"written to: {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CRAG threshold calibration")
    parser.add_argument("--build-cache", action="store_true",
                        help="Run retrieval + forced rewrite over the golden set and cache it")
    parser.add_argument("--grid", action="store_true",
                        help="Replay the threshold grid offline against the cache (free)")
    parser.add_argument("--golden-set",
                        default=str(_REPO_ROOT / "eval" / "golden_queries.json"))
    parser.add_argument("--cache", default=str(_DEFAULT_CACHE))
    parser.add_argument("--results", default=str(_DEFAULT_RESULTS))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N queries (smoke test)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for the selection-stability check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.build_cache or args.grid):
        parser.error("nothing to do — pass --build-cache and/or --grid")

    if args.build_cache:
        asyncio.run(build_cache(
            golden_set_path=args.golden_set,
            out_path=Path(args.cache),
            limit=args.limit,
            concurrency=args.concurrency,
        ))

    if args.grid:
        run_grid(
            cache_path=Path(args.cache),
            golden_set_path=args.golden_set,
            out_path=Path(args.results),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
