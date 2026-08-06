# Adjudication round 3 — pools over everything the CRAG threshold grid can surface.
#
# Rounds 1 and 2 labeled only a subset of what the system returned, leaving ~61% of
# top-10 slots unjudged. Unlabeled counts as relevance 0, so Recall@10 and NDCG@10 are
# both systematically understated.
#
# Round 3 also closes a bias that would otherwise distort threshold calibration:
# different threshold combos route different queries down the retry path, and
# retry returns rewritten-query candidates that are MORE likely to be unlabeled.
# Combos would be penalised for surfacing unjudged products rather than for being
# worse. Pooling over the union of both candidate sets removes that.
#
# Pool = union over every query of:
#     rerank(initial_candidates)[:depth]  ∪  rerank(rewritten_candidates)[:depth]
# which is exactly the set any of the 50 grid combos could return.
#
# Usage (run from rag-service/):
#   python3 eval/pull_adjudication_round3.py                 # depth 10 (full pool)
#   python3 eval/pull_adjudication_round3.py --max-rank 3    # shallow pass first
#
# Candidates are ordered by rank so you can label top-down and stop at any point;
# metrics improve monotonically as depth increases. Rerun the eval between passes
# to see the marginal effect of labeling depth.
#
# Labeling rubric (unchanged from rounds 1–2):
#   2 = directly answers the query; you'd be satisfied stopping here
#   1 = partially relevant; addresses a subset of the query intent
#   0 = irrelevant or misleading
#   null = not yet judged

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

load_dotenv(_REPO_ROOT / ".env", override=True)

_POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://gorse:gorse_pass@localhost:5432/gorse",
)

_CACHE_PATH  = _REPO_ROOT / "eval" / "calibration_cache.json"
_GOLDEN_PATH = _REPO_ROOT / "eval" / "golden_queries.json"
_OUTPUT_PATH = _REPO_ROOT / "eval" / "adjudication_round3.json"

_POOL_DEPTH = 10


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    conn = psycopg2.connect(_POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, name, category, price, description
                FROM   rag_products
                WHERE  product_id = ANY(%s)
                """,
                (product_ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        pid: {"name": name, "category": category,
              "price": float(price) if price is not None else None,
              "description": description}
        for pid, name, category, price, description in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull adjudication round 3 candidates")
    parser.add_argument("--max-rank", type=int, default=_POOL_DEPTH,
                        help="Only pool candidates at or above this rank (default 10)")
    parser.add_argument("--cache", default=str(_CACHE_PATH))
    parser.add_argument("--golden", default=str(_GOLDEN_PATH))
    parser.add_argument("--out", default=str(_OUTPUT_PATH))
    args = parser.parse_args()

    from pipeline.reranker import Reranker

    cache = json.loads(Path(args.cache).read_text())["entries"]
    golden = {q["id"]: q for q in json.loads(Path(args.golden).read_text())}

    print(f"Reranking {2 * len(cache)} candidate sets (pool depth {args.max_rank}) …")
    reranker = Reranker()

    # qid -> {product_id: best rank across both candidate sets}
    pool: dict[str, dict[str, int]] = {}
    for i, e in enumerate(cache, 1):
        best: dict[str, int] = {}
        for which in ("initial", "rewritten"):
            chunks = e[f"{which}_candidates"]
            if not chunks:
                continue
            for rank, c in enumerate(reranker.rerank(e["query"], chunks, top_k=_POOL_DEPTH), 1):
                pid = c["product_id"]
                best[pid] = min(best.get(pid, 10**6), rank)
        pool[e["id"]] = best
        if i % 25 == 0:
            print(f"  {i}/{len(cache)} queries pooled")

    # Keep only unjudged (query, product) pairs within the requested depth.
    needed: dict[str, list[tuple[str, int]]] = {}
    all_pids: set[str] = set()
    for qid, ranked in pool.items():
        existing = golden[qid].get("relevance", {})
        unjudged = [(pid, rk) for pid, rk in ranked.items()
                    if rk <= args.max_rank and pid not in existing]
        if unjudged:
            unjudged.sort(key=lambda x: x[1])
            needed[qid] = unjudged
            all_pids.update(pid for pid, _ in unjudged)

    print(f"Fetching descriptions for {len(all_pids)} products …")
    details = _fetch_product_details(sorted(all_pids))

    entries = []
    for qid in sorted(needed):
        g = golden[qid]
        entries.append({
            "id":                 qid,
            "query":              g["query"],
            "type":               g.get("type", "unknown"),
            "current_relevance":  g.get("relevance", {}),
            "candidates": [
                {
                    "product_id":  pid,
                    "rank":        rk,
                    "name":        details.get(pid, {}).get("name", ""),
                    "category":    details.get(pid, {}).get("category", ""),
                    "price":       details.get(pid, {}).get("price"),
                    "description": details.get(pid, {}).get("description", ""),
                    "your_label":  None,
                }
                for pid, rk in needed[qid]
            ],
        })

    Path(args.out).write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    total = sum(len(e["candidates"]) for e in entries)
    by_rank: dict[int, int] = {}
    for e in entries:
        for c in e["candidates"]:
            by_rank[c["rank"]] = by_rank.get(c["rank"], 0) + 1

    print()
    print("=" * 62)
    print(f"Adjudication round 3 — pool depth {args.max_rank}")
    print("=" * 62)
    print(f"  queries needing judgments : {len(entries)}")
    print(f"  (query, product) pairs    : {total}")
    print(f"  unique products           : {len(all_pids)}")
    missing = [p for p in all_pids if p not in details]
    if missing:
        print(f"  WARNING: {len(missing)} products missing from rag_products")
    print("  by rank:", "  ".join(f"r{k}:{v}" for k, v in sorted(by_rank.items())))
    print(f"  written to: {args.out}")
    print()
    print("  Label top-down (rank 1 first) and stop whenever you like — metrics")
    print("  improve monotonically with depth. Then:")
    print(f"    python3 eval/apply_adjudication.py --input {args.out}")
    print("    python3 eval/run_eval.py --golden-set eval/golden_queries.json --skip-faithfulness")
    print("=" * 62)


if __name__ == "__main__":
    main()
