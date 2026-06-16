# Pulls the 14 false-zero queries (exploratory/edge, synthesize path, NDCG=0)
# from the latest eval results and fetches product descriptions from PostgreSQL
# for every retrieved ASIN. Output is written to eval/adjudication_candidates.json
# for manual relevance labeling.
#
# Usage (run from rag-service/):
#   python3 eval/pull_adjudication_candidates.py
#
# Output format:
#   [
#     {
#       "id": "q022",
#       "query": "flowy dress for a beach wedding guest",
#       "type": "exploratory",
#       "current_relevance": { "B07CQ84KLT": 2, ... },   # existing labels
#       "candidates": [
#         {
#           "product_id": "B07PJXX8XV",
#           "name": "...",
#           "category": "...",
#           "price": 34.99,
#           "description": "...",
#           "your_label": null        # <-- fill in 0, 1, or 2
#         },
#         ...
#       ]
#     },
#     ...
#   ]
#
# Labeling rubric:
#   2 = directly answers the query; you'd be satisfied stopping here
#   1 = partially relevant; addresses a subset of the query intent
#   0 = irrelevant or misleading
#   null = not yet judged (leave as null if unsure, revisit later)

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

_POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://gorse:gorse_pass@localhost:5432/gorse",
)

OUTPUT_PATH = Path("eval/adjudication_candidates.json")


def load_latest_results() -> dict:
    files = sorted(glob.glob("eval/results_*.json"))
    if not files:
        print("ERROR: no eval/results_*.json found — run run_eval.py first.")
        sys.exit(1)
    latest = files[-1]
    print(f"Loading results from: {latest}")
    with open(latest) as f:
        return json.load(f)


def find_false_zeros(results: dict) -> list[dict]:
    """Return per-query result dicts that are false zeros:
    any query on the synthesize or best_effort path with NDCG=0.

    These are structural false zeros: the system retrieved real products but
    the golden label set doesn't include any of the retrieved ASINs.
    best_effort queries are especially likely to have unlabeled retrieved
    products because their cosine scores are low but the products may still
    be genuinely relevant.
    """
    return [
        q for q in results["per_query"]
        if q["retrieval_path"] in ("synthesize", "best_effort")
        and q["ndcg_at_10"] == 0.0
    ]


def fetch_product_descriptions(product_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch name, category, price, description for a list of ASINs."""
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
            return {
                row[0]: {
                    "name":        row[1] or "",
                    "category":    row[2] or "",
                    "price":       float(row[3]) if row[3] is not None else None,
                    "description": row[4] or "",
                }
                for row in cur.fetchall()
            }
    finally:
        conn.close()


def build_output(
    false_zeros: list[dict],
    golden_queries: list[dict],
    descriptions: dict[str, dict],
) -> list[dict]:
    golden_by_id = {q["id"]: q for q in golden_queries}

    output = []
    for result in false_zeros:
        qid = result["id"]
        golden = golden_by_id.get(qid, {})

        # Deduplicate retrieved IDs while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for pid in result.get("retrieved_ids", []):
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)

        candidates = []
        for pid in unique_ids:
            info = descriptions.get(pid, {})
            candidates.append(
                {
                    "product_id":  pid,
                    "name":        info.get("name", "(not in rag_products)"),
                    "category":    info.get("category", ""),
                    "price":       info.get("price"),
                    "description": info.get("description", "")[:300],
                    "your_label":  None,   # <-- fill in 0 / 1 / 2
                }
            )

        output.append(
            {
                "id":               qid,
                "query":            result["query"],
                "type":             result["type"],
                "dense_only":       golden.get("dense_only", False),
                "current_relevance": golden.get("relevance", {}),
                "candidates":       candidates,
                "notes": (
                    "Label each candidate 0/1/2. "
                    "Then copy labeled IDs into golden_queries.json under 'relevance'."
                ),
            }
        )

    return output


def main() -> None:
    # Load latest eval results
    results = load_latest_results()

    # Find the 14 false-zero queries
    false_zeros = find_false_zeros(results)
    print(f"Found {len(false_zeros)} false-zero queries to adjudicate.")

    if not false_zeros:
        print("Nothing to adjudicate — all exploratory/edge synthesize queries have NDCG > 0.")
        sys.exit(0)

    # Load golden queries for current labels
    golden_path = Path("eval/golden_queries.json")
    with open(golden_path) as f:
        golden_queries = json.load(f)

    # Collect all unique product IDs across all false-zero queries
    all_pids: list[str] = []
    seen_global: set[str] = set()
    for q in false_zeros:
        for pid in q.get("retrieved_ids", []):
            if pid not in seen_global:
                seen_global.add(pid)
                all_pids.append(pid)

    print(f"Fetching descriptions for {len(all_pids)} unique product IDs from PostgreSQL…")
    descriptions = fetch_product_descriptions(all_pids)
    found = sum(1 for pid in all_pids if pid in descriptions)
    print(f"  Found in rag_products: {found}/{len(all_pids)}")

    # Build output
    output = build_output(false_zeros, golden_queries, descriptions)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten to: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Open {OUTPUT_PATH}")
    print(f"  2. For each query, set 'your_label' to 0, 1, or 2 for each candidate")
    print(f"  3. Run:  python3 eval/apply_adjudication.py")
    print(f"     to merge labels back into golden_queries.json automatically")


if __name__ == "__main__":
    main()
