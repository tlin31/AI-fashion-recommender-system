# Full eval harness runner. Loads golden_queries.json, queries the live /query endpoint
# for each entry, and computes Recall@10, NDCG@10, and Ragas faithfulness. Writes
# results to eval/latest_metrics.json for the regression gate to compare against baseline.

from __future__ import annotations

import argparse
import json


async def run(golden_set_path: str) -> dict:
    # Loads golden_queries.json, runs each query through the live /query endpoint,
    # and computes Recall@10, NDCG@10 (custom), and faithfulness (Ragas).
    # Writes results to eval/latest_metrics.json for the regression gate.
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", required=True)
    args = parser.parse_args()

    import asyncio
    asyncio.run(run(args.golden_set))
