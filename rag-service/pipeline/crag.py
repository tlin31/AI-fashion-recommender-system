# Corrective RAG loop: grades retrieved candidates by cosine similarity, then decides
# whether to synthesize directly, rewrite the query and retry, or fall back to trending
# products. A hard time budget (CRAG_TIME_BUDGET_S) prevents the retry loop from
# blowing the latency SLA.

from __future__ import annotations

from typing import Literal


RetriivalPath = Literal["synthesize", "retry", "fallback"]


async def run_crag(
    query: str,
    candidates: list[dict],
    query_embedding: list[float],
) -> tuple[list[dict], RetriivalPath]:
    # Grade candidates by cosine similarity against the query embedding.
    # score > HIGH_THRESHOLD  → pass candidates through unchanged
    # score in [LOW, HIGH]    → rewrite query with GPT-4o-mini, retry retrieval once
    # score < LOW_THRESHOLD
    #   or max retries hit    → swap in non-personalized trending products as fallback
    #
    # A hard wall-clock budget (CRAG_TIME_BUDGET_S) prevents the retry loop from
    # blowing the latency SLA — hits fallback path immediately if exceeded.
    raise NotImplementedError
