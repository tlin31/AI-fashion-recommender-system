# Corrective RAG loop: grades retrieved candidates by cosine similarity, then decides
# whether to synthesize directly, rewrite the query and retry, or fall back to trending
# products. A hard time budget (CRAG_TIME_BUDGET_S) prevents the retry loop from
# blowing the latency SLA.

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Literal

import psycopg2

from pipeline.retrieval import hybrid_search
from pipeline.utils import get_openai_client

logger = logging.getLogger(__name__)

# Threshold calibration: above HIGH → synthesize, between LOW and HIGH → rewrite,
# below LOW → trending fallback. Values from Week 2 eval notebook.
_CRAG_HIGH_THRESHOLD = float(os.environ.get("CRAG_HIGH_THRESHOLD", "0.75"))
_CRAG_LOW_THRESHOLD  = float(os.environ.get("CRAG_LOW_THRESHOLD",  "0.45"))
_CRAG_MAX_RETRIES    = int(os.environ.get("CRAG_MAX_RETRIES",      "2"))
_CRAG_TIME_BUDGET_S  = float(os.environ.get("CRAG_TIME_BUDGET_S",  "3.5"))

_POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://gorse:gorse_pass@localhost:5432/gorse",
)

# Intentional typo preserved from the original stub to avoid rename ripple.
RetriivalPath = Literal["synthesize", "retry", "fallback", "best_effort"]


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _grade(candidates: list[dict]) -> float:
    """Return mean Milvus cosine score across all candidates.

    Milvus uses COSINE metric, so each chunk's milvus_score IS the cosine
    similarity between the query vector and that chunk's vector — no extra
    API call needed. Mean over the batch gives a single quality signal.
    Returns 0.0 on empty input so callers always see a below-threshold score.
    """
    if not candidates:
        return 0.0
    scores = [c.get("milvus_score", 0.0) for c in candidates]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

async def _rewrite_query(query: str) -> str:
    """Rephrase the query with broader fashion vocabulary via GPT-4o-mini.

    2s timeout — tight enough to keep the CRAG loop within budget.
    On any failure the original query is returned so the retry still runs.
    """
    try:
        client = get_openai_client()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Rephrase this fashion query with broader vocabulary: {query}"
                        ),
                    }
                ],
                max_tokens=60,
                temperature=0.3,
            ),
            timeout=2.0,
        )
        rewritten = (response.choices[0].message.content or query).strip()
        logger.debug("CRAG rewrite: %r → %r", query, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed (%s); using original query", exc)
        return query


# ---------------------------------------------------------------------------
# Trending fallback
# ---------------------------------------------------------------------------

def _fetch_trending_sync(top_k: int) -> list[dict]:
    """Blocking fetch of the top-k most-rated products from PostgreSQL.

    Ordered by rating_count DESC as a popularity proxy for "trending".
    Called via run_in_executor so it never blocks the async event loop.
    Returns chunks in the same dict shape as hybrid_search results so the
    reranker can process them without any special-casing.
    """
    conn = psycopg2.connect(_POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, name, description,
                       price_range, category, occasion, brand
                FROM   rag_products
                ORDER  BY rating_count DESC NULLS LAST
                LIMIT  %s
                """,
                (top_k,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    chunks = []
    for product_id, name, description, price_range, category, occasion, brand in rows:
        # Combine name and description into a single text field for the reranker.
        text = f"{name or ''}. {description or ''}".strip(". ")
        chunks.append(
            {
                "chunk_id":    f"trending_{product_id}",
                "product_id":  product_id,
                "text":        text,
                "score":       0.0,
                "milvus_score": 0.0,
                "metadata": {
                    "chunk_type":  "description",
                    "price_range": price_range or "",
                    "category":    category or "",
                    "occasion":    occasion or "",
                    "brand":       brand or "",
                },
            }
        )
    return chunks


async def _fetch_trending_fallback(top_k: int = 5) -> list[dict]:
    """Async wrapper: offloads the blocking Postgres call to a thread pool."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch_trending_sync, top_k)
    except Exception as exc:
        logger.error("Trending fallback query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# CRAG main loop
# ---------------------------------------------------------------------------

async def run_crag(
    query: str,
    candidates: list[dict],
    query_embedding: list[float],   # kept for API symmetry; grading uses milvus_score
    *,
    bm25_index,
    milvus_client,
    collection_name: str,
    max_retries: int = _CRAG_MAX_RETRIES,
    time_budget: float = _CRAG_TIME_BUDGET_S,
) -> tuple[list[dict], RetriivalPath]:
    """Grade retrieval quality and choose the next pipeline step.

    Returns (chunks, path) where path is one of:
        "synthesize" — initial candidates scored above the high threshold.
        "retry"      — a rewritten query produced high-quality candidates.
        "fallback"   — quality never reached the high threshold; trending
                       products are returned instead.

    The time budget is checked before each retry attempt, not after, so the
    total CRAG wall-clock time stays within CRAG_TIME_BUDGET_S.
    """
    start = time.monotonic()

    # Empty candidates means retrieval failed entirely — skip straight to fallback.
    if not candidates:
        logger.info("CRAG: empty candidates, falling back to trending")
        return await _fetch_trending_fallback(), "fallback"

    score = _grade(candidates)
    logger.info(
        "CRAG initial score=%.3f (high=%.2f, low=%.2f)",
        score, _CRAG_HIGH_THRESHOLD, _CRAG_LOW_THRESHOLD,
    )

    # High quality — pass candidates straight through to the reranker.
    if score >= _CRAG_HIGH_THRESHOLD:
        return candidates, "synthesize"

    # Very low quality — rewriting is unlikely to jump two thresholds; return the
    # best candidates we already have rather than discarding them for trending products.
    # Trending products have zero overlap with any labeled ASINs and hurt recall badly.
    if score < _CRAG_LOW_THRESHOLD:
        logger.info(
            "CRAG: score=%.3f below low threshold — skipping retries, "
            "returning best retrieved candidates (best_effort)",
            score,
        )
        return candidates, "best_effort"

    # Borderline quality — attempt query rewrites up to max_retries times.
    for attempt in range(1, max_retries + 1):
        elapsed = time.monotonic() - start
        if elapsed >= time_budget:
            logger.warning(
                "CRAG: time budget %.1fs exceeded after %.2fs (attempt %d) — "
                "returning best candidates so far",
                time_budget, elapsed, attempt,
            )
            return candidates, "best_effort"

        rewritten = await _rewrite_query(query)
        new_candidates = await hybrid_search(
            query=rewritten,
            filters=None,
            top_k=20,
            bm25_index=bm25_index,
            milvus_client=milvus_client,
            collection_name=collection_name,
        )

        if not new_candidates:
            logger.info(
                "CRAG attempt %d: empty results after rewrite — keeping pre-rewrite candidates",
                attempt,
            )
            return candidates, "best_effort"

        new_score = _grade(new_candidates)
        logger.info("CRAG attempt %d score=%.3f", attempt, new_score)

        if new_score >= _CRAG_HIGH_THRESHOLD:
            return new_candidates, "retry"

        if new_score < _CRAG_LOW_THRESHOLD:
            # Rewrite degraded quality further — return best candidates seen so far.
            logger.info(
                "CRAG attempt %d: rewrite degraded score to %.3f — "
                "returning best candidates so far",
                attempt, new_score,
            )
            return candidates, "best_effort"

        # Still borderline — keep the better candidates and try once more.
        if new_score > score:
            candidates, score = new_candidates, new_score

    # All retries exhausted without crossing the high threshold — return whatever
    # candidates we have (best seen across all attempts).
    logger.info(
        "CRAG: max retries (%d) exhausted — returning best candidates (score=%.3f)",
        max_retries, score,
    )
    return candidates, "best_effort"
