# Tests for pipeline/crag.py.
# No external services required — hybrid_search, _rewrite_query, and
# _fetch_trending_fallback are all mocked.
#
# Cases:
#   _grade()                   — mean milvus_score, empty input
#   High score                 → "synthesize" path, no rewrite called
#   Low score                  → "best_effort" path immediately (returns original candidates)
#   Empty candidates           → "fallback" path immediately (trending — nothing else to return)
#   Medium score, retry good   → "retry" path after one rewrite
#   Medium score, retry bad    → "best_effort" after rewrite score drops (keeps pre-rewrite candidates)
#   Time budget exceeded       → "best_effort" before first retry attempt (returns best candidates so far)

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pipeline.crag import _grade, run_crag

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(milvus_score: float, n: int = 3) -> list[dict]:
    """Return n identical chunks with a fixed milvus_score."""
    return [
        {
            "chunk_id":    f"chunk_{i}",
            "product_id":  f"product_{i:03d}",
            "text":        f"Some product text {i}",
            "score":       0.01,
            "milvus_score": milvus_score,
            "metadata":    {"chunk_type": "description", "category": "tops"},
        }
        for i in range(n)
    ]


_DUMMY_EMBEDDING = [0.0] * 1536
_DUMMY_BM25      = object()   # opaque — never inspected in these tests
_DUMMY_MILVUS    = object()


# ---------------------------------------------------------------------------
# _grade unit tests
# ---------------------------------------------------------------------------

def test_grade_returns_mean_milvus_score():
    chunks = _make_chunks(0.0)
    chunks[0]["milvus_score"] = 0.8
    chunks[1]["milvus_score"] = 0.6
    chunks[2]["milvus_score"] = 0.4
    assert abs(_grade(chunks) - 0.6) < 1e-9


def test_grade_empty_candidates_returns_zero():
    assert _grade([]) == 0.0


def test_grade_missing_milvus_score_treated_as_zero():
    chunks = [{"chunk_id": "x", "product_id": "p", "text": "t", "score": 0.0}]
    assert _grade(chunks) == 0.0


# ---------------------------------------------------------------------------
# run_crag — happy paths
# ---------------------------------------------------------------------------

async def test_high_score_returns_synthesize_path():
    """Candidates above HIGH_THRESHOLD — pass through with no rewrite."""
    chunks = _make_chunks(milvus_score=0.85)

    with patch("pipeline.crag._rewrite_query", new_callable=AsyncMock) as mock_rewrite, \
         patch("pipeline.crag._fetch_trending_fallback", new_callable=AsyncMock) as mock_fallback:

        result_chunks, path = await run_crag(
            "blue dress", chunks, _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
        )

    assert path == "synthesize"
    assert result_chunks is chunks
    mock_rewrite.assert_not_called()
    mock_fallback.assert_not_called()


async def test_low_score_returns_best_effort_path():
    """Candidates below LOW_THRESHOLD — return original candidates as best_effort, no rewrite."""
    chunks = _make_chunks(milvus_score=0.30)

    with patch("pipeline.crag._rewrite_query", new_callable=AsyncMock) as mock_rewrite, \
         patch("pipeline.crag._fetch_trending_fallback", new_callable=AsyncMock) as mock_fallback:

        result_chunks, path = await run_crag(
            "blue dress", chunks, _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
        )

    assert path == "best_effort"
    assert result_chunks is chunks          # original candidates returned, not trending
    mock_rewrite.assert_not_called()
    mock_fallback.assert_not_called()       # trending NOT fetched


async def test_empty_candidates_returns_fallback_immediately():
    """Empty candidate list from retrieval failure — skip grading, go straight to fallback."""
    fallback_chunks = _make_chunks(milvus_score=0.0)

    with patch("pipeline.crag._fetch_trending_fallback",
               new_callable=AsyncMock, return_value=fallback_chunks):
        result_chunks, path = await run_crag(
            "blue dress", [], _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
        )

    assert path == "fallback"
    assert result_chunks is fallback_chunks


# ---------------------------------------------------------------------------
# run_crag — retry paths
# ---------------------------------------------------------------------------

async def test_medium_score_retry_returns_retry_path():
    """Initial score is borderline; rewritten query retrieves high-quality chunks."""
    initial_chunks = _make_chunks(milvus_score=0.60)  # between 0.45 and 0.75
    good_chunks    = _make_chunks(milvus_score=0.80)  # above HIGH_THRESHOLD

    with patch("pipeline.crag._rewrite_query",
               new_callable=AsyncMock, return_value="rewritten query"), \
         patch("pipeline.crag.hybrid_search",
               new_callable=AsyncMock, return_value=good_chunks), \
         patch("pipeline.crag._fetch_trending_fallback", new_callable=AsyncMock) as mock_fallback:

        result_chunks, path = await run_crag(
            "blue dress", initial_chunks, _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
        )

    assert path == "retry"
    assert result_chunks is good_chunks
    mock_fallback.assert_not_called()


async def test_medium_score_retry_bad_result_returns_best_effort():
    """Initial score borderline; rewrite produces worse results — return pre-rewrite candidates."""
    initial_chunks = _make_chunks(milvus_score=0.60)
    bad_chunks     = _make_chunks(milvus_score=0.20)  # below LOW_THRESHOLD

    with patch("pipeline.crag._rewrite_query",
               new_callable=AsyncMock, return_value="rewritten query"), \
         patch("pipeline.crag.hybrid_search",
               new_callable=AsyncMock, return_value=bad_chunks), \
         patch("pipeline.crag._fetch_trending_fallback", new_callable=AsyncMock) as mock_fallback:

        result_chunks, path = await run_crag(
            "blue dress", initial_chunks, _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
        )

    assert path == "best_effort"
    assert result_chunks is initial_chunks  # pre-rewrite candidates kept
    mock_fallback.assert_not_called()       # trending NOT fetched


# ---------------------------------------------------------------------------
# run_crag — time budget
# ---------------------------------------------------------------------------

async def test_time_budget_exceeded_returns_best_effort():
    """If the time budget is already exhausted before the first retry, return best candidates so far."""
    initial_chunks = _make_chunks(milvus_score=0.60)  # borderline — would retry

    # Simulate start=0.0, then elapsed check returns 100s — way over any budget.
    with patch("pipeline.crag.time") as mock_time, \
         patch("pipeline.crag._fetch_trending_fallback", new_callable=AsyncMock) as mock_fallback, \
         patch("pipeline.crag._rewrite_query", new_callable=AsyncMock) as mock_rewrite:

        mock_time.monotonic.side_effect = [0.0, 100.0]

        result_chunks, path = await run_crag(
            "blue dress", initial_chunks, _DUMMY_EMBEDDING,
            bm25_index=_DUMMY_BM25, milvus_client=_DUMMY_MILVUS,
            collection_name="fashion_rag",
            time_budget=3.5,
        )

    assert path == "best_effort"
    assert result_chunks is initial_chunks  # best candidates so far, not trending
    mock_rewrite.assert_not_called()        # budget expired before rewrite could run
    mock_fallback.assert_not_called()       # trending NOT fetched
