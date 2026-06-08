# Tests for pipeline/reranker.py.
# CrossEncoder model loading is mocked — no weights downloaded during tests.
#
# Cases:
#   Empty candidates           → returns []
#   Score ordering             → descending by adjusted score
#   top_k cap                  → never returns more than top_k results
#   Description boost          → description chunk outranks review at same raw score
#   Boost amount               → exactly _DESCRIPTION_BOOST added

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.reranker import _DESCRIPTION_BOOST, Reranker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_reranker(raw_scores: list[float]) -> Reranker:
    """Return a Reranker whose CrossEncoder.predict returns raw_scores."""
    with patch("pipeline.reranker.CrossEncoder"):
        r = Reranker()
    r._model.predict = MagicMock(return_value=raw_scores)
    return r


def _make_chunk(
    product_id: str,
    text: str = "some text",
    chunk_type: str = "description",
) -> dict:
    return {
        "chunk_id":    f"chunk_{product_id}",
        "product_id":  product_id,
        "text":        text,
        "score":       0.0,
        "milvus_score": 0.5,
        "metadata":    {"chunk_type": chunk_type, "category": "tops"},
    }


# ---------------------------------------------------------------------------
# Guard cases
# ---------------------------------------------------------------------------

def test_empty_candidates_returns_empty():
    r = _make_reranker([])
    assert r.rerank("blue dress", []) == []


def test_single_candidate_returned():
    r = _make_reranker([0.7])
    chunk = _make_chunk("p001")
    result = r.rerank("blue dress", [chunk], top_k=5)
    assert len(result) == 1
    assert result[0]["product_id"] == "p001"


# ---------------------------------------------------------------------------
# Ordering and top-k
# ---------------------------------------------------------------------------

def test_scores_sorted_descending():
    # Raw scores: p002 highest, p003 middle, p001 lowest.
    chunks = [_make_chunk("p001"), _make_chunk("p002"), _make_chunk("p003")]
    r = _make_reranker([0.3, 0.9, 0.6])
    result = r.rerank("blue dress", chunks, top_k=3)
    assert [c["product_id"] for c in result] == ["p002", "p003", "p001"]


def test_top_k_limits_results():
    chunks = [_make_chunk(f"p{i:03d}") for i in range(10)]
    r = _make_reranker([float(i) for i in range(10)])
    result = r.rerank("blue dress", chunks, top_k=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Business rule: description boost
# ---------------------------------------------------------------------------

def test_description_chunk_outranks_review_at_equal_raw_score():
    """When raw cross-encoder scores are equal, description ranks above review."""
    description = _make_chunk("p001", chunk_type="description")
    review      = _make_chunk("p002", chunk_type="review")
    # Same raw score for both — description boost should push p001 above p002.
    r = _make_reranker([0.5, 0.5])
    result = r.rerank("blue dress", [review, description], top_k=2)
    assert result[0]["product_id"] == "p001"


def test_description_boost_value_is_correct():
    """The adjusted score for a description chunk equals raw + _DESCRIPTION_BOOST."""
    chunk = _make_chunk("p001", chunk_type="description")
    r = _make_reranker([0.5])
    result = r.rerank("blue dress", [chunk])
    assert abs(result[0]["score"] - (0.5 + _DESCRIPTION_BOOST)) < 1e-6


def test_review_chunk_receives_no_boost():
    """Review chunks get no score adjustment."""
    chunk = _make_chunk("p001", chunk_type="review")
    r = _make_reranker([0.5])
    result = r.rerank("blue dress", [chunk])
    assert abs(result[0]["score"] - 0.5) < 1e-6
