# Tests for eval/metrics.py — Recall@K, NDCG@K, and the deduplication precondition.
#
# Cases:
#   Hand-computed NDCG/recall values   → formula correctness
#   Duplicate IDs in the top-k window  → ValueError (the 2026-08-03 baseline bug)
#   Metrics never exceed 1.0

from __future__ import annotations

import math

import pytest

from eval.metrics import ndcg_at_k, recall_at_k

# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def test_recall_all_relevant_retrieved():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_half_retrieved():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_ignores_hits_outside_k():
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_no_relevant_ids_returns_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], relevance, k=2) == pytest.approx(1.0)


def test_ndcg_inverted_ranking_below_one():
    relevance = {"a": 2, "b": 1}
    assert ndcg_at_k(["b", "a"], relevance, k=2) < 1.0


def test_ndcg_hand_computed_value():
    # One relevant item (rel=2) at rank 2. Gain = 2^2-1 = 3, discount = log2(3).
    # IDCG places it at rank 1: 3 / log2(2) = 3.
    relevance = {"a": 2}
    expected = (3 / math.log2(3)) / 3
    assert ndcg_at_k(["x", "a"], relevance, k=2) == pytest.approx(expected)


def test_ndcg_no_relevant_items_returns_zero():
    assert ndcg_at_k(["x", "y"], {}, k=2) == 0.0


def test_ndcg_never_exceeds_one():
    relevance = {"a": 2, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], relevance, k=3) <= 1.0


# ---------------------------------------------------------------------------
# Deduplication precondition
# ---------------------------------------------------------------------------

def test_recall_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate product IDs"):
        recall_at_k(["a", "b", "a"], {"a"}, k=3)


def test_ndcg_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate product IDs"):
        ndcg_at_k(["a", "b", "a"], {"a": 2}, k=3)


def test_error_names_the_duplicated_ids():
    with pytest.raises(ValueError, match=r"\['a'\]"):
        recall_at_k(["a", "a", "b"], {"a"}, k=3)


def test_duplicates_outside_k_window_are_allowed():
    """Only the scored window must be unique — the tail is never read."""
    assert recall_at_k(["a", "b", "a"], {"a"}, k=2) == 1.0
