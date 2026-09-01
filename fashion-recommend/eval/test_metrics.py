"""Unit tests for eval/metrics.py. No database, no network.

Several of these pin behaviour that is easy to get subtly wrong in ways that do
not raise -- the failure mode this project keeps meeting.
"""

import math
import sys
from pathlib import Path

import pytest

import metrics  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from metrics import (catalog_coverage, gini, hit_rate_at_k,  # noqa: E402
                     intra_list_diversity, mrr, ndcg_at_k, novelty,
                     recall_at_k, recall_ceiling)


# ── NDCG ─────────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=10) == pytest.approx(1.0)


def test_ndcg_ideal_accounts_for_short_test_sets():
    """A user with one held-out item must reach 1.0 by ranking it first.

    If the ideal DCG were computed over k slots instead of min(k, |relevant|),
    this would return ~0.35 and every user with a short test set would be
    silently penalised. On this corpus that is nearly all of them.
    """
    assert ndcg_at_k(["a"] + [f"x{i}" for i in range(9)], {"a"}, k=10) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_positions():
    early = ndcg_at_k(["a", "x", "y"], {"a"}, k=10)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=10)
    assert early > late


def test_ndcg_empty_relevant_is_zero_not_error():
    assert ndcg_at_k(["a"], set(), k=10) == 0.0


# ── Recall ───────────────────────────────────────────────────────────────────

def test_recall_counts_unique_hits():
    assert recall_at_k(["a", "a", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)


def test_recall_ceiling_is_below_one_when_relevant_exceeds_k():
    """Documents the structural cap rather than hiding it."""
    assert recall_ceiling([20, 20], k=10) == pytest.approx(0.5)
    assert recall_ceiling([5, 5], k=10) == pytest.approx(1.0)


# ── HitRate / MRR ────────────────────────────────────────────────────────────

def test_hit_rate_is_binary():
    assert hit_rate_at_k(["x", "a"], {"a"}, k=10) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, k=10) == 0.0


def test_hit_rate_respects_k():
    assert hit_rate_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_mrr_uses_first_hit():
    assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)
    assert mrr(["x", "y"], {"a"}) == 0.0


# ── Coverage ─────────────────────────────────────────────────────────────────

def test_catalog_coverage_counts_distinct_items():
    assert catalog_coverage([["a", "b"], ["b", "c"]], catalogue_size=10) == pytest.approx(0.3)


def test_catalog_coverage_refuses_to_guess_its_denominator():
    """The denominator is a decision, not something to infer from the input."""
    with pytest.raises(ValueError):
        catalog_coverage([["a"]], catalogue_size=0)


# ── Gini ─────────────────────────────────────────────────────────────────────

def test_gini_uniform_exposure_is_zero():
    assert gini([["a", "b", "c"], ["a", "b", "c"]]) == pytest.approx(0.0, abs=1e-9)


def test_gini_rises_with_concentration():
    spread = gini([["a", "b", "c", "d"]])
    skewed = gini([["a", "a", "a", "b"]])
    assert skewed > spread


def test_gini_empty_is_zero_not_error():
    assert gini([]) == 0.0


# ── Novelty ──────────────────────────────────────────────────────────────────

def test_novelty_higher_for_rarer_items():
    pop = {"head": 1000, "tail": 1}
    head = novelty([["head"]], pop, n_interactions=10_000)
    tail = novelty([["tail"]], pop, n_interactions=10_000)
    assert tail > head
    assert head == pytest.approx(-math.log2(1000 / 10_000))


def test_novelty_treats_unseen_items_as_rare_not_missing():
    """Skipping unknown items would bias the mean toward popular ones."""
    assert novelty([["never-seen"]], {}, n_interactions=100) == pytest.approx(math.log2(100))


# ── ILD ──────────────────────────────────────────────────────────────────────

def test_ild_identical_items_have_zero_distance():
    labels = {"a": ["style:minimalist", "color:black"],
              "b": ["style:minimalist", "color:black"]}
    assert intra_list_diversity([["a", "b"]], labels)["ild"] == pytest.approx(0.0)


def test_ild_disjoint_items_have_max_distance():
    labels = {"a": ["style:minimalist"], "b": ["style:sporty"]}
    assert intra_list_diversity([["a", "b"]], labels)["ild"] == pytest.approx(1.0)


def test_ild_ignores_carrier_labels():
    """Carriers are near-unique per item; including them would floor every
    distance around 0.5 for reasons unrelated to diversity."""
    labels = {
        "a": ["style:minimalist", "item_name:Widget A", "brand:Acme",
              "price:19", "avg_rating:4.5", "price_range:mid"],
        "b": ["style:minimalist", "item_name:Widget B", "brand:Globex",
              "price:23", "avg_rating:4.5", "price_range:mid"],
    }
    # Only style: survives, and it matches -> distance 0
    assert intra_list_diversity([["a", "b"]], labels)["ild"] == pytest.approx(0.0)


def test_ild_skips_pairs_with_an_unlabelled_item():
    """An untagged item must not be scored as maximally diverse, or the metric
    rewards missing labels -- the failure it exists to detect."""
    labels = {"a": ["style:minimalist"], "b": ["item_name:Untagged"]}
    out = intra_list_diversity([["a", "b"]], labels)
    assert out["ild_pairs_scored"] == 0.0
    assert out["ild"] == 0.0
    assert out["ild_item_coverage"] == pytest.approx(0.5)


def test_ild_reports_its_own_validity_data():
    labels = {"a": ["style:minimalist", "color:black"],
              "b": ["style:sporty"],
              "c": []}
    out = intra_list_diversity([["a", "b", "c"]], labels)
    assert out["ild_item_coverage"] == pytest.approx(2 / 3)
    assert out["ild_pairs_scored"] == pytest.approx(1 / 3)   # only (a,b) scorable
    assert out["ild_labels_per_item"] == pytest.approx(1.5)  # 2 and 1


def test_ild_unknown_item_is_not_a_crash():
    out = intra_list_diversity([["ghost", "a"]], {"a": ["style:minimalist"]})
    assert out["ild_pairs_scored"] == 0.0


# ── Label schema: the map form is what item-to-item actually ranks on ────────

def test_feature_labels_reads_the_map_form():
    """`Labels.f` is exactly the branch column = "item.Labels.f" hands to Gorse.

    Reading it directly means ILD is computed over the recommender's own feature
    space instead of a prefix list kept in sync by hand.
    """
    got = metrics._feature_labels({
        "f": ["type:t-shirt", "cat:tops", "style:casual"],
        "brand": "zara",
        "price_range": "mid",
        "avg_rating": "4.5",
    })
    assert got == frozenset({"type:t-shirt", "cat:tops", "style:casual"})


def test_feature_labels_does_not_silently_score_dict_keys():
    """The failure this guards against returns 0.0, not an exception.

    Iterating a dict yields its KEYS -- "f", "brand", "price_range" -- and none
    of them start with a feature prefix, so a naive implementation reports ILD
    0.0 at full-looking coverage: a schema that never landed would read as a
    diversity collapse.
    """
    labels = {"f": ["style:casual", "color:black"], "brand": "zara"}
    assert "brand" not in metrics._feature_labels(labels)
    assert "f" not in metrics._feature_labels(labels)

    out = metrics.intra_list_diversity([["a", "b"]], {
        "a": labels,
        "b": {"f": ["style:formal", "color:white"], "brand": "cos"},
    })
    assert out["ild"] == 1.0, "disjoint feature sets are maximally distant"
    assert out["ild_item_coverage"] == 1.0
    assert out["ild_pairs_scored"] == 1.0


def test_feature_labels_still_reads_the_flat_schema():
    """So a pre-restructure run can be re-scored for the before/after."""
    got = metrics._feature_labels(
        ["style:casual", "brand:zara", "item_name:Plain Tee", "price:29"])
    assert got == frozenset({"style:casual"})


def test_empty_feature_branch_is_uncovered_not_zero_distance():
    """An item with no features must be skipped, never scored as maximally distant.

    Scoring it would make the metric reward missing labels -- the exact failure
    ILD is meant to detect. Adding type:/cat: took this population from 21.0% of
    the eval catalogue to zero, but the guard stays.
    """
    out = metrics.intra_list_diversity([["a", "b"]], {
        "a": {"f": []},
        "b": {"f": ["style:formal"]},
    })
    assert out["ild_pairs_scored"] == 0.0
    assert out["ild_item_coverage"] == 0.5
