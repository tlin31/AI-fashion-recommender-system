"""Unit tests for eval/baselines.py. No database.

The properties pinned here are the ones whose absence would not raise: a
baseline that quietly depends on cohort ordering, or on the test split, still
produces plausible numbers.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from baselines import (MostPopular, RandomBaseline,  # noqa: E402
                       TrainSignals, build_baselines)


def signals() -> TrainSignals:
    return TrainSignals(
        popularity=Counter({"hot": 100, "warm": 50, "mild": 50, "cold": 1}),
        exclude={"u1": {"hot"}, "u2": set()},
        catalogue=["hot", "warm", "mild", "cold", "unseen1", "unseen2"],
        n_positive_interactions=201,
    )


# ── MostPopular ──────────────────────────────────────────────────────────────

def test_most_popular_orders_by_count():
    assert MostPopular(signals()).recommend("u2", set(), k=4) == [
        "hot", "mild", "warm", "cold"]   # mild before warm: tie broken by id


def test_most_popular_breaks_ties_deterministically():
    """Two runs must agree. Without an explicit tie-break the order would come
    from dict iteration and the arm would not be reproducible -- silently, since
    aggregate metrics would still look stable."""
    a = MostPopular(signals()).recommend("u2", set(), k=4)
    b = MostPopular(signals()).recommend("u2", set(), k=4)
    assert a == b


def test_most_popular_applies_the_exclude_set():
    out = MostPopular(signals()).recommend("u1", {"hot"}, k=3)
    assert "hot" not in out
    assert len(out) == 3


def test_most_popular_never_returns_items_without_train_signal():
    """Popularity is built from train interactions, so an item nobody touched
    cannot appear -- it has no count to rank on."""
    out = MostPopular(signals()).recommend("u2", set(), k=6)
    assert "unseen1" not in out and "unseen2" not in out


def test_most_popular_returns_short_list_rather_than_padding():
    assert len(MostPopular(signals()).recommend("u2", set(), k=99)) == 4


# ── Random ───────────────────────────────────────────────────────────────────

def test_random_is_reproducible_for_a_user():
    r1 = RandomBaseline(signals(), seed=7).recommend("u1", set(), k=4)
    r2 = RandomBaseline(signals(), seed=7).recommend("u1", set(), k=4)
    assert r1 == r2


def test_random_differs_between_users():
    r = RandomBaseline(signals(), seed=7)
    assert r.recommend("u1", set(), k=4) != r.recommend("zzz", set(), k=4)


def test_random_does_not_depend_on_call_order():
    """Seeded per user, not globally. A global RNG would make the arm depend on
    how many users were scored first -- a dependency that turns a baseline into
    a second treatment."""
    r = RandomBaseline(signals(), seed=7)
    first_alone = RandomBaseline(signals(), seed=7).recommend("u2", set(), k=3)
    r.recommend("u1", set(), k=3)          # burn a call
    assert r.recommend("u2", set(), k=3) == first_alone


def test_random_respects_exclude_and_has_no_duplicates():
    out = RandomBaseline(signals(), seed=7).recommend("u1", {"hot", "warm"}, k=4)
    assert "hot" not in out and "warm" not in out
    assert len(out) == len(set(out))


def test_random_samples_items_with_no_train_signal():
    """A floor should be able to draw anything in the catalogue. Restricting to
    items Gorse could recommend would make it a weak content baseline instead."""
    draws = set()
    for u in range(60):
        draws.update(RandomBaseline(signals(), seed=1).recommend(f"u{u}", set(), k=3))
    assert draws & {"unseen1", "unseen2"}


def test_random_terminates_when_catalogue_is_exhausted():
    small = TrainSignals(catalogue=["a", "b"], popularity=Counter(),
                         exclude={}, n_positive_interactions=0)
    assert sorted(RandomBaseline(small, seed=3).recommend("u", set(), k=10)) == ["a", "b"]


# ── Wiring ───────────────────────────────────────────────────────────────────

def test_build_baselines_returns_both_named_arms():
    assert {b.name for b in build_baselines(signals())} == {"Random", "MostPopular"}


def test_summary_reports_signal_coverage():
    s = signals().summary()
    assert "6 items" in s and "4 with positive train signal" in s
