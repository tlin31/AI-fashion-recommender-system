# Unit tests for the interaction-graph maths in build_interactions.py.
#
#   pytest fashion-recommend/data/test_interactions.py -v
#
# Pure — no raw dump, no Postgres, no cache.
#
# These exist because the k-core numbers this script reports contradict the
# figures recorded in the project plan, and "my code disagrees with the notes"
# is only a useful statement if the code is demonstrably right on cases whose
# answers can be worked out by hand.

import numpy as np
import pytest

from build_interactions import dedupe_edges, iterative_core


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_dedupe_keeps_one_edge_per_user_item_pair():
    u = np.array([0, 0, 0, 1, 1])
    i = np.array([5, 5, 6, 5, 5])
    keep = dedupe_edges(u, i)
    pairs = sorted(zip(u[keep].tolist(), i[keep].tolist()))
    assert pairs == [(0, 5), (0, 6), (1, 5)]


def test_dedupe_is_a_noop_when_all_edges_are_distinct():
    u = np.array([0, 1, 2])
    i = np.array([0, 1, 2])
    assert len(dedupe_edges(u, i)) == 3


def test_dedupe_returns_ascending_indices():
    """Callers index parallel arrays (rating, verified, ts) with this, so the
    order must stay aligned with the original event order."""
    u = np.array([2, 0, 1, 0, 2])
    i = np.array([9, 7, 8, 7, 9])
    keep = dedupe_edges(u, i)
    assert list(keep) == sorted(keep)


# ── k-core ────────────────────────────────────────────────────────────────────

def test_core_finds_a_clean_block_and_discards_the_fringe():
    """Users 0-2 x items 0-2 is a complete 3x3 block = a true 3-core.

    User 3 touches item 0 once and user 0 touches item 3 once; both must be
    stripped, and stripping them must not disturb the block.
    """
    u = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 0])
    i = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3])
    got = iterative_core(u, i, 3)
    assert (got["events"], got["users"], got["items"]) == (9, 3, 3)
    assert got["density"] == 1.0


def test_core_is_empty_when_k_exceeds_every_degree():
    u = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    i = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    got = iterative_core(u, i, 4)
    assert (got["events"], got["users"], got["items"]) == (0, 0, 0)
    assert got["density"] == 0.0


def test_core_cascades_instead_of_filtering_once():
    """The property that separates a k-core from a one-pass degree filter.

    Users 0,1,2 each have 2 items; user 3 has 1. A single pass would drop only
    user 3. Dropping user 3 leaves item 0 with degree 3, so nothing else
    changes here — but the block survives intact, which is the fixed point.
    """
    u = np.array([0, 0, 1, 1, 2, 2, 3])
    i = np.array([0, 1, 0, 1, 0, 1, 0])
    got = iterative_core(u, i, 2)
    assert (got["events"], got["users"], got["items"]) == (6, 3, 2)


def test_core_cascade_collapses_a_chain_completely():
    """A chain has every user at degree <= 2 but no item shared widely enough.

    One-pass on users >= 2 would keep users 0,1,2. The true 2-core is empty
    because each removal starves the next node in the chain.
    """
    u = np.array([0, 0, 1, 1, 2, 2])
    i = np.array([0, 1, 1, 2, 2, 3])
    one_pass_users = int((np.bincount(u) >= 2).sum())
    assert one_pass_users == 3

    got = iterative_core(u, i, 2)
    assert got["users"] == 0, (
        "a chain has no 2-core; if this passes with users>0 the loop is "
        "stopping before the fixed point"
    )


def test_core_is_monotonically_non_increasing_in_k():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 60, 4000)
    i = rng.integers(0, 40, 4000)
    keep = dedupe_edges(u, i)
    u, i = u[keep], i[keep]

    sizes = [iterative_core(u, i, k)["events"] for k in (2, 3, 5, 10, 20)]
    assert sizes == sorted(sizes, reverse=True)


def test_core_never_exceeds_the_one_pass_filter():
    """The k-core is a subset of the one-pass result, never larger.

    This is the invariant that the plan's recorded figures violate: they report
    a 5-core larger than what iterating to a fixed point can yield.
    """
    rng = np.random.default_rng(7)
    u = rng.integers(0, 200, 3000)
    i = rng.integers(0, 150, 3000)
    keep = dedupe_edges(u, i)
    u, i = u[keep], i[keep]

    for k in (2, 3, 5):
        core_users = iterative_core(u, i, k)["users"]
        one_pass_users = int((np.bincount(u) >= k).sum())
        assert core_users <= one_pass_users


@pytest.mark.parametrize("k", [2, 3, 5])
def test_core_result_actually_satisfies_the_degree_condition(k):
    """Verify the output directly rather than trusting the loop.

    Every surviving user and item must genuinely have degree >= k within the
    surviving subgraph.
    """
    rng = np.random.default_rng(3)
    u = rng.integers(0, 80, 2000)
    i = rng.integers(0, 50, 2000)
    keep = dedupe_edges(u, i)
    u, i = u[keep], i[keep]

    # Re-derive the surviving edge set the same way the function does, then
    # check the invariant on it.
    n_u, n_i = u.max() + 1, i.max() + 1
    alive = np.ones(len(u), dtype=bool)
    for _ in range(1000):
        ud = np.bincount(u[alive], minlength=n_u)
        idg = np.bincount(i[alive], minlength=n_i)
        nxt = alive & (ud[u] >= k) & (idg[i] >= k)
        if nxt.sum() == alive.sum():
            break
        alive = nxt

    if alive.any():
        ud = np.bincount(u[alive], minlength=n_u)
        idg = np.bincount(i[alive], minlength=n_i)
        assert ud[np.unique(u[alive])].min() >= k
        assert idg[np.unique(i[alive])].min() >= k

    assert iterative_core(u, i, k)["events"] == int(alive.sum())
