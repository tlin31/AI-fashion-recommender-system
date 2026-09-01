"""Ranking and beyond-accuracy metrics for the recommender evaluation.

Pure functions over Python lists. No database, no network, no Gorse -- so this
file is testable in isolation and the numbers it produces can be traced to an
input rather than to the state of a running system.

Two conventions hold throughout and are load-bearing:

1.  **`relevant` is a set of item ids, not a graded relevance map.** The eval
    dataset is implicit feedback: an item is either in the user's held-out test
    events or it is not. Graded relevance would be inventing a signal the data
    does not carry.

2.  **Every population-level statistic takes its denominator as an argument.**
    Catalog coverage over "the catalogue" is meaningless unless the catalogue is
    named, and this project has two of them (5,000 for retrieval, 95,335 for
    evaluation) plus a sampled user cohort. Passing the denominator in makes it
    impossible to report a coverage figure without having decided what it is
    relative to.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ndcg_at_k", "recall_at_k", "hit_rate_at_k", "mrr",
    "catalog_coverage", "gini", "novelty", "intra_list_diversity",
    "FEATURE_LABEL_PREFIXES",
]


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Normalised discounted cumulative gain over binary relevance.

    The ideal DCG is computed over `min(k, len(relevant))` hits, so a user with
    two held-out items cannot be penalised for the eight slots that could never
    have been filled. Getting this wrong caps NDCG below 1.0 for every user with
    a short test set, which on this corpus is nearly all of them -- 85% of the
    evaluable cohort holds a single training event and correspondingly few test
    events.
    """
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2)
              for rank, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 2)
                for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Fraction of the user's relevant items that appear in the top k.

    Structurally capped at `min(k, |relevant|) / |relevant|`. When a cohort's
    mean |relevant| exceeds k, the ceiling is below 1.0 and the raw number
    understates performance -- report the ceiling alongside it rather than
    lowering the target to meet the result.
    """
    if not relevant or k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def recall_ceiling(relevant_sizes: Iterable[int], k: int = 10) -> float:
    """The highest Recall@k this cohort could reach with a perfect ranker.

    Exists so a Recall figure is never reported without the bound it is
    measured against.
    """
    sizes = [n for n in relevant_sizes if n > 0]
    if not sizes:
        return 0.0
    return sum(min(k, n) / n for n in sizes) / len(sizes)


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """1.0 if any relevant item is in the top k, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant item; 0.0 if none is ranked."""
    for rank, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Beyond-accuracy
# ─────────────────────────────────────────────────────────────────────────────

def catalog_coverage(all_recommendations: Iterable[Sequence[str]],
                     catalogue_size: int) -> float:
    """Share of the catalogue that appears anywhere in any recommendation list.

    `catalogue_size` is required rather than inferred. This project has two
    catalogues and a sampled user cohort, and a coverage number is only
    interpretable next to the denominator it used.
    """
    if catalogue_size <= 0:
        raise ValueError("catalogue_size must be positive; it is the denominator "
                         "and cannot be guessed from the recommendations")
    seen: set[str] = set()
    for rec in all_recommendations:
        seen.update(rec)
    return len(seen) / catalogue_size


def gini(all_recommendations: Iterable[Sequence[str]]) -> float:
    """Concentration of exposure across items. 0 = uniform, →1 = winner-take-all.

    Computed over items that were recommended at least once. Including the
    unrecommended tail as zero-exposure would make Gini a restatement of catalog
    coverage on this corpus, where the catalogue is 95,335 items and a sampled
    cohort touches a small fraction of it.
    """
    counts = Counter()
    for rec in all_recommendations:
        counts.update(rec)
    if not counts:
        return 0.0
    values = sorted(counts.values())
    n = len(values)
    cumulative = sum((2 * i - n - 1) * v for i, v in enumerate(values, start=1))
    total = sum(values)
    return cumulative / (n * total) if total else 0.0


def novelty(all_recommendations: Iterable[Sequence[str]],
            popularity: Mapping[str, int],
            n_interactions: int) -> float:
    """Mean self-information of recommended items: -log2(p(item)).

    Higher means the recommender surfaces less-popular items.

    `popularity` must be computed from the **full** interaction table, not from
    whatever the recommender happens to hold. The evaluation cohort is sampled
    to fit a memory budget, and a popularity distribution derived from the
    sampled system would move with the sampling -- making novelty a function of
    the independent variable rather than a property of the recommendations.

    Items absent from `popularity` are treated as maximally novel via a count of
    1 rather than skipped, since dropping them would bias the mean toward
    popular items.
    """
    if n_interactions <= 0:
        raise ValueError("n_interactions must be positive")
    total, n = 0.0, 0
    for rec in all_recommendations:
        for item in rec:
            p = max(popularity.get(item, 1), 1) / n_interactions
            total += -math.log2(p)
            n += 1
    return total / n if n else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Intra-list diversity
# ─────────────────────────────────────────────────────────────────────────────

# Distance is computed over *feature* labels only. The excluded prefixes are
# carriers -- per-item identifiers that ride along in the label array so the API
# can read them back -- not descriptions of what the item is like.
#
# Measured over 3,000 items: `item_name` has 2,186 distinct values and `brand`
# 1,492, so both enter the union of every pair and the intersection of none. Two
# byte-identical products would still score below 0.5 Jaccard similarity, which
# compresses the whole ILD range into roughly 0.5-0.9 for a reason that has
# nothing to do with diversity.
#
# `price_range` is excluded for the opposite reason: 4 distinct values with ~80%
# on "mid" make it a near-constant shared element, deflating every distance by
# about the same amount. `avg_rating` is excluded because two items rated 4.5 are
# not similar in any sense a user would recognise.
#
# Since the label restructure the item side draws exactly this distinction
# itself: features live under `Labels.f` and carriers do not. This list is the
# fallback for the OLD flat schema, and it is kept honest about what that schema
# actually contained -- `occasion:` and `material:` are listed because
# seed_gorse.py emits them, but the 95,335-product eval catalogue is built by
# build_interactions.py, which never did. Against that catalogue those two
# prefixes matched nothing, which is why style/color coverage alone decided the
# ILD denominator.
FEATURE_LABEL_PREFIXES = ("type:", "cat:", "style:", "color:",
                          "occasion:", "material:")


def _feature_labels(labels) -> frozenset[str]:
    """Extract an item's similarity features from either label schema.

    The map form is authoritative: `Labels.f` is exactly the branch
    `column = "item.Labels.f"` gives to tags item-to-item, so reading it here
    means ILD is computed over the same feature space the recommender ranks on
    rather than a prefix list maintained in parallel.

    Getting this wrong would not raise. `frozenset(l for l in some_dict ...)`
    iterates KEYS -- "f", "brand", "price_range" -- none of which match a
    prefix, so ILD would come back a clean 0.0 with full-looking coverage.
    """
    if isinstance(labels, Mapping):
        return frozenset(labels.get("f") or ())
    return frozenset(l for l in (labels or ())
                     if l.startswith(FEATURE_LABEL_PREFIXES))


def intra_list_diversity(all_recommendations: Iterable[Sequence[str]],
                         item_labels: Mapping[str, Sequence[str]]) -> dict:
    """Mean pairwise Jaccard distance within each list, plus its own validity data.

    Returns four numbers, and the last three are not decoration -- they decide
    whether the first one may be compared across arms:

        ild                 the metric
        ild_item_coverage   share of recommended items carrying >=1 feature label
        ild_pairs_scored    share of pairs that could be scored at all
        ild_labels_per_item mean feature labels among items that have any

    Excluding carrier labels (see FEATURE_LABEL_PREFIXES) fixes the distance
    function but sharpens a second problem: `style` covers 51% of the catalogue
    and `color` 29%, so roughly 40% of items end up with an empty feature set,
    for which Jaccard is undefined. Pairs where either side is empty are skipped
    rather than scored as maximally distant -- scoring them would make the metric
    reward missing labels, which is the exact failure it is meant to detect.

    **Comparison rule.** ILD is only comparable between arms whose
    `ild_item_coverage` is close. A popularity-biased arm recommends head items,
    which carry more metadata; a tail-heavy arm recommends poorly-labelled ones.
    Comparing their ILD without coverage compares labelling quality and yields a
    plausible, wrong conclusion -- that random recommendation is more diverse.
    """
    lists = [list(rec) for rec in all_recommendations]

    recommended: set[str] = set()
    for rec in lists:
        recommended.update(rec)

    features = {i: _feature_labels(item_labels.get(i, ())) for i in recommended}
    with_labels = [i for i, f in features.items() if f]

    total_pairs = scored_pairs = 0
    distance_sum = 0.0
    for rec in lists:
        for a in range(len(rec)):
            for b in range(a + 1, len(rec)):
                total_pairs += 1
                fa, fb = features.get(rec[a], frozenset()), features.get(rec[b], frozenset())
                if not fa or not fb:
                    continue
                union = len(fa | fb)
                distance_sum += 1.0 - (len(fa & fb) / union if union else 0.0)
                scored_pairs += 1

    return {
        "ild": distance_sum / scored_pairs if scored_pairs else 0.0,
        "ild_item_coverage": len(with_labels) / len(recommended) if recommended else 0.0,
        "ild_pairs_scored": scored_pairs / total_pairs if total_pairs else 0.0,
        "ild_labels_per_item": (sum(len(features[i]) for i in with_labels) / len(with_labels)
                                if with_labels else 0.0),
    }
