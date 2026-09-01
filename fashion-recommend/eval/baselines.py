"""Reference arms for the recommender evaluation: Random and MostPopular.

**These are computed here, from the full interaction table, and never read back
out of Gorse.** That is the whole point of the file.

The evaluation cohort has to be sampled to fit a memory budget (see
`build_interactions.py --cohort-report`). Gorse's own popularity list is derived
from whatever feedback Gorse currently holds, so a popularity baseline taken
from it would move with the sampling — 'popular' would mean 'popular among the
50K users we happened to push'. A baseline whose value depends on the
independent variable is not a baseline; it is a second treatment.

So popularity is counted from `reco_interactions` on the host, which is the
source of truth and does not change when the pushed cohort does. The same
reasoning applies to novelty's popularity denominator and to catalog coverage's
catalogue size, both of which take their denominator from here.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["TrainSignals", "load_train_signals", "MostPopular", "RandomBaseline",
           "POSITIVE_FEEDBACK_TYPES"]

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse")

# Mirrors config.toml's [recommend.data_source] positive_feedback_types.
#
# Popularity counts positive feedback only. Counting every interaction would let
# `dislike` — which this dataset carries as an explicit negative, 1-2 star
# reviews — push an item up the popularity ranking, so a widely disliked product
# would be recommended *because* it was disliked. `view` is excluded for a
# weaker reason: it is a read signal, and mixing intent strengths into one count
# makes the baseline harder to interpret than it needs to be.
POSITIVE_FEEDBACK_TYPES = ("purchase", "favorite", "add_to_cart")


@dataclass
class TrainSignals:
    """Everything the baselines and the population metrics need, from train only.

    Nothing here may be derived from the test split. Popularity computed over
    all splits would rank items by information the model is being tested on,
    which inflates MostPopular for free and is unrecoverable once reported.
    """

    popularity: Counter = field(default_factory=Counter)
    """item_id -> positive train interactions."""

    exclude: dict[str, set[str]] = field(default_factory=dict)
    """user_id -> items they already interacted with in train, any feedback type."""

    catalogue: list[str] = field(default_factory=list)
    """Every item in reco_products. The coverage denominator."""

    n_positive_interactions: int = 0
    """Total positive train interactions. Novelty's denominator."""

    def summary(self) -> str:
        with_signal = len(self.popularity)
        return (f"catalogue {len(self.catalogue):,} items "
                f"({with_signal:,} with positive train signal, "
                f"{with_signal / max(len(self.catalogue), 1):.1%}); "
                f"{self.n_positive_interactions:,} positive train interactions; "
                f"exclude sets for {len(self.exclude):,} users")


def load_train_signals(dsn: str | None = None) -> TrainSignals:
    """Read popularity, exclude sets and the catalogue from the host Postgres.

    Deliberately reads `reco_interactions` rather than asking Gorse. See the
    module docstring.
    """
    import psycopg2

    sig = TrainSignals()
    conn = psycopg2.connect(dsn or POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT product_id FROM reco_products")
            sig.catalogue = [r[0] for r in cur.fetchall()]

            # WHERE split = 'train' is the line that keeps this honest.
            cur.execute(
                """
                SELECT product_id, count(*)
                FROM reco_interactions
                WHERE split = 'train' AND feedback_type = ANY(%s)
                GROUP BY product_id
                """,
                (list(POSITIVE_FEEDBACK_TYPES),),
            )
            for item_id, n in cur.fetchall():
                sig.popularity[item_id] = n
            sig.n_positive_interactions = sum(sig.popularity.values())

            # Exclude sets use every feedback type, not just positive: an item
            # the user viewed or disliked is still one they have already seen,
            # and re-recommending it wastes a slot in every arm equally.
            cur.execute(
                "SELECT user_id, product_id FROM reco_interactions WHERE split = 'train'")
            for user_id, item_id in cur.fetchall():
                sig.exclude.setdefault(user_id, set()).add(item_id)
    finally:
        conn.close()
    return sig


class MostPopular:
    """Rank every user identically by train-split positive interaction count.

    Ties are broken by item id so two runs produce the same list. Without it the
    ordering would depend on dict iteration and the arm would not be
    reproducible — a silent problem, since the metrics would still look stable
    in aggregate while individual lists shifted.
    """

    name = "MostPopular"

    def __init__(self, signals: TrainSignals):
        self._ranked = [
            item for item, _ in sorted(
                signals.popularity.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def recommend(self, user_id: str, exclude: set[str], k: int) -> list[str]:
        out: list[str] = []
        for item in self._ranked:
            if item in exclude:
                continue
            out.append(item)
            if len(out) == k:
                break
        return out


class RandomBaseline:
    """Uniform sample from the catalogue, excluding what the user has seen.

    Seeded per user rather than globally, so the result depends only on
    (seed, user_id) and not on how many users were scored before this one or in
    what order. A globally seeded RNG would make the arm sensitive to cohort
    ordering, which is exactly the kind of dependency that makes a baseline
    stop being one.

    Sampling is over the whole catalogue, including items with no training
    signal. Restricting to items Gorse could actually recommend would make this
    a weak content baseline rather than a floor.
    """

    name = "Random"

    def __init__(self, signals: TrainSignals, seed: int = 20260831):
        self._catalogue = list(signals.catalogue)
        self._seed = seed

    def recommend(self, user_id: str, exclude: set[str], k: int) -> list[str]:
        rng = random.Random(f"{self._seed}:{user_id}")
        out: list[str] = []
        seen: set[str] = set()
        # Rejection sampling beats shuffling a 95K list per user by a wide
        # margin; the exclude set is tiny relative to the catalogue, so the
        # expected number of retries is negligible.
        attempts = 0
        limit = max(k * 50, 1000)
        while len(out) < k and attempts < limit:
            attempts += 1
            pick = rng.choice(self._catalogue)
            if pick in exclude or pick in seen:
                continue
            seen.add(pick)
            out.append(pick)
        return out


def build_baselines(signals: TrainSignals, seed: int = 20260831
                    ) -> Sequence["MostPopular | RandomBaseline"]:
    return (RandomBaseline(signals, seed=seed), MostPopular(signals))
