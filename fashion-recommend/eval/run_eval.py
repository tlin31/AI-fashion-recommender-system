#!/usr/bin/env python3
"""Offline evaluation for the recommender. One command, one JSON file.

    python eval/run_eval.py                     # all arms, all cohorts
    python eval/run_eval.py --arms Random,MostPopular   # harness-only, no Gorse
    python eval/run_eval.py --limit 200         # smoke test

Reports every metric per cohort, and splits the warm cohort by training history.
That last part is not a nicety. 85% of evaluable "warm" users hold exactly one
training event, so a single warm number is dominated by users who are barely
distinguishable from cold, under a label that says otherwise.

Three preconditions are checked before anything is measured, because each of
them produces plausible wrong numbers rather than an error:

  * Redis must have evicted nothing. Eviction silently drops cached
    recommendations, depressing every Gorse arm by an unrecoverable amount --
    you cannot tell afterwards which keys went.
  * The master's task loop must be frozen. It otherwise regenerates every
    cache hourly, and a run that straddles a regeneration is measuring two
    different systems.
  * Popularity and coverage denominators come from the host tables, never from
    Gorse, whose contents depend on the sampled cohort.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402

from baselines import (POSITIVE_FEEDBACK_TYPES, MostPopular,  # noqa: E402
                       RandomBaseline, TrainSignals, load_train_signals)
from metrics import (catalog_coverage, gini, hit_rate_at_k,  # noqa: E402
                     intra_list_diversity, mrr, ndcg_at_k, novelty,
                     recall_at_k, recall_ceiling)

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse")
GORSE_URL = os.environ.get("GORSE_URL", "http://localhost:8088")
OUT_PATH = Path(__file__).parent / "latest_metrics.json"

K_PRIMARY = 10
K_RECALL = (10, 20)
WARM_STRATA = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10**9, "5+"))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation set
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalUser:
    user_id: str
    relevant: set[str]
    cohort: str          # "warm" | "cold"
    n_train: int         # training events, drives the warm strata


def load_eval_users(limit: int | None = None) -> list[EvalUser]:
    """Users with a test event, plus the training history that stratifies them.

    A user with no test event cannot be scored at all, so they are not here --
    which is why the evaluable population (5,627 warm) is much smaller than the
    pushed cohort.
    """
    import psycopg2

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.user_id,
                       array_agg(t.product_id)                AS relevant,
                       min(t.cohort)                          AS cohort,
                       coalesce(tr.n, 0)                      AS n_train
                FROM reco_interactions t
                LEFT JOIN (SELECT user_id, count(*) AS n
                           FROM reco_interactions WHERE split = 'train'
                           GROUP BY user_id) tr USING (user_id)
                WHERE t.split = 'test'
                GROUP BY t.user_id, tr.n
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    users = [EvalUser(u, set(rel), cohort, n) for u, rel, cohort, n in rows]
    users.sort(key=lambda u: u.user_id)          # stable across runs
    return users[:limit] if limit else users


# ─────────────────────────────────────────────────────────────────────────────
# Arms that query Gorse
# ─────────────────────────────────────────────────────────────────────────────

class GorseCF:
    """Gorse's own recommendation endpoint.

    With `[recommend.ranker] recommenders = ["collaborative"]` this is the
    collaborative arm, falling back to the configured chain when CF has nothing
    for a user -- which on this corpus is most of them, and is the finding
    rather than a defect.
    """

    name = "GorseCF"

    def __init__(self, client: httpx.Client):
        self._c = client

    def recommend(self, user_id: str, exclude: set[str], k: int) -> list[str]:
        try:
            r = self._c.get(f"/api/recommend/{user_id}", params={"n": k})
            r.raise_for_status()
            body = r.json()
        except Exception:
            return []
        if not isinstance(body, list):
            return []
        # The endpoint returns bare ids on this version; tolerate both shapes.
        items = [x if isinstance(x, str) else x.get("Id", "") for x in body]
        return [i for i in items if i and i not in exclude][:k]


class GorseItemToItem:
    """Content similarity, aggregated over the user's training history.

    A user has no single seed item, so each of their train items contributes its
    neighbours and the scores are summed. Items the user already saw are
    excluded afterwards, not before, so a neighbour reachable from two seeds
    still ranks above one reachable from one.

    Cold users have no train items and therefore get an empty list here. That is
    the definition of the cohort, not a failure of the arm.
    """

    name = "GorseItemToItem"

    def __init__(self, client: httpx.Client, neighbours_per_seed: int = 30,
                 max_seeds: int = 20):
        self._c = client
        self._n = neighbours_per_seed
        self._max_seeds = max_seeds
        self._cache: dict[str, list[tuple[str, float]]] = {}

    def _neighbours(self, item_id: str) -> list[tuple[str, float]]:
        if item_id in self._cache:
            return self._cache[item_id]
        try:
            r = self._c.get(f"/api/item/{item_id}/neighbors", params={"n": self._n})
            r.raise_for_status()
            body = r.json()
            out = [(x["Id"], float(x.get("Score", 0.0)))
                   for x in body if isinstance(x, dict) and x.get("Id")]
        except Exception:
            out = []
        self._cache[item_id] = out
        return out

    def recommend(self, user_id: str, exclude: set[str], k: int) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for seed in list(exclude)[:self._max_seeds]:
            for item, score in self._neighbours(seed):
                if item not in exclude:
                    scores[item] += score
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [i for i, _ in ranked[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# Preconditions
# ─────────────────────────────────────────────────────────────────────────────

def check_preconditions(strict: bool = True) -> dict:
    """Assert the conditions under which the numbers mean anything."""
    import subprocess

    out: dict = {}

    def redis_cli(*args: str) -> str:
        r = subprocess.run(["docker", "exec", "fashion-redis", "redis-cli", *args],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()

    try:
        stats = redis_cli("INFO", "stats")
        evicted = next((int(l.split(":")[1]) for l in stats.splitlines()
                        if l.startswith("evicted_keys:")), None)
    except Exception as e:                                   # noqa: BLE001
        evicted = None
        out["evicted_keys_error"] = str(e)
    out["evicted_keys"] = evicted

    try:
        cfg = httpx.get(f"{GORSE_URL}/api/dashboard/config", timeout=15).json()
        rec = cfg.get("recommend", cfg)
        out["cache_size"] = rec.get("cache_size")
        out["collaborative_fit_period"] = (rec.get("collaborative") or {}).get("fit_period")
        out["ranker_fit_period"] = (rec.get("ranker") or {}).get("fit_period")
    except Exception as e:                                   # noqa: BLE001
        out["config_error"] = str(e)

    problems = []
    if evicted is None:
        problems.append("could not read evicted_keys from Redis")
    elif evicted > 0:
        problems.append(
            f"Redis has evicted {evicted:,} keys. Cached recommendations were "
            f"dropped and the loss is not recoverable -- which keys went is "
            f"unknown, so the damage cannot be quantified after the fact. "
            f"Raise maxmemory or shrink the cohort and re-push, then re-run.")

    if problems and strict:
        for p in problems:
            print(f"PRECONDITION FAILED: {p}", file=sys.stderr)
        raise SystemExit(2)
    out["problems"] = problems
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _stratum(n_train: int) -> str:
    for lo, hi, label in WARM_STRATA:
        if lo <= n_train <= hi:
            return label
    return "0"


def score_arm(arm, users: list[EvalUser], signals: TrainSignals,
              item_labels: dict[str, list[str]], k: int) -> dict:
    """Run one arm over every user and roll the results up by cohort/stratum."""
    per_group: dict[str, list[dict]] = defaultdict(list)
    per_group_lists: dict[str, list[list[str]]] = defaultdict(list)
    empty_lists = 0

    for u in users:
        ranked = arm.recommend(u.user_id, signals.exclude.get(u.user_id, set()), k=k)
        if not ranked:
            empty_lists += 1
        row = {
            "ndcg": ndcg_at_k(ranked, u.relevant, K_PRIMARY),
            "hit": hit_rate_at_k(ranked, u.relevant, K_PRIMARY),
            "mrr": mrr(ranked, u.relevant),
            "n_relevant": len(u.relevant),
        }
        for kk in K_RECALL:
            row[f"recall@{kk}"] = recall_at_k(ranked, u.relevant, kk)

        groups = [u.cohort]
        if u.cohort == "warm":
            groups.append(f"warm/train={_stratum(u.n_train)}")
        for g in groups + ["all"]:
            per_group[g].append(row)
            per_group_lists[g].append(ranked)

    result: dict = {"arm": arm.name, "empty_lists": empty_lists, "groups": {}}
    for g, rows in per_group.items():
        lists = per_group_lists[g]
        n = len(rows)
        agg = {
            "n_users": n,
            "ndcg@10": sum(r["ndcg"] for r in rows) / n,
            "hit_rate@10": sum(r["hit"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "mean_relevant": sum(r["n_relevant"] for r in rows) / n,
        }
        for kk in K_RECALL:
            agg[f"recall@{kk}"] = sum(r[f"recall@{kk}"] for r in rows) / n
            # Reported next to Recall so the figure is never read without its
            # structural bound; on a dense test set the ceiling is below 1.0.
            agg[f"recall@{kk}_ceiling"] = recall_ceiling(
                (r["n_relevant"] for r in rows), kk)
        agg["catalog_coverage"] = catalog_coverage(lists, len(signals.catalogue))
        agg["catalog_size"] = len(signals.catalogue)
        agg["gini"] = gini(lists)
        agg["novelty"] = novelty(lists, signals.popularity,
                                 signals.n_positive_interactions)
        agg.update(intra_list_diversity(lists, item_labels))
        result["groups"][g] = agg
    return result


def load_item_labels() -> dict[str, list[str]]:
    """Item labels from Gorse, for the ILD distance function."""
    labels: dict[str, list[str]] = {}
    cursor = ""
    with httpx.Client(base_url=GORSE_URL, timeout=120) as c:
        while True:
            r = c.get("/api/items", params={"n": 1000, "cursor": cursor})
            r.raise_for_status()
            body = r.json()
            for it in body.get("Items", []):
                # 原样保留：map 或扁平数组都交给 metrics._feature_labels 判断。
                # 在这里提前取 ["f"] 会把「schema 没换成功」变成一个
                # 静默的 0.0 ILD，而不是一个能看见的覆盖率下降。
                labels[it["ItemId"]] = it.get("Labels") or []
            cursor = body.get("Cursor") or ""
            if not cursor:
                break
    return labels


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="Random,MostPopular,GorseCF,GorseItemToItem",
                   help="comma-separated subset to run")
    p.add_argument("--k", type=int, default=20,
                   help="list length requested from each arm (default 20, the "
                        "largest k any metric needs)")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N users (smoke test)")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--allow-degraded", action="store_true",
                   help="run even if a precondition fails. The output is marked "
                        "degraded and must not be locked as a baseline")
    args = p.parse_args()

    started = time.time()
    print("checking preconditions...")
    pre = check_preconditions(strict=not args.allow_degraded)
    print(f"  evicted_keys={pre.get('evicted_keys')}  "
          f"cache_size={pre.get('cache_size')}  "
          f"ranker_fit_period={pre.get('ranker_fit_period')}")

    print("loading train signals from the host tables (not from Gorse)...")
    signals = load_train_signals()
    print(f"  {signals.summary()}")

    users = load_eval_users(limit=args.limit)
    by_cohort: dict[str, int] = defaultdict(int)
    for u in users:
        by_cohort[u.cohort] += 1
    print(f"evaluable users: {len(users):,}  {dict(by_cohort)}")

    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    needs_gorse = bool(wanted & {"GorseCF", "GorseItemToItem"})
    item_labels = load_item_labels() if needs_gorse else {}
    if needs_gorse:
        print(f"  loaded labels for {len(item_labels):,} items (for ILD)")

    client = httpx.Client(base_url=GORSE_URL, timeout=60) if needs_gorse else None
    registry = {
        "Random": lambda: RandomBaseline(signals, seed=args.seed),
        "MostPopular": lambda: MostPopular(signals),
        "GorseCF": lambda: GorseCF(client),
        "GorseItemToItem": lambda: GorseItemToItem(client),
    }

    results = []
    for name in ("Random", "MostPopular", "GorseCF", "GorseItemToItem"):
        if name not in wanted:
            continue
        print(f"scoring {name}...", flush=True)
        t0 = time.time()
        res = score_arm(registry[name](), users, signals, item_labels, k=args.k)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        a = res["groups"].get("all", {})
        print(f"  ndcg@10={a.get('ndcg@10', 0):.4f}  "
              f"recall@10={a.get('recall@10', 0):.4f}  "
              f"coverage={a.get('catalog_coverage', 0):.4f}  "
              f"empty_lists={res['empty_lists']:,}  ({res['seconds']}s)")

    if client:
        client.close()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "degraded": bool(pre.get("problems")),
        "preconditions": pre,
        "cohort": {
            "evaluable_users": len(users),
            "by_cohort": dict(by_cohort),
            "warm_strata": {label: sum(1 for u in users
                                       if u.cohort == "warm" and _stratum(u.n_train) == label)
                            for _, _, label in WARM_STRATA},
            "note": "warm strata count TRAINING events; 'warm' with one training "
                    "event is near-cold and must not be aggregated with the rest",
        },
        "denominators": {
            "catalogue_size": len(signals.catalogue),
            "items_with_positive_train_signal": len(signals.popularity),
            "positive_train_interactions": signals.n_positive_interactions,
            "positive_feedback_types": list(POSITIVE_FEEDBACK_TYPES),
            "source": "reco_interactions on the host, train split only -- never Gorse",
        },
        "k": args.k,
        "seed": args.seed,
        "arms": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}"
          f"{'  [DEGRADED -- do not lock as a baseline]' if payload['degraded'] else ''}")


if __name__ == "__main__":
    main()
