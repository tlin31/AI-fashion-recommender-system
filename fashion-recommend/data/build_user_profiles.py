#!/usr/bin/env python3
"""Derive user profiles by aggregating the item labels of a user's TRAIN items.

This is the Day 5 ablation's zero-cost arm. It answers the question that makes
the LLM arm's cost interpretable: an LLM trait extractor is only worth its bill
if it beats what a single SQL join already gives you for free.

    # the simulated cold-start cohort -- profiles from their WITHHELD train events
    python3 data/build_user_profiles.py --users eval/cold_sim_users.json --push

    # the reference population -- everyone Gorse knows, minus the cold cohort
    python3 data/build_user_profiles.py --reference --push

    # inspect without writing
    python3 data/build_user_profiles.py --users eval/cold_sim_users.json --dry-run

── Three properties this file is responsible for ────────────────────────────

**Only the train split is read.** A profile built from test events is the answer
leaking into the question: the arm would be scored on items it was told about.
The WHERE clause below is the single most important line here, exactly as it is
in build_interactions.push_gorse.

**Only users Gorse already has.** `POST /api/users` is an upsert and creates
unknown ids, so pushing a derived population wholesale silently enlarges the
evaluation cohort -- measured once at 233,368 phantom users with no feedback,
all of which would have landed in the cold cohort and moved every denominator.
See --push, which intersects first and reports what it dropped.

**The label vocabulary is the item side's, unchanged.** Profiles reuse
seed_gorse.build_item_labels rather than restating the schema, because the two
seeders drifted apart once already and the eval silently scored two prefixes
that existed on zero items.

── Why the full vocabulary, not the shared one ──────────────────────────────

An earlier design restricted these profiles to `style:` and `color:` -- the only
prefixes the item side and the LLM extractor have in common -- so the arms would
differ in derivation method rather than vocabulary width. Measured, that is not
viable: with 16 distinct label strings the median user lands in an equivalence
class of 506 people and only 18% get any user-to-user neighbour at all. A
controlled comparison in which both arms emit noise is not controlled, it is
empty. The full vocabulary (179 strings) puts the median user in a class of 9
and 70% get neighbours.

`--shared-vocab` keeps the narrow version available, because "what is vocabulary
width alone worth?" is a real question -- it is just a separate measurement, not
the main comparison.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse")
GORSE_URL = os.environ.get("GORSE_URL", "http://localhost:8088")

# Matches traits.GorseSync's cap. It exists so that label COUNT is not a
# confound between ablation arms -- without it, an arm could win merely by
# emitting more labels per user.
MAX_PER_PREFIX = 5
SHARED_PREFIXES = ("style:", "color:")


def _gorse_user_ids() -> set[str]:
    """Users Gorse actually holds.

    Read through `docker exec` because the Docker Postgres deliberately does not
    publish 5432 -- the host has its own on that port, and confusing the two is
    a documented trap in this project.
    """
    r = subprocess.run(
        ["docker", "exec", "fashion-postgres", "psql", "-U", "gorse", "-d",
         "gorse", "-tA", "-c", "select user_id from users"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"could not read Gorse users: {r.stderr.strip()}")
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def build_profiles(target: set[str] | None, shared_vocab: bool) -> dict[str, list[str]]:
    """Aggregate each user's TRAIN items' feature labels into a user profile."""
    import psycopg2
    import psycopg2.extras
    from seed_gorse import build_item_labels, flatten_labels

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT product_id, name, brand, category, price, "
                        "price_range, avg_rating, description FROM reco_products")
            features: dict[str, list[str]] = {}
            for row in cur.fetchall():
                f = [l for _, l, on_sim
                     in flatten_labels(build_item_labels(dict(row))[0]) if on_sim]
                if shared_vocab:
                    f = [x for x in f if x.startswith(SHARED_PREFIXES)]
                features[row["product_id"]] = f

            # split = 'train' is the whole point. A profile from test events is
            # the label leaking into the evaluation.
            cur.execute("SELECT user_id, product_id FROM reco_interactions "
                        "WHERE split = 'train'")
            edges = cur.fetchall()
    finally:
        conn.close()

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for e in edges:
        if target is not None and e["user_id"] not in target:
            continue
        for f in features.get(e["product_id"], ()):
            counts[e["user_id"]][f] += 1

    profiles: dict[str, list[str]] = {}
    for uid, c in counts.items():
        by_prefix: dict[str, list] = collections.defaultdict(list)
        for label, n in c.items():
            by_prefix[label.split(":", 1)[0]].append((n, label))
        labels = []
        for prefix in sorted(by_prefix):
            # frequency desc, then name -- deterministic. Go's map iteration is
            # randomised, and the trait sync used to write a differently ordered
            # array on every sync of unchanged data.
            ranked = sorted(by_prefix[prefix], key=lambda t: (-t[0], t[1]))
            labels.extend(l for _, l in ranked[:MAX_PER_PREFIX])
        if labels:
            profiles[uid] = labels
    return profiles


def report(profiles: dict[str, list[str]], target: set[str] | None) -> None:
    n = len(profiles)
    if not n:
        print("no profiles derived")
        return
    sizes = [len(v) for v in profiles.values()]
    vocab = collections.Counter(l for v in profiles.values() for l in v)
    sets = [frozenset(v) for v in profiles.values()]
    classes = collections.Counter(sets)
    print(f"  users with a profile : {n:,}"
          + (f" of {len(target):,} requested" if target else ""))
    if target:
        missing = len(target) - n
        print(f"  users with none      : {missing:,}"
              + ("  (no labelled train item)" if missing else ""))
    print(f"  labels per user      : median {statistics.median(sizes):.0f}, "
          f"mean {statistics.mean(sizes):.2f}")
    print(f"  distinct label strings: {len(vocab)}")
    print(f"  equivalence classes  : {len(classes):,}   "
          f"median user's class: {statistics.median([classes[s] for s in sets]):,.0f}")


def push(profiles: dict[str, list[str]]) -> None:
    import httpx
    gorse = _gorse_user_ids()
    keep = {u: v for u, v in profiles.items() if u in gorse}
    dropped = len(profiles) - len(keep)
    print(f"\nintersecting with Gorse: {len(keep):,} kept, {dropped:,} dropped")
    if dropped:
        print(f"  those {dropped:,} would have been CREATED by the upsert, "
              f"enlarging the cohort with feedback-less users")
    if not keep:
        raise SystemExit("nothing to push after intersection")

    payload = [{"UserId": u, "Labels": v} for u, v in sorted(keep.items())]
    t0 = time.time()
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(payload), 2000):
            for attempt in range(1, 5):
                try:
                    c.post(f"{GORSE_URL}/api/users",
                           json=payload[i:i + 2000]).raise_for_status()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(attempt * 2)
    print(f"pushed {len(payload):,} profiles in {time.time() - t0:.0f}s")
    print("  verify in the store, not from RowAffected:")
    print('  docker exec fashion-postgres psql -U gorse -d gorse -tA -c '
          '"select count(*) from users where labels::text not in (\'[]\',\'null\')"')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--users", metavar="FILE",
                   help="JSON list, {\"users\": [...]}, or one id per line")
    g.add_argument("--reference", action="store_true",
                   help="every Gorse user EXCEPT those in --exclude (default: "
                        "eval/cold_sim_users.json). This is the population the "
                        "cold users are matched against; its profiles must stay "
                        "IDENTICAL across arms or the comparison moves its own "
                        "baseline.")
    p.add_argument("--exclude", metavar="FILE",
                   default=str(Path(__file__).parent.parent / "eval" / "cold_sim_users.json"),
                   help="with --reference: ids to leave out")
    p.add_argument("--shared-vocab", action="store_true",
                   help="restrict to style:/color: -- the prefixes the LLM arm "
                        "can also emit. A separate measurement of what "
                        "vocabulary width alone is worth, NOT the main arm; at "
                        "16 label strings only 18%% of users get a neighbour.")
    p.add_argument("--out", type=Path, default=None, help="write profiles as JSON")
    p.add_argument("--push", action="store_true", help="load into Gorse")
    p.add_argument("--dry-run", action="store_true", help="report only")
    args = p.parse_args()

    from build_interactions import _load_excluded_users

    if args.reference:
        excluded = _load_excluded_users(args.exclude)
        target = _gorse_user_ids() - excluded
        print(f"reference population: {len(target):,} users "
              f"(all Gorse users minus {len(excluded):,} excluded)")
    else:
        target = _load_excluded_users(args.users)
        print(f"target: {len(target):,} users from {args.users}")

    profiles = build_profiles(target, args.shared_vocab)
    print(f"\nvocabulary: {'style/color only (shared)' if args.shared_vocab else 'full Labels.f'}")
    report(profiles, target)

    if args.out:
        args.out.write_text(json.dumps(
            {"users": [{"UserId": u, "Labels": v} for u, v in sorted(profiles.items())]},
            indent=1))
        print(f"\nwrote {args.out}")

    if args.dry_run or not args.push:
        print("\n(no --push: nothing written to Gorse)")
        return
    push(profiles)


if __name__ == "__main__":
    main()
