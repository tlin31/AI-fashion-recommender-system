# Build the real interaction dataset for recommender evaluation, from the
# McAuley 2023 Amazon Fashion review dump.
#
#   python data/build_interactions.py --stats
#   python data/build_interactions.py --stats --catalogue-size 5000
#
# WHY THIS EXISTS
# ---------------
# seed_gorse.py generates 5 demo users x 11 events = 55 feedback rows. Matrix
# factorisation on 5 users produces nothing meaningful, and every downstream
# metric is undefined rather than merely noisy: NDCG has no basis, catalog
# coverage has a numerator of at most 55, Gini over 5 users is noise, and a
# cold-start ablation has no warm control group to compare against.
#
# The real signal is already in the repo: 2.5M Amazon Fashion reviews whose
# parent_asin IS rag_products.product_id (rag-service/data/normalize.py:327),
# so the join is on a primary key and needs no fuzzy matching.
#
# WHAT --stats IS FOR
# -------------------
# The sparsity of this corpus is the single most important fact about the
# project, and it has to be reproducible from a command rather than quoted from
# a chat log. --stats writes no data; it only measures. Its output is the table
# that belongs at the TOP of the README, because it is what justifies the
# architecture: if ~85% of users appear exactly once, user-based CF is
# undefined for most of the population, and leaning on content signals and
# LLM-extracted traits is a measured decision rather than a preference.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_REVIEWS = (REPO_ROOT / "rag-service" / "amazon_data" / "raw" /
               "review_categories" / "Amazon_Fashion.jsonl")

# Extracted columns are cached so that repeated analysis (and the catalogue-size
# sweep, which needs many passes) does not re-parse 1 GB of JSON every time.
CACHE_PATH = Path(__file__).parent / "cache" / "amazon_fashion_events.parquet"

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse"
)

# k values reported in the degree table and the iterative core table.
DEGREE_KS = (2, 3, 5, 10, 20)
CORE_KS = (2, 3, 5, 10)


# ── Stage 1: extract the five fields we need from the raw dump ────────────────

def extract_events(limit: int | None = None) -> pa.Table:
    """Stream the JSONL dump and keep only the interaction-graph columns.

    The raw records carry `title`, `text` and `images`; `text` alone averages a
    few hundred bytes and dominates parse time. We still pay for full JSON
    parsing per line (a regex over 1 GB is faster but silently wrong on escaped
    quotes inside review text, which this corpus has plenty of), so the result
    is cached to Parquet and this function runs once.
    """
    if not RAW_REVIEWS.exists():
        sys.exit(
            f"Raw reviews not found at {RAW_REVIEWS}\n"
            f"This file is the McAuley 2023 Amazon Fashion dump (~1.0 GB) and is "
            f"not in git. Restore it before running --stats."
        )

    users: list[str] = []
    items: list[str] = []
    ratings: list[float] = []
    verified: list[bool] = []
    stamps: list[int] = []

    malformed = 0
    started = time.time()
    size_mb = RAW_REVIEWS.stat().st_size / 1e6
    print(f"Scanning {RAW_REVIEWS.name} ({size_mb:,.0f} MB)...")

    with open(RAW_REVIEWS, "rb") as fh:
        for n, line in enumerate(fh, 1):
            if limit is not None and n > limit:
                break
            try:
                rec = orjson.loads(line)
                users.append(rec["user_id"])
                # parent_asin, not asin: parent_asin is what rag_products keys
                # on, and it collapses size/colour variants of one product.
                items.append(rec["parent_asin"])
                ratings.append(rec["rating"])
                verified.append(rec["verified_purchase"])
                stamps.append(rec["timestamp"])
            except (orjson.JSONDecodeError, KeyError):
                malformed += 1
            if n % 500_000 == 0:
                rate = n / (time.time() - started)
                print(f"  {n:,} lines ({rate:,.0f}/s)")

    elapsed = time.time() - started
    print(f"  done: {len(users):,} events in {elapsed:.0f}s"
          + (f", {malformed:,} malformed lines skipped" if malformed else ""))

    return pa.table({
        "user_id":   pa.array(users).dictionary_encode(),
        "product_id": pa.array(items).dictionary_encode(),
        "rating":    pa.array(ratings, type=pa.float32()),
        "verified":  pa.array(verified, type=pa.bool_()),
        "ts_ms":     pa.array(stamps, type=pa.int64()),
    })


def load_events(refresh: bool = False, limit: int | None = None) -> pa.Table:
    if CACHE_PATH.exists() and not refresh and limit is None:
        print(f"Using cached events: {CACHE_PATH}")
        print("  (pass --refresh to re-parse the raw dump)")
        return pq.read_table(CACHE_PATH)

    table = extract_events(limit=limit)
    if limit is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, CACHE_PATH, compression="zstd")
        print(f"  cached -> {CACHE_PATH} "
              f"({CACHE_PATH.stat().st_size / 1e6:,.0f} MB)")
    return table


# ── Stage 2: graph statistics ─────────────────────────────────────────────────

def dedupe_edges(u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Index of the first occurrence of each distinct (user, item) pair.

    The interaction GRAPH is defined over distinct edges, and reco_interactions
    has PRIMARY KEY (user_id, product_id), so repeat reviews of the same
    parent_asin by one user must not be counted as extra degree. Reporting both
    raw and deduped counts keeps the difference visible instead of letting a
    dedup silently change every downstream number.
    """
    order = np.lexsort((i, u))
    su, si = u[order], i[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    np.not_equal(su[1:], su[:-1], out=first[1:])
    first[1:] |= si[1:] != si[:-1]
    return np.sort(order[first])


def iterative_core(u: np.ndarray, i: np.ndarray, k: int) -> dict:
    """Iterative bipartite k-core: every user AND every item has degree >= k.

    This is NOT the same as filtering users with < k events. Dropping a user
    removes edges, which can push items below k; dropping those items removes
    more edges, which can push more users below k. The process is repeated to a
    fixed point, and the gap between the one-pass filter and this fixed point is
    itself the measure of how sparse the corpus is.
    """
    n_u, n_i = u.max() + 1, i.max() + 1
    alive = np.ones(len(u), dtype=bool)

    for _ in range(1000):  # converges in a handful of rounds; bound is a guard
        u_deg = np.bincount(u[alive], minlength=n_u)
        i_deg = np.bincount(i[alive], minlength=n_i)
        keep = alive & (u_deg[u] >= k) & (i_deg[i] >= k)
        if keep.sum() == alive.sum():
            break
        alive = keep
        if not alive.any():
            break

    events = int(alive.sum())
    n_users = len(np.unique(u[alive])) if events else 0
    n_items = len(np.unique(i[alive])) if events else 0
    density = events / (n_users * n_items) if events else 0.0
    return {"k": k, "events": events, "users": n_users,
            "items": n_items, "density": density}


def degree_table(u: np.ndarray, n_users: int) -> list[dict]:
    counts = np.bincount(u, minlength=n_users)
    rows = []
    for k in DEGREE_KS:
        sel = counts >= k
        rows.append({
            "k": k,
            "users": int(sel.sum()),
            "pct": sel.sum() / n_users,
            "events": int(counts[sel].sum()),
        })
    return rows, counts


# ── Stage 3: reporting ────────────────────────────────────────────────────────

def rule(title: str = "") -> None:
    print(f"\n{'─' * 78}")
    if title:
        print(title)
        print("─" * 78)


def report_corpus(u: np.ndarray, i: np.ndarray, raw_events: int, label: str) -> None:
    n_users = int(u.max()) + 1
    n_items = int(i.max()) + 1
    events = len(u)

    rule(f"{label}: interaction graph")
    print("Degrees and cores below are computed on DISTINCT (user, item) edges,")
    print("not raw events — that is what the graph is, and what the")
    print("reco_interactions PRIMARY KEY (user_id, product_id) will store.")
    print()
    print(f"raw events         : {raw_events:,}")
    print(f"distinct (u,i)     : {events:,}"
          + (f"   ({raw_events - events:,} duplicate edges collapsed)"
             if raw_events != events else "   (no duplicates)"))
    print(f"unique users       : {n_users:,}")
    print(f"unique items       : {n_items:,}")
    print(f"events/user        : {events / n_users:.2f}")
    print(f"events/item        : {events / n_items:.2f}")

    rows, counts = degree_table(u, n_users)
    rule(f"{label}: user degree distribution (one-pass filter)")
    print(f"{'threshold':<20} {'users':>12} {'% of users':>12} {'events':>14}")
    exactly_one = int((counts == 1).sum())
    print(f"{'users == 1 event':<20} {exactly_one:>12,} "
          f"{exactly_one / n_users:>11.2%} {exactly_one:>14,}")
    for r in rows:
        print(f"{'users >= ' + str(r['k']) + ' events':<20} {r['users']:>12,} "
              f"{r['pct']:>11.2%} {r['events']:>14,}")

    rule(f"{label}: iterative bipartite k-core (user >= k AND item >= k)")
    print("A one-pass filter and a k-core are different things. The collapse")
    print("between the two columns below is the sparsity of this corpus.")
    print()
    print(f"{'k':>3} {'events':>12} {'users':>10} {'items':>10} {'density':>10}"
          f"   {'vs one-pass users':>18}")
    onepass = {r["k"]: r["users"] for r in rows}
    for k in CORE_KS:
        c = iterative_core(u, i, k)
        ref = onepass.get(k)
        cmp = (f"{ref:,} -> {c['users']:,}" if ref else "")
        shrink = f" ({ref / c['users']:.0f}x)" if ref and c["users"] else ""
        print(f"{c['k']:>3} {c['events']:>12,} {c['users']:>10,} "
              f"{c['items']:>10,} {c['density']:>9.4%}   {cmp:>18}{shrink}")


def report_cohorts(u: np.ndarray, n_users: int, label: str) -> None:
    """Warm/cold split, and how thin the warm cohort actually is.

    The headline risk this guards against: at k=2 the warm cohort sounds like it
    has usable history, but under leave-last-out a user with exactly 2 events
    contributes exactly ONE training event. If most of the warm cohort is in
    that bucket, a single averaged warm metric is really measuring near-cold
    users, and the warm/cold contrast that the whole evaluation rests on gets
    diluted without anyone noticing.
    """
    counts = np.bincount(u, minlength=n_users)
    cold = int((counts == 1).sum())
    warm = int((counts >= 2).sum())

    rule(f"{label}: cohorts and training history")
    print(f"cold cohort (exactly 1 event) : {cold:>10,}  {cold / n_users:>7.2%}")
    print(f"warm cohort (>= 2 events)     : {warm:>10,}  {warm / n_users:>7.2%}")
    print()
    print("Under leave-last-out the last event goes to test, so training")
    print("history = events - 1. Warm cohort broken down by TRAINING history:")
    print()
    print(f"{'train events':<16} {'users':>12} {'% of warm':>12}")
    buckets = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, None, ">= 10")]
    for lo, hi, name in buckets:
        lo_e, hi_e = lo + 1, (hi + 1 if hi else None)
        sel = counts >= lo_e
        if hi_e:
            sel &= counts <= hi_e
        n = int(sel.sum())
        print(f"{name:<16} {n:>12,} {(n / warm if warm else 0):>11.2%}")

    thin = int((counts == 2).sum())
    usable = int((counts >= 3).sum())
    if warm:
        print()
        print(f"NOTE: {thin:,} of {warm:,} warm users ({thin / warm:.1%}) have exactly")
        print("      ONE training event. Report warm metrics stratified by this")
        print("      bucket, or the warm number is mostly measuring near-cold users.")
        print()
        print(f"      Users with >= 2 TRAINING events: {usable:,} "
              f"({usable / n_users:.2%} of this scope).")
        print("      That is the population on which user-based CF is not")
        print("      degenerate, and it is the number that gates the evaluation.")
    return usable


def report_feedback_taxonomy(rating: np.ndarray, verified: np.ndarray,
                             label: str = "") -> None:
    """Exact counts for the feedback-type mapping.

    The plan's version of this table was extrapolated from the rating
    distribution times the overall verified rate. These are the measured
    cross-tab counts, which is what any claim about the taxonomy has to cite.
    """
    rule(f"{label}: feedback taxonomy — measured cross-tab (verified x rating)")
    print(f"{'rating':>7} {'verified':>12} {'not verified':>14} {'total':>12}")
    for r in (1, 2, 3, 4, 5):
        sel = rating == r
        v = int((sel & verified).sum())
        nv = int((sel & ~verified).sum())
        print(f"{r:>6}* {v:>12,} {nv:>14,} {v + nv:>12,}")
    tv, tnv = int(verified.sum()), int((~verified).sum())
    print(f"{'total':>7} {tv:>12,} {tnv:>14,} {tv + tnv:>12,}")
    print(f"\nverified_purchase rate: {tv / len(rating):.2%}")

    # Precedence is explicit because the plan's table is ambiguous for the
    # overlapping cases (an unverified 1-star matches two rows). Explicit
    # negatives win: a 1-2* review is a strong negative signal whether or not
    # the purchase was verified.
    purchase = int((verified & (rating >= 4)).sum())
    dislike = int((rating <= 2).sum())
    view_v3 = int((verified & (rating == 3)).sum())
    view_nv = int((~verified & (rating >= 3)).sum())

    rule(f"{label}: feedback taxonomy — applied mapping")
    print("precedence: rating <= 2 -> dislike, before any verified/unverified rule")
    print()
    print(f"{'gorse feedback type':<22} {'events':>12} {'share':>8}   rule")
    total = len(rating)
    for name, n, why in [
        ("purchase", purchase, "verified AND rating >= 4"),
        ("dislike",  dislike,  "rating <= 2 (explicit negative)"),
        ("view",     view_v3 + view_nv, "verified 3*, or unverified 3-5*"),
    ]:
        print(f"{name:<22} {n:>12,} {n / total:>7.2%}   {why}")
    print(f"{'TOTAL':<22} {purchase + dislike + view_v3 + view_nv:>12,}")


def report_scope(u: np.ndarray, i: np.ndarray, rating: np.ndarray,
                 verified: np.ndarray, ts_ms: np.ndarray,
                 raw_events: int, label: str) -> int:
    """Full measurement panel for one scope (whole corpus, or one catalogue).

    Both scopes get the identical panel on purpose: the catalogue-restricted
    numbers are the ones the recommender will actually be evaluated on, so
    quoting a corpus-level statistic as though it described the served system
    would overstate the density by a wide margin.
    """
    # Reindex densely HERE rather than at the call sites. Every count below
    # derives the population size from max(code)+1, which is only correct for
    # dense codes; passing a subset's original codes silently inflates the
    # denominator and deflates every per-user average.
    u_dense = np.unique(u, return_inverse=True)[1]
    i_dense = np.unique(i, return_inverse=True)[1]

    report_corpus(u_dense, i_dense, raw_events, label)
    report_timespan(ts_ms)
    usable = report_cohorts(u_dense, int(u_dense.max()) + 1, label)
    report_feedback_taxonomy(rating, verified, label)
    return usable


def report_timespan(ts_ms: np.ndarray) -> None:
    import datetime as dt
    lo = dt.datetime.fromtimestamp(int(ts_ms.min()) / 1000, dt.timezone.utc)
    hi = dt.datetime.fromtimestamp(int(ts_ms.max()) / 1000, dt.timezone.utc)
    print(f"\ntimestamp range    : {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")


# ── Stage 4: catalogue join ───────────────────────────────────────────────────

def load_catalogue(n: int | None) -> list[str] | None:
    """product_ids from rag_products, top-N by rating_count (seed_gorse order).

    Returns None if Postgres is unreachable, so --stats still produces the
    corpus-level numbers on a machine with no database running.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(POSTGRES_URL)
    except Exception as e:
        print(f"\n[skip] catalogue join: Postgres unavailable ({type(e).__name__})")
        return None
    try:
        with conn.cursor() as cur:
            if n:
                cur.execute("SELECT product_id FROM rag_products "
                            "ORDER BY rating_count DESC LIMIT %s", (n,))
            else:
                cur.execute("SELECT product_id FROM rag_products")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_stats(args) -> None:
    table = load_events(refresh=args.refresh, limit=args.limit)
    raw_events = table.num_rows

    # Dictionary-encoded columns give integer codes for free.
    u_all = table.column("user_id").combine_chunks().indices.to_numpy().astype(np.int64)
    i_all = table.column("product_id").combine_chunks().indices.to_numpy().astype(np.int64)
    rating = table.column("rating").to_numpy().astype(np.int8)
    verified = table.column("verified").to_numpy(zero_copy_only=False)
    ts_ms = table.column("ts_ms").to_numpy()

    keep = dedupe_edges(u_all, i_all)
    u, i = u_all[keep], i_all[keep]

    # rating/verified/ts must follow the same dedup selection as the edges,
    # otherwise the taxonomy counts a duplicate edge the graph has dropped.
    rating_d, verified_d, ts_d = rating[keep], verified[keep], ts_ms[keep]

    rule("FULL CORPUS")
    print("Every review in the dump, over the complete Amazon Fashion catalogue.")
    print("This is the population the architecture argument is about.")
    usable_full = report_scope(u, i, rating_d, verified_d, ts_d,
                               raw_events, "full corpus")

    # ── Catalogue-restricted view ────────────────────────────────────────────
    catalogue = load_catalogue(args.catalogue_size)
    if catalogue is None:
        return

    dict_items = table.column("product_id").combine_chunks().dictionary.to_pylist()
    code_of = {pid: n for n, pid in enumerate(dict_items)}
    wanted = np.zeros(len(dict_items), dtype=bool)
    hit = 0
    for pid in catalogue:
        c = code_of.get(pid)
        if c is not None:
            wanted[c] = True
            hit += 1

    sel = wanted[i]
    cu, ci = u[sel], i[sel]
    if not len(cu):
        print("\n[skip] catalogue join: no events matched the catalogue")
        return

    size_label = args.catalogue_size or len(catalogue)
    rule(f"CATALOGUE-RESTRICTED  (top {size_label:,} rag_products by rating_count)")
    print("Only events on products the recommender actually knows about.")
    print("Coverage denominators below use the CATALOGUE size, not the corpus.")
    print()
    print(f"catalogue size     : {len(catalogue):,}")
    print(f"  present in dump  : {hit:,} ({hit / len(catalogue):.1%})")
    print(f"joined events      : {len(cu):,} "
          f"({len(cu) / len(u):.1%} of corpus edges)")
    items_hit = len(np.unique(ci))
    print(f"items with >=1 event: {items_hit:,} / {len(catalogue):,} "
          f"= {items_hit / len(catalogue):.1%} catalogue coverage")

    usable_cat = report_scope(cu, ci, rating_d[sel], verified_d[sel],
                              ts_d[sel], len(cu), f"catalogue({size_label})")

    rule("WHAT THIS MEANS FOR THE EVALUATION")
    print(f"full corpus      : {usable_full:,} users have >= 2 training events")
    print(f"catalogue({size_label:,})  : {usable_cat:,} users have >= 2 training events")
    print()
    print("The catalogue truncates user histories, so restricting to the served")
    print("catalogue costs most of the evaluable warm population. A warm cohort")
    print(f"of {usable_cat:,} is too small to carry a credible NDCG comparison, which is")
    print("what --sweep-catalogue exists to fix: find the catalogue size where")
    print("the warm cohort becomes large enough without inventing signal.")
    if usable_cat < 1000:
        print()
        print(f"WARNING: {usable_cat:,} evaluable warm users at this catalogue size.")
        print("         Do not lock an eval baseline on this.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure and build the Amazon Fashion interaction dataset.")
    p.add_argument("--stats", action="store_true",
                   help="measure only, write nothing")
    p.add_argument("--catalogue-size", type=int, default=None,
                   help="restrict the join to the top-N rag_products by "
                        "rating_count (default: the whole table)")
    p.add_argument("--refresh", action="store_true",
                   help="re-parse the raw dump instead of using the cache")
    p.add_argument("--limit", type=int, default=None,
                   help="only read the first N lines (smoke test; not cached)")
    args = p.parse_args()

    if not args.stats:
        p.error("only --stats is implemented so far; "
                "--build and --sweep-catalogue are the next step")
    run_stats(args)


if __name__ == "__main__":
    main()
