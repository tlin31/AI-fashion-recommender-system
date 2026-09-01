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
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_REVIEWS = (REPO_ROOT / "rag-service" / "amazon_data" / "raw" /
               "review_categories" / "Amazon_Fashion.jsonl")

RAW_META = (REPO_ROOT / "rag-service" / "amazon_data" / "raw" /
            "meta_categories" / "meta_Amazon_Fashion.jsonl")

# Extracted columns are cached so that repeated analysis (and the catalogue-size
# sweep, which needs many passes) does not re-parse 1 GB of JSON every time.
CACHE_PATH = Path(__file__).parent / "cache" / "amazon_fashion_events.parquet"
META_CACHE_PATH = Path(__file__).parent / "cache" / "amazon_fashion_meta.parquet"

# Where the demo catalogue's product images live. The sweep reports the overlap
# because the eval catalogue and the demo catalogue are deliberately different
# things: eval needs scale and no images, the demo needs images and no scale.
IMAGES_DIR = Path(__file__).parent.parent / "public" / "images"

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

def rule_hdr(title: str = "") -> None:
    print(f"\n{'─' * 78}")
    if title:
        print(title)
        print("─" * 78)


def report_corpus(u: np.ndarray, i: np.ndarray, raw_events: int, label: str) -> None:
    n_users = int(u.max()) + 1
    n_items = int(i.max()) + 1
    events = len(u)

    rule_hdr(f"{label}: interaction graph")
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
    rule_hdr(f"{label}: user degree distribution (one-pass filter)")
    print(f"{'threshold':<20} {'users':>12} {'% of users':>12} {'events':>14}")
    exactly_one = int((counts == 1).sum())
    print(f"{'users == 1 event':<20} {exactly_one:>12,} "
          f"{exactly_one / n_users:>11.2%} {exactly_one:>14,}")
    for r in rows:
        print(f"{'users >= ' + str(r['k']) + ' events':<20} {r['users']:>12,} "
              f"{r['pct']:>11.2%} {r['events']:>14,}")

    rule_hdr(f"{label}: iterative bipartite k-core (user >= k AND item >= k)")
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

    rule_hdr(f"{label}: cohorts and training history")
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
    rule_hdr(f"{label}: feedback taxonomy — measured cross-tab (verified x rating)")
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

    rule_hdr(f"{label}: feedback taxonomy — applied mapping")
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


# ── Catalogue-size sweep ──────────────────────────────────────────────────────
#
# THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT
# --------------------------------------------------
# Answers: how large must the catalogue be before the warm cohort is big enough
# to carry a credible NDCG comparison? At the current 5,000 products there are
# 370 users with >= 2 training events, which is not enough to lock a baseline on.
#
# Does NOT answer: exactly which ASINs a larger rag_products would contain. That
# is normalize.py's business, and reproducing its filter chain here was tried and
# abandoned -- a faithful re-implementation of its published steps still only
# reproduces 56% of the actual 5,000-row catalogue, so any "predicted catalogue"
# would be a fiction with a precise-looking membership.
#
# So the sweep reports the SHAPE of the curve under two independent
# catalogue-selection rules. If both rules agree on the catalogue size needed,
# the decision is robust to the thing that could not be reproduced exactly.

def extract_meta() -> pa.Table:
    """parent_asin + the fields normalize.py filters on, from the metadata dump."""
    if not RAW_META.exists():
        sys.exit(f"Product metadata not found at {RAW_META}")

    sys.path.insert(0, str(REPO_ROOT / "rag-service" / "data"))
    try:
        from normalize import _derive_category, _parse_price
    except ImportError as e:
        sys.exit(f"Could not import rag-service/data/normalize.py helpers: {e}")

    asins, counts, titles, brands, feats, descs, prices = [], [], [], [], [], [], []
    desc_txt, ratings_avg = [], []
    started = time.time()
    print(f"Scanning {RAW_META.name} "
          f"({RAW_META.stat().st_size / 1e6:,.0f} MB)...")

    with open(RAW_META, "rb") as fh:
        for n, line in enumerate(fh, 1):
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            title = r.get("title") or ""
            f = r.get("features") or []
            d = r.get("description") or []
            asins.append(r.get("parent_asin") or "")
            counts.append(r.get("rating_number") or 0)
            titles.append(title)
            brands.append((r.get("store") or "").lower().strip())
            joined = " ".join(d) if isinstance(d, list) else ""
            feats.append(len(f) if isinstance(f, list) else 0)
            descs.append(len(joined))
            prices.append(_parse_price(r.get("price")) or -1.0)
            # Kept for --build: the eval catalogue needs the same text the
            # seeder tags style/colour from. Truncated because only the first
            # few hundred characters carry the descriptive keywords, and the
            # full field would triple the cache for no labelling gain.
            desc_txt.append(joined[:300])
            ratings_avg.append(r.get("average_rating") or 0.0)
            if n % 500_000 == 0:
                print(f"  {n:,} lines ({n / (time.time() - started):,.0f}/s)")

    print(f"  done: {len(asins):,} products in {time.time() - started:.0f}s")

    # Category derivation is the expensive per-row step; do it once, here.
    cats = [(_derive_category(t) or "") for t in titles]
    return pa.table({
        "product_id":   pa.array(asins),
        "rating_count": pa.array(counts, type=pa.float64()),
        "title":        pa.array(titles),
        "title_len":    pa.array([len(t) for t in titles], type=pa.int32()),
        "brand":        pa.array(brands).dictionary_encode(),
        "feat_n":       pa.array(feats, type=pa.int32()),
        "desc_len":     pa.array(descs, type=pa.int32()),
        "description":  pa.array(desc_txt),
        "price":        pa.array(prices, type=pa.float64()),
        "avg_rating":   pa.array(ratings_avg, type=pa.float64()),
        "category":     pa.array(cats).dictionary_encode(),
    })


def load_meta_pool(refresh: bool = False):
    """The pool normalize.py subsamples from, as best it can be reconstructed.

    Only the FINAL step of normalize.py's pipeline depends on catalogue size
    (`nlargest(SUBSAMPLE_SIZE, rating_count)`); everything before it is
    size-independent. So reconstructing the pool once lets any N be taken from
    it. The reconstruction is imperfect (see the section header) and its
    fidelity is measured and printed rather than assumed.
    """
    import pandas as pd

    required = {"product_id", "rating_count", "title", "brand", "feat_n",
                "desc_len", "description", "price", "avg_rating", "category"}
    stale = (META_CACHE_PATH.exists()
             and not required.issubset(set(pq.read_schema(META_CACHE_PATH).names)))
    if stale:
        print("Cached metadata predates the current schema — re-extracting.")

    if META_CACHE_PATH.exists() and not refresh and not stale:
        print(f"Using cached product metadata: {META_CACHE_PATH}")
        table = pq.read_table(META_CACHE_PATH)
    else:
        table = extract_meta()
        META_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, META_CACHE_PATH, compression="zstd")
        print(f"  cached -> {META_CACHE_PATH}")

    df = table.to_pandas()
    stages = [("raw metadata", len(df))]
    df = df[df["category"] != ""];                     stages.append(("fashion keyword", len(df)))
    df = df[(df["title_len"] >= 60) &
            ((df["feat_n"] >= 2) | (df["desc_len"] >= 100))]
    stages.append(("richness", len(df)))
    df = df[df["rating_count"] >= 5];                  stages.append(("rating_count >= 5", len(df)))
    df = df[(df["price"] < 0) | (df["price"] <= 500)]; stages.append(("price <= 500 or null", len(df)))
    df = df.sort_values("rating_count", ascending=False)
    df = df.drop_duplicates(subset=["product_id"]);    stages.append(("dedup parent_asin", len(df)))

    print("\nreconstructed selection pool (normalize.py's filter chain):")
    for name, n in stages:
        print(f"  {name:<24} {n:>10,}")
    return df


def sweep(args, u, i, item_names) -> None:
    import pandas as pd

    sizes = []
    for tok in args.sweep_catalogue.split(","):
        tok = tok.strip().lower()
        sizes.append(None if tok == "all" else int(tok))

    code_of = {pid: n for n, pid in enumerate(item_names)}
    n_items_total = len(item_names)

    # Rule B key: how many times each item is actually interacted with here.
    dump_counts = np.bincount(i, minlength=n_items_total)

    # Rule A key: metadata rating_count, over the reconstructed pool.
    pool = load_meta_pool(refresh=args.refresh)
    pool_codes = np.array([code_of.get(p, -1) for p in pool["product_id"]])
    pool_rank = pool_codes[pool_codes >= 0]  # already sorted by rating_count desc

    # Fidelity of the reconstruction, measured at the one size where ground
    # truth exists. Printed, never assumed.
    real = load_catalogue(None)
    if real:
        real_set = {code_of[p] for p in real if p in code_of}
        recon = set(pool_rank[:len(real)].tolist())
        ov = len(real_set & recon)
        print(f"\nreconstruction fidelity at N={len(real):,}: "
              f"{ov:,}/{len(real):,} = {ov / len(real):.1%} of the real "
              f"rag_products catalogue")
        print("  -> the sweep therefore reports curve SHAPE, not a predicted "
              "catalogue membership")

    have_image = set()
    if IMAGES_DIR.exists():
        have_image = {p.stem for p in IMAGES_DIR.glob("*.jpg")}

    rows = []
    for rule, order in (("meta rating_count", pool_rank),
                        ("dump review count", np.argsort(-dump_counts))):
        seen_sizes = set()
        for N in sizes:
            picked = order if N is None else order[:N]
            # A requested N larger than the available pool clamps to the pool,
            # which would otherwise emit the same row twice under two labels.
            if len(picked) in seen_sizes:
                continue
            seen_sizes.add(len(picked))
            sel_items = np.zeros(n_items_total, dtype=bool)
            sel_items[picked] = True
            sel = sel_items[i]
            cu = u[sel]
            if not len(cu):
                continue
            cu_d = np.unique(cu, return_inverse=True)[1]
            ci_d = np.unique(i[sel], return_inverse=True)[1]
            deg = np.bincount(cu_d)
            core3 = iterative_core(cu_d, ci_d, 3)
            imgs = sum(1 for c in picked if item_names[c] in have_image)
            rows.append({
                "rule": rule,
                "N": len(picked),
                "events": len(cu),
                "users": len(deg),
                "warm": int((deg >= 2).sum()),
                "evaluable": int((deg >= 3).sum()),
                "core3_users": core3["users"],
                "images": imgs,
            })

    rule_hdr("CATALOGUE-SIZE SWEEP")
    print("evaluable = users with >= 2 TRAINING events (>= 3 total, under")
    print("leave-last-out). That is the population user-based CF can serve,")
    print("and the number that decides whether an eval baseline is lockable.")
    print()
    print(f"{'selection rule':<19} {'catalogue N':>12} {'events':>11} "
          f"{'users':>10} {'warm':>9} {'evaluable':>10} {'3-core u':>9} {'imgs':>6}")
    last_rule = None
    for r in rows:
        if r["rule"] != last_rule:
            print(f"{'─' * 96}")
            last_rule = r["rule"]
        flag = "" if r["evaluable"] >= 1000 else "  <- too small"
        print(f"{r['rule']:<19} {r['N']:>12,} {r['events']:>11,} "
              f"{r['users']:>10,} {r['warm']:>9,} {r['evaluable']:>10,} "
              f"{r['core3_users']:>9,} {r['images']:>6,}{flag}")

    rule_hdr("READING THE SWEEP")
    df = pd.DataFrame(rows)
    thresholds = {}
    for rname in df["rule"].unique():
        sub = df[df["rule"] == rname]
        ok = sub[sub["evaluable"] >= 1000]
        thresholds[rname] = int(ok.iloc[0]["N"]) if len(ok) else None
        if thresholds[rname]:
            print(f"{rname:<19}: >= 1,000 evaluable warm users from N = "
                  f"{thresholds[rname]:,}")
        else:
            print(f"{rname:<19}: never reaches 1,000 evaluable warm users")

    print()
    print("THE TWO RULES DISAGREE, AND ONLY ONE OF THEM IS HONEST.")
    print()
    print("'dump review count' selects items BECAUSE they have many observed")
    print("interactions, then measures how many interactions the selection has.")
    print("That is selection on the outcome: it inflates the density it reports")
    print("and cannot predict what a real catalogue would look like. It is shown")
    print("only as the optimistic bound.")
    print()
    print("'meta rating_count' is the rule normalize.py actually keys on, and it")
    print("is independent of this dump's interaction counts. Size the catalogue")
    print("from that row.")

    pool_rows = df[df["rule"] == "meta rating_count"]
    if len(pool_rows):
        ceiling = pool_rows.iloc[-1]
        print()
        print("CEILING: normalize.py's quality filters (fashion keyword, richness,")
        print(f"         rating_count >= 5, price) leave only {ceiling['N']:,} candidate")
        print("         products in total. The catalogue cannot be grown past that")
        print(f"         without relaxing those filters, which caps the evaluable")
        print(f"         warm cohort at {ceiling['evaluable']:,} users.")

    print()
    print("NOTE: eval catalogue != demo catalogue. The 'imgs' column is how many")
    print(f"      of the {len(have_image):,} downloaded product images fall inside each")
    print("      catalogue; it saturates early because images were fetched for the")
    print("      top of the same ranking. The demo front-end stays on the")
    print("      image-backed subset regardless of how large the eval catalogue grows.")


# ── Build: write the eval dataset to Postgres ─────────────────────────────────
#
# WHY A SEPARATE CATALOGUE TABLE
# ------------------------------
# The obvious route -- raise normalize.py's SUBSAMPLE_SIZE and regrow
# rag_products -- was rejected. rag_products is rag-service's RETRIEVAL CORPUS:
# the BM25 index is built from it (rag-service/main.py:98,
# pipeline/retrieval.py:170) and CRAG queries it. Growing it 5,000 -> 95,335
# would change retrieval for every query, require re-embedding 19x more products
# into Milvus, and silently invalidate the 1,481 locked relevance judgments,
# because products entering the corpus unjudged count as non-relevant and would
# depress NDCG/Recall for reasons that have nothing to do with retrieval quality.
# That is a large cascading cost to another subsystem's locked baseline, and it
# buys the recommender nothing.
#
# So the eval catalogue is its own table. This also dissolves the fidelity
# problem the sweep had to hedge around: we are no longer trying to predict what
# a bigger rag_products would contain, we are defining our own catalogue by a
# rule that is stated and reproducible.

# Both tables are fully derived from the dump, so --build drops and recreates
# them rather than migrating. CREATE TABLE IF NOT EXISTS silently keeps an old
# column layout, which is how a schema change turns into a confusing runtime
# error two steps later; dropping makes a schema edit take effect on the next
# build with no migration to remember.
SCHEMA_SQL = """
DROP TABLE IF EXISTS reco_interactions;
DROP TABLE IF EXISTS reco_products;

CREATE TABLE IF NOT EXISTS reco_products (
    product_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    brand        TEXT,
    category     TEXT,
    -- Carried so that style/colour tagging has the same text to work from as
    -- seed_gorse.py does. Tagging from the title alone drops style coverage
    -- from 68.9% to 37.8%, and the content arms depend on that signal.
    description  TEXT,
    price        DOUBLE PRECISION,
    price_range  TEXT,
    avg_rating   DOUBLE PRECISION,
    rating_count BIGINT
);

CREATE TABLE IF NOT EXISTS reco_interactions (
    user_id       TEXT NOT NULL,
    product_id    TEXT NOT NULL REFERENCES reco_products(product_id),
    feedback_type TEXT NOT NULL,
    rating        SMALLINT,
    verified      BOOLEAN,
    ts            TIMESTAMPTZ NOT NULL,
    split         TEXT NOT NULL,          -- train | test
    cohort        TEXT NOT NULL,          -- warm | cold
    PRIMARY KEY (user_id, product_id)
);

CREATE INDEX IF NOT EXISTS reco_interactions_split_cohort
    ON reco_interactions (split, cohort);
CREATE INDEX IF NOT EXISTS reco_interactions_user
    ON reco_interactions (user_id);
"""


def assign_feedback_type(rating: np.ndarray, verified: np.ndarray) -> np.ndarray:
    """Map (rating, verified) to a Gorse feedback type.

    Precedence is explicit: an explicit negative wins over the verified split,
    so an unverified 1-star is a dislike rather than a view. The plan's table
    left this ambiguous, and the two readings differ by tens of thousands of
    events.
    """
    out = np.full(len(rating), "view", dtype=object)
    out[(rating >= 4) & verified] = "purchase"
    out[rating <= 2] = "dislike"
    return out


def global_temporal_split(ts: np.ndarray, quantile: float) -> tuple[np.ndarray, int]:
    """Split at a single wall-clock cutoff: everything after it is test.

    WHY THIS IS THE DEFAULT, AND NOT LEAVE-LAST-OUT
    ----------------------------------------------
    Leave-last-out is the textbook protocol, but on a corpus where 86% of users
    appear exactly once it sends ~95% of all events to test. Measured on this
    catalogue it leaves 30,798 training events and gives only 17,644 of 95,335
    items (18.5%) any training signal at all. An item with no training
    interaction cannot be recommended by CF, popularity, or item-to-item, so
    catalog coverage would be structurally capped at 18.5% and Gini would be
    computed over a truncated universe -- the beyond-accuracy metrics that are
    supposed to be the interesting part would be measuring the protocol rather
    than the recommender.

    A single global cutoff fixes that (65% item coverage at q=0.70) and is
    strictly MORE leak-proof than leave-last-out, not less: leave-last-out will
    happily train on a 2022 event while testing a 2015 event belonging to a
    different user, which is future-to-past leakage across users. One cutoff
    makes every training event older than every test event.

    The plan's prohibition was on a global RANDOM split, which this is not.
    """
    cutoff = int(np.quantile(ts, quantile))
    return ts > cutoff, cutoff


def leave_last_out_split(u: np.ndarray, ts: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Leave-last-out per user. Returns a boolean mask: True = test.

    The last event by timestamp goes to test, everything earlier to train. A
    single-event user therefore contributes one test row and NO training row,
    which is not an edge case to paper over -- it IS the definition of the cold
    cohort, and it is 86% of this corpus.

    Kept as an option because it maximises the evaluable warm cohort (3,420
    users vs 859 under a global cutoff); see the docstring above for what that
    costs on the item side.
    """
    # Ties on timestamp are common (same-day imports). Break them on product_id
    # so that a re-run produces the identical split rather than a new one.
    order = np.lexsort((pid, ts, u))
    is_test = np.zeros(len(u), dtype=bool)
    su = u[order]
    last_of_user = np.empty(len(order), dtype=bool)
    last_of_user[-1] = True
    np.not_equal(su[1:], su[:-1], out=last_of_user[:-1])
    is_test[order[last_of_user]] = True
    return is_test


def build(args, table) -> None:
    import pandas as pd
    import psycopg2
    from psycopg2.extras import execute_values

    sys.path.insert(0, str(Path(__file__).parent))
    from seed_gorse import _colour_labels, _style_labels  # noqa: F401  (used by push)

    i_col = table.column("product_id").combine_chunks()
    item_names = i_col.dictionary.to_pylist()
    u_all = table.column("user_id").combine_chunks()
    user_names = u_all.dictionary.to_pylist()

    u = u_all.indices.to_numpy().astype(np.int64)
    i = i_col.indices.to_numpy().astype(np.int64)
    rating = table.column("rating").to_numpy().astype(np.int8)
    verified = table.column("verified").to_numpy(zero_copy_only=False)
    ts_ms = table.column("ts_ms").to_numpy()

    keep = dedupe_edges(u, i)
    u, i, rating, verified, ts_ms = (u[keep], i[keep], rating[keep],
                                     verified[keep], ts_ms[keep])

    # ── Select the eval catalogue ────────────────────────────────────────────
    pool = load_meta_pool(refresh=False)
    n = args.catalogue_size
    cat = pool if n is None else pool.nlargest(n, "rating_count")
    code_of = {pid: c for c, pid in enumerate(item_names)}
    cat = cat[cat["product_id"].isin(code_of)]
    print(f"\neval catalogue: {len(cat):,} products "
          f"({'entire filtered pool' if n is None else f'top {n:,} by rating_count'})")

    cat_codes = np.zeros(len(item_names), dtype=bool)
    cat_codes[[code_of[p] for p in cat["product_id"]]] = True
    sel = cat_codes[i]
    u, i, rating, verified, ts_ms = (u[sel], i[sel], rating[sel],
                                     verified[sel], ts_ms[sel])
    print(f"joined events : {len(u):,}")

    # ── Split and cohort ─────────────────────────────────────────────────────
    u_dense = np.unique(u, return_inverse=True)[1]
    n_users = int(u_dense.max()) + 1

    if args.split == "temporal":
        is_test, cutoff = global_temporal_split(ts_ms, args.split_quantile)
        import datetime as _dt
        cut_date = _dt.datetime.fromtimestamp(cutoff / 1000, _dt.timezone.utc)
        print(f"split         : global temporal at q={args.split_quantile} "
              f"({cut_date:%Y-%m-%d})")
        # Cohort is decided by history BEFORE the cutoff -- that is the only
        # information a recommender would have had at prediction time.
        train_hist = np.bincount(u_dense[~is_test], minlength=n_users)
        cohort = np.where(train_hist[u_dense] >= 1, "warm", "cold")
        evaluable = int((train_hist[np.unique(u_dense[is_test])] >= 2).sum())
        cold_test = int((train_hist[np.unique(u_dense[is_test])] == 0).sum())
    else:
        is_test = leave_last_out_split(u_dense, ts_ms, i)
        deg = np.bincount(u_dense, minlength=n_users)
        print("split         : leave-last-out per user")
        cohort = np.where(deg[u_dense] >= 2, "warm", "cold")
        train_hist = np.bincount(u_dense[~is_test], minlength=n_users)
        evaluable = int((deg >= 3).sum())
        cold_test = int((deg == 1).sum())
        # Under this protocol a cold user must contribute exactly one test row
        # and no training row; that is what makes them cold.
        cold_train = int(((cohort == "cold") & ~is_test).sum())
        assert cold_train == 0, f"{cold_train} cold-cohort rows landed in train"

    ftype = assign_feedback_type(rating, verified)

    n_train, n_test = int((~is_test).sum()), int(is_test.sum())
    train_items = len(np.unique(i[~is_test]))
    n_cat = len(cat)
    print(f"train / test  : {n_train:,} / {n_test:,}")
    print(f"users         : {n_users:,}")
    print(f"  cold test   : {cold_test:,} (no history before the split)")
    print(f"  evaluable   : {evaluable:,} test users with >= 2 training events")
    print(f"items w/ train: {train_items:,} / {n_cat:,} = {train_items / n_cat:.1%}")
    if train_items / n_cat < 0.30:
        print("  WARNING: most of the catalogue has no training signal, so catalog")
        print("           coverage and Gini will measure the split, not the model.")

    # No training event may be newer than the oldest test event under a global
    # cutoff -- that is the property the whole protocol is bought for.
    if args.split == "temporal" and n_train and n_test:
        assert ts_ms[~is_test].max() <= ts_ms[is_test].min(), "temporal split leaked"

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # ── Write ────────────────────────────────────────────────────────────────
    conn = psycopg2.connect(POSTGRES_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # SCHEMA_SQL drops both tables first, so this is inherently
            # re-runnable: a second build replaces, it never appends.
            cur.execute(SCHEMA_SQL)

            prices = cat["price"].to_numpy()
            rows = [
                (r.product_id, r.title[:500], r.brand, r.category, r.description,
                 None if r.price < 0 else float(r.price),
                 price_band(r.price, prices),
                 float(r.avg_rating) if r.avg_rating else None,
                 int(r.rating_count))
                for r in cat.itertuples()
            ]
            execute_values(cur, """
                INSERT INTO reco_products (product_id, name, brand, category,
                                           description, price, price_range,
                                           avg_rating, rating_count)
                VALUES %s
            """, rows, page_size=5000)
            print(f"\nwrote reco_products    : {len(rows):,}")

            inter = [
                (user_names[uu], item_names[ii], ft, int(rt), bool(vf),
                 int(t), "test" if te else "train", ch)
                for uu, ii, ft, rt, vf, t, te, ch
                in zip(u, i, ftype, rating, verified, ts_ms, is_test, cohort)
            ]
            execute_values(cur, """
                INSERT INTO reco_interactions (user_id, product_id, feedback_type,
                                               rating, verified, ts, split, cohort)
                VALUES %s
            """, inter, template="(%s,%s,%s,%s,%s,to_timestamp(%s/1000.0),%s,%s)",
                page_size=10000)
            print(f"wrote reco_interactions: {len(inter):,}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("\nBuild complete. Re-running replaces both tables; it does not append.")


def price_band(price: float, all_prices: np.ndarray) -> str:
    """Terciles over the catalogue's own priced products.

    Mirrors normalize.py's bucket_price, except that a missing price is
    'unknown' rather than being folded into 'mid' -- see seed_gorse.py for why
    that conflation matters.
    """
    if price is None or price < 0:
        return "unknown"
    valid = all_prices[all_prices >= 0]
    if not len(valid):
        return "unknown"
    p33, p67 = np.quantile(valid, [0.33, 0.67])
    return "budget" if price < p33 else ("mid" if price < p67 else "premium")


# ── Push: load the built dataset into Gorse ───────────────────────────────────

GORSE_URL = os.environ.get("GORSE_URL", "http://localhost:8088")


def _post_batched(path: str, payload: list, label: str, batch: int = 2000) -> int:
    import httpx

    sent = 0
    for start in range(0, len(payload), batch):
        chunk = payload[start:start + batch]
        for attempt in range(1, 5):
            try:
                resp = httpx.post(f"{GORSE_URL}{path}", json=chunk, timeout=120)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = attempt * 2
                print(f"  {label} {start:,}: {type(e).__name__}, retry in {wait}s")
                time.sleep(wait)
        sent += len(chunk)
        if (start // batch) % 10 == 0 or sent == len(payload):
            print(f"  {label}: {sent:,}/{len(payload):,}")
    return sent


# ── Evaluation cohort ────────────────────────────────────────────────────────
#
# Pushing every user is not an option on this hardware. Gorse materialises a
# per-user recommendation cache for every user that has at least one positive
# feedback -- 265,828 of the 371,207 train users -- and at cache_size=30 that is
# roughly 8M entries before item-to-item's own 11.5M. Measured: it does not fit,
# and LRU eviction during a batch build never converges, silently.
#
# What the evaluation actually needs is much smaller. Only users with BOTH a
# train and a test event can be scored at all; there are 5,627 of them, holding
# 7,333 train events between them.
#
# The extra sample exists for one reason only: to leave the collaborative
# filtering model a graph with realistic density. Without it CF would be fitted
# on 7,333 events and its (already measured) ineffectiveness would become an
# artefact of our sampling rather than a property of the corpus.
#
# Note that the sampling does NOT affect the primary cold-start arm.
# `[[recommend.item-to-item]] style_similarity` is `type = "tags"` over
# `item.Labels`, and `tagsItemToItem.Push(item, _ []int32)` discards the
# feedback argument outright -- it reads item labels and nothing else.
#
# A tempting alternative was rejected after reading the source: pushing every
# interaction while creating only some user rows. `storage/data/sql.go:1121`
# gates feedback insertion on `users.Contains(f.UserId)`, and with
# `Server.AutoInsertUser` off the loop above it removes any user missing from
# the table. The feedback is silently dropped, not stored user-less.

_EVALUABLE_SQL = """
    SELECT user_id
    FROM reco_interactions
    WHERE split = 'train'
      AND user_id IN (SELECT user_id FROM reco_interactions WHERE split = 'test')
    GROUP BY user_id
"""


def _select_cohort(cur, sample: int | None, seed: int) -> tuple[set[str], dict]:
    """Choose which users' train events to push.

    Returns (user_ids, stats). `sample is None` means push everyone, which is
    the pre-existing behaviour and is kept so the flag is additive.
    """
    cur.execute("SELECT count(DISTINCT user_id) AS n FROM reco_interactions "
                "WHERE split = 'train'")
    n_all = cur.fetchone()["n"]

    if sample is None:
        return set(), {"mode": "all", "n_all": n_all}

    cur.execute(_EVALUABLE_SQL)
    evaluable = {r["user_id"] for r in cur.fetchall()}

    cur.execute("SELECT DISTINCT user_id FROM reco_interactions WHERE split = 'train'")
    others = [r["user_id"] for r in cur.fetchall() if r["user_id"] not in evaluable]

    rng = random.Random(seed)
    extra = set(rng.sample(others, min(sample, len(others))))
    cohort = evaluable | extra

    return cohort, {
        "mode": "sampled", "n_all": n_all, "evaluable": len(evaluable),
        "extra_requested": sample, "extra_taken": len(extra), "total": len(cohort),
    }


def cohort_report() -> None:
    """Print the evaluable strata and what each sample size would cost.

    Exists because the two numbers that get quoted for "the warm cohort" -- 859
    and 5,627 -- differ by 6.5x and mean different things, and because 85% of
    the larger figure holds exactly one training event, which makes an
    unstratified "warm" metric a measurement of near-cold users.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                WITH tr AS (
                    SELECT user_id, count(*) AS n FROM reco_interactions
                    WHERE split = 'train' GROUP BY user_id),
                     te AS (
                    SELECT DISTINCT user_id FROM reco_interactions WHERE split = 'test')
                SELECT CASE WHEN n = 1 THEN '1'
                            WHEN n = 2 THEN '2'
                            WHEN n <= 4 THEN '3-4'
                            WHEN n <= 9 THEN '5-9'
                            ELSE '10+' END AS bucket,
                       count(*) AS users, sum(n) AS train_events
                FROM tr JOIN te USING (user_id)
                GROUP BY bucket ORDER BY min(n)
            """)
            strata = cur.fetchall()

            cur.execute("SELECT count(DISTINCT user_id) AS n FROM reco_interactions "
                        "WHERE split = 'train'")
            n_all = cur.fetchone()["n"]
            cur.execute("""SELECT count(DISTINCT user_id) AS n FROM reco_interactions
                           WHERE split = 'train'
                             AND feedback_type IN ('purchase','favorite','add_to_cart')""")
            n_positive = cur.fetchone()["n"]
    finally:
        conn.close()

    rule_hdr("evaluable users, by training history")
    print("Only users with BOTH a train and a test event can be scored.\n")
    print(f"{'train events':>14}  {'users':>8}  {'share':>7}  {'train events held':>18}")
    total_u = sum(s["users"] for s in strata)
    for s in strata:
        print(f"{s['bucket']:>14}  {s['users']:>8,}  "
              f"{s['users'] / total_u:>6.1%}  {s['train_events']:>18,}")
    print(f"{'TOTAL':>14}  {total_u:>8,}")

    ge2 = sum(s["users"] for s in strata if s["bucket"] != "1")
    print(f"\n  train >= 1 : {total_u:,}")
    print(f"  train >= 2 : {ge2:,}   <- the figure quoted as 'the warm cohort'")
    print(f"\n  {total_u - ge2:,} of the {total_u:,} ({(total_u - ge2) / total_u:.0%}) hold "
          f"exactly one training event.")
    print("  Report warm metrics stratified by this column, or 'warm' names two "
          "different populations.")

    rule_hdr("what each --cohort-sample costs")
    print(f"all train users            : {n_all:,}")
    print(f"  of which have positive fb: {n_positive:,}  <- these get a per-user cache")
    print(f"\n{'--cohort-sample':>18}  {'users pushed':>13}  {'cache entries @30':>18}")
    for n in (0, 10_000, 25_000, 50_000, 100_000, None):
        pushed = n_all if n is None else min(total_u + n, n_all)
        label = "(omit flag = all)" if n is None else f"{n:,}"
        print(f"{label:>18}  {pushed:>13,}  {pushed * 30:>18,}")
    print("\nMeasured: ~22M entries did not fit in maxmemory 2300mb and evicted "
          "1.85M keys.\n165K-2M is comfortable.")


GORSE_DB_DSN = os.environ.get(
    "GORSE_DB_DSN",
    "postgres://gorse:gorse_pass@localhost:5432/gorse?sslmode=disable")


def reset_gorse_entities() -> None:
    """Clear Gorse's users and feedback, keeping items.

    Needed because pushing is an upsert. Shrinking the cohort without this
    leaves all 371K previously-pushed users in Gorse, every one of them still
    eligible for a per-user cache -- the flag would silently do nothing.

    Only derived state is destroyed. Gorse's data store is rebuilt from
    reco_products / reco_interactions on the host, which this never touches.

    Runs through `docker exec` rather than a direct connection because the
    Docker Postgres deliberately does not publish 5432 -- the host already has
    one there, and confusing the two is a documented trap in this project.
    """
    import subprocess
    for table in ("feedback", "users"):          # feedback first: it references users
        r = subprocess.run(
            ["docker", "exec", "fashion-postgres", "psql", "-U", "gorse",
             "-d", "gorse", "-c", f"TRUNCATE TABLE {table} CASCADE"],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise SystemExit(f"failed to truncate {table}: {r.stderr.strip()}")
        print(f"  truncated {table}")


def push_gorse(args) -> None:
    """Load reco_products + the TRAIN half of reco_interactions into Gorse.

    ONLY the training split is pushed. Sending test events would put the
    held-out interactions inside the model being evaluated, which silently
    inflates every metric and is unrecoverable once the model is fitted -- the
    numbers would look good and mean nothing. The WHERE clause below is the
    single most important line in this function.
    """
    import psycopg2
    import psycopg2.extras

    sys.path.insert(0, str(Path(__file__).parent))
    from seed_gorse import (_normalise_category, build_item_comment,
                            build_item_labels, flatten_labels)

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT product_id, name, brand, category, description,
                       price, price_range, avg_rating
                FROM reco_products
            """)
            products = cur.fetchall()

            cohort, cstats = _select_cohort(cur, args.cohort_sample, args.cohort_seed)

            if cstats["mode"] == "all":
                cur.execute("""
                    SELECT user_id, product_id, feedback_type, ts
                    FROM reco_interactions
                    WHERE split = 'train'
                """)
            else:
                # A temp table beats an IN (...) list of 55K ids by a wide margin
                # and keeps the plan stable as the cohort grows.
                cur.execute("CREATE TEMP TABLE _cohort (user_id TEXT PRIMARY KEY) "
                            "ON COMMIT DROP")
                psycopg2.extras.execute_values(
                    cur, "INSERT INTO _cohort (user_id) VALUES %s",
                    [(u,) for u in cohort], page_size=5000)
                cur.execute("""
                    SELECT i.user_id, i.product_id, i.feedback_type, i.ts
                    FROM reco_interactions i
                    JOIN _cohort c USING (user_id)
                    WHERE i.split = 'train'
                """)
            feedback = cur.fetchall()

            cur.execute("SELECT count(*) AS n FROM reco_interactions WHERE split='test'")
            n_test = cur.fetchone()["n"]
    finally:
        conn.close()

    if cstats["mode"] == "all":
        print(f"loaded {len(products):,} products, {len(feedback):,} TRAIN events "
              f"({n_test:,} test events deliberately withheld)")
        print(f"cohort: ALL {cstats['n_all']:,} train users "
              f"-- run --cohort-report to see why this may not fit")
    else:
        print(f"loaded {len(products):,} products, {len(feedback):,} TRAIN events "
              f"({n_test:,} test events deliberately withheld)")
        print(f"cohort: {cstats['total']:,} of {cstats['n_all']:,} train users "
              f"= {cstats['evaluable']:,} evaluable (train+test) "
              f"+ {cstats['extra_taken']:,} sampled (seed {args.cohort_seed})")
        print(f"        the sample exists to leave CF a graph with realistic "
              f"density; it does not affect tags item-to-item, which reads "
              f"item labels only")

    # ── Item labels ───────────────────────────────────────────────────────────
    #
    # The schema lives in seed_gorse.build_item_labels and is imported rather
    # than restated. It used to be restated, and the two copies had already
    # drifted: this file never emitted `occasion:` or `material:`, so the eval
    # harness's FEATURE_LABEL_PREFIXES listed two prefixes that did not exist on
    # a single one of the 95,335 items it was scoring.
    items, style_hits, colour_hits = [], 0, 0
    for p in products:
        labels, styles, colours = build_item_labels(p)
        style_hits += bool(styles)
        colour_hits += bool(colours)
        items.append({
            "ItemId":     p["product_id"],
            "Categories": [_normalise_category(p["category"])],
            "Labels":     labels,
            "Comment":    build_item_comment(p),
            "Timestamp":  datetime.now(timezone.utc).isoformat() + "Z",
        })

    n = len(products)
    print(f"label coverage: style {style_hits / n:.1%}, colour {colour_hits / n:.1%}")

    # ── What the similarity path can actually resolve ────────────────────────
    #
    # Printed on every push because it is the number the label restructure was
    # for. Before: 21.0% of items had an empty feature set once carriers were
    # excluded, and the surviving vocabulary partitioned the catalogue into
    # ~336 classes -- so most pairs were mutually indistinguishable to Jaccard
    # and all 330 sampled scores landed inside 1% of [0,1].
    feature_sets = [frozenset(l for _, l, sim in flatten_labels(it["Labels"]) if sim)
                    for it in items]
    empty = sum(1 for fs in feature_sets if not fs)
    classes = len(set(feature_sets))
    print(f"similarity path: {empty:,}/{n:,} = {empty/n:.1%} items with an empty "
          f"feature set, {classes:,} equivalence classes "
          f"({n/max(classes,1):.1f} items each)")

    events = [{
        "FeedbackType": f["feedback_type"],
        "UserId":       f["user_id"],
        "ItemId":       f["product_id"],
        "Timestamp":    f["ts"].isoformat(),
    } for f in feedback]

    by_type: dict[str, int] = {}
    for e in events:
        by_type[e["FeedbackType"]] = by_type.get(e["FeedbackType"], 0) + 1
    print(f"feedback types: {by_type}")
    print("  NOTE: config.toml lists positive_feedback_types = "
          "[purchase, favorite, add_to_cart] and read_feedback_types = [view].")
    print("        'dislike' is in neither, so Gorse stores it but the CF model")
    print("        does not currently train on it. That is a modelling decision")
    print("        for the ablation, not something this script should change.")

    if cstats["mode"] == "sampled" and not args.reset_gorse:
        print("\n  WARNING: --cohort-sample without --reset-gorse.")
        print("  Pushing is an upsert, so every user from a previous larger push")
        print("  stays in Gorse and stays eligible for a per-user cache. The")
        print("  cohort will not actually shrink. Add --reset-gorse.")

    if args.dry_run:
        print("\n--dry-run: nothing pushed.")
        return

    if args.reset_gorse:
        print("\nClearing Gorse users + feedback (items kept)...")
        reset_gorse_entities()

    print(f"\nPushing to Gorse ({GORSE_URL})...")
    if args.skip_items:
        print(f"  items: skipped ({len(items):,} unchanged, --skip-items)")
    else:
        _post_batched("/api/items", items, "items")
    _post_batched("/api/feedback", events, "feedback")
    print("\nPush complete.")


def verify_gorse(sample: int = 40) -> None:
    """Check the push landed, by sampling entities rather than reading counters.

    /api/dashboard/stats is NOT a verification source: the master recomputes it
    on a schedule, so immediately after a push it still reports the previous
    numbers. Reading it and concluding the push failed is exactly the wrong
    call, so this samples real records through the API instead -- which also
    makes the check independent of where Gorse happens to persist (its data
    store is the Docker Postgres, while reco_* live in the host Postgres; these
    are two different databases).

    Two things are checked, and the second matters more:
      1. every sampled train event is present in Gorse
      2. NO sampled test event is present -- if a held-out interaction reached
         the model, every metric computed later is inflated and worthless
    """
    import random

    import httpx
    import psycopg2

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM reco_products")
            n_products = cur.fetchone()[0]
            cur.execute("""SELECT feedback_type, count(*) FROM reco_interactions
                           WHERE split='train' GROUP BY 1 ORDER BY 2 DESC""")
            by_type = cur.fetchall()

            cur.execute("""SELECT user_id FROM reco_interactions WHERE split='train'
                           GROUP BY 1 HAVING count(*) >= 2 LIMIT 4000""")
            train_users = [r[0] for r in cur.fetchall()]
            picked = random.Random(0).sample(train_users, min(sample, len(train_users)))

            cur.execute("""SELECT user_id, product_id FROM reco_interactions
                           WHERE split='test' AND user_id = ANY(%s)""", (picked,))
            test_pairs = cur.fetchall()

            cur.execute("""SELECT user_id, count(*) FROM reco_interactions
                           WHERE split='train' AND user_id = ANY(%s)
                           GROUP BY 1""", (picked,))
            pg_counts = dict(cur.fetchall())
    finally:
        conn.close()

    rule_hdr("GORSE CROSS-CHECK")
    print(f"postgres reco_products      : {n_products:,}")
    print(f"postgres train events       : {sum(n for _, n in by_type):,}  "
          f"{{{', '.join(f'{t}: {n:,}' for t, n in by_type)}}}")

    stats = httpx.get(f"{GORSE_URL}/api/dashboard/stats", timeout=30).json()
    print(f"gorse dashboard NumItems    : {stats.get('NumItems', 0):,}  "
          f"(recomputed on a schedule — stale right after a push)")
    print(f"gorse user labels indexed   : {stats.get('NumUserLabels', 0):,}"
          + ("   <- 0 confirms the trait-sync bug is still live"
             if not stats.get("NumUserLabels") else ""))

    missing_train, wrong_count, leaked = 0, 0, 0
    for uid in picked:
        try:
            got = httpx.get(f"{GORSE_URL}/api/user/{uid}/feedback", timeout=30).json()
        except Exception:
            missing_train += 1
            continue
        in_gorse = {(f["ItemId"]) for f in (got or [])}
        if len(in_gorse) < pg_counts.get(uid, 0):
            wrong_count += 1
        for u2, pid in test_pairs:
            if u2 == uid and pid in in_gorse:
                leaked += 1

    print()
    print(f"sampled {len(picked)} users with >= 2 training events:")
    print(f"  users unreachable in Gorse       : {missing_train}")
    print(f"  users with fewer events in Gorse : {wrong_count}")
    print(f"  HELD-OUT test events found       : {leaked}")
    ok = missing_train == 0 and wrong_count == 0 and leaked == 0
    print()
    print("PASS: train data present, no test leakage into Gorse." if ok
          else "FAIL: see the counts above before trusting any metric.")


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

    rule_hdr("FULL CORPUS")
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
    rule_hdr(f"CATALOGUE-RESTRICTED  (top {size_label:,} rag_products by rating_count)")
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

    rule_hdr("WHAT THIS MEANS FOR THE EVALUATION")
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
    p.add_argument("--build", action="store_true",
                   help="write reco_products + reco_interactions to Postgres")
    p.add_argument("--dry-run", action="store_true",
                   help="with --build: compute and report, write nothing")
    p.add_argument("--split", choices=("temporal", "leave-last-out"),
                   default="temporal",
                   help="temporal (default): one global cutoff, keeps item "
                        "signal. leave-last-out: per-user, maximises the warm "
                        "cohort but starves 81%% of the catalogue")
    p.add_argument("--split-quantile", type=float, default=0.70,
                   help="with --split temporal: where to cut (default 0.70, "
                        "which peaks the evaluable warm cohort)")
    p.add_argument("--push-gorse", action="store_true",
                   help="load reco_products + the TRAIN split into Gorse")
    p.add_argument("--cohort-sample", type=int, default=None, metavar="N",
                   help="with --push-gorse: push every evaluable user (those "
                        "with both train and test events) plus N randomly "
                        "sampled other train users, instead of all 371K. "
                        "Materialising a per-user cache for every user does "
                        "not fit the memory budget; see --cohort-report")
    p.add_argument("--cohort-seed", type=int, default=20260831,
                   help="RNG seed for --cohort-sample, so the pushed cohort is "
                        "reproducible across runs (default: 20260831)")
    p.add_argument("--cohort-report", action="store_true",
                   help="print the evaluable-user strata and what each "
                        "--cohort-sample size would cost, then exit")
    p.add_argument("--skip-items", action="store_true",
                   help="with --push-gorse: push feedback only, leaving the "
                        "catalogue as it is. --reset-gorse keeps items, so a "
                        "re-push with a different cohort re-sends 95K unchanged "
                        "items for nothing -- and each one costs a separate "
                        "Redis round-trip inside Gorse (server/rest.go:1240), "
                        "so it dominates the run")
    p.add_argument("--reset-gorse", action="store_true",
                   help="with --push-gorse: clear Gorse's users and feedback "
                        "first. REQUIRED when shrinking the cohort -- pushing "
                        "is an upsert, so a smaller cohort otherwise leaves "
                        "every previously-pushed user in place and still "
                        "caching. Items are kept: the catalogue is the "
                        "denominator for coverage and does not shrink")
    p.add_argument("--verify-gorse", action="store_true",
                   help="cross-check Gorse's counters against Postgres")
    p.add_argument("--sweep-catalogue", metavar="N,N,...",
                   help="report the evaluable warm cohort at each catalogue "
                        "size, e.g. 1000,5000,20000,50000,100000,all")
    p.add_argument("--catalogue-size", type=int, default=None,
                   help="restrict the join to the top-N rag_products by "
                        "rating_count (default: the whole table)")
    p.add_argument("--refresh", action="store_true",
                   help="re-parse the raw dump instead of using the cache")
    p.add_argument("--limit", type=int, default=None,
                   help="only read the first N lines (smoke test; not cached)")
    args = p.parse_args()

    if not (args.stats or args.sweep_catalogue or args.build
            or args.push_gorse or args.verify_gorse or args.cohort_report):
        p.error("pass at least one of --stats, --sweep-catalogue, --build, "
                "--push-gorse, --verify-gorse, --cohort-report")

    if args.cohort_report:
        cohort_report()
        return

    if args.stats:
        run_stats(args)

    if args.build:
        build(args, load_events(refresh=False))

    if args.push_gorse:
        push_gorse(args)

    if args.verify_gorse:
        verify_gorse()

    if args.sweep_catalogue:
        table = load_events(refresh=False)
        i_col = table.column("product_id").combine_chunks()
        u_all = (table.column("user_id").combine_chunks()
                 .indices.to_numpy().astype(np.int64))
        i_all = i_col.indices.to_numpy().astype(np.int64)
        keep = dedupe_edges(u_all, i_all)
        sweep(args, u_all[keep], i_all[keep], i_col.dictionary.to_pylist())


if __name__ == "__main__":
    main()
