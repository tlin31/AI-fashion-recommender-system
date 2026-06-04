# Lightweight review scraper to augment the Kaggle catalog with ~500-1000 real reviews.
# Not load-bearing — the pipeline works on product descriptions alone if scraping fails.
# Run manually: python data/scraper.py
#
# Strategy: use the Kaggle "Amazon Fashion Reviews" dataset as a clean fallback
# rather than live scraping, which risks rate-limiting and ToS issues.
# Match reviews to catalog products by product name keyword overlap.

from __future__ import annotations

import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ftfy
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")
MIN_WORDS = 20
MAX_OVERLAP_RATIO = 0.30  # drop review if >30% token overlap with another for the same product

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_reviews (
    id          SERIAL PRIMARY KEY,
    product_id  TEXT NOT NULL REFERENCES rag_products(product_id),
    text        TEXT NOT NULL,
    source      TEXT DEFAULT 'kaggle',
    created_at  TIMESTAMP DEFAULT NOW()
);
"""

INSERT_SQL = """
INSERT INTO rag_reviews (product_id, text, source)
VALUES (%(product_id)s, %(text)s, %(source)s)
ON CONFLICT DO NOTHING;
"""


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap between token sets of two strings."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicate(reviews: list[str]) -> list[str]:
    """Drop reviews with >MAX_OVERLAP_RATIO token overlap against any already-kept review."""
    kept: list[str] = []
    for candidate in reviews:
        if all(_token_overlap(candidate, k) <= MAX_OVERLAP_RATIO for k in kept):
            kept.append(candidate)
    return kept


def scrape_reviews(
    product_ids: list[str],
    reviews_csv: str | None = None,
) -> list[dict]:
    """Load reviews from a Kaggle CSV and match them to catalog products.

    Args:
        product_ids: list of product_id strings from rag_products.
        reviews_csv: path to the Amazon Fashion reviews CSV.
                     Expected columns: reviewText, summary, asin or productTitle.

    Returns:
        list of dicts with keys: product_id, text, source.
    """
    if reviews_csv is None or not os.path.exists(reviews_csv):
        print("No reviews CSV provided or file not found — skipping review scraping.")
        print("To add reviews: download an Amazon Fashion reviews dataset from Kaggle")
        print("and pass its path as reviews_csv.")
        return []

    print(f"Reading {reviews_csv}...")
    df = pd.read_csv(reviews_csv, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Use reviewText or body column
    text_col = next((c for c in ["reviewtext", "body", "review_text", "text"] if c in df.columns), None)
    if text_col is None:
        print(f"Could not find review text column in {reviews_csv}. Columns: {list(df.columns)}")
        return []

    df = df.dropna(subset=[text_col])
    print(f"  After dropna: {len(df):,} rows")

    # Cleaning pipeline (order matters):
    #   1. ftfy          - fix mojibake before anything else touches the bytes
    #   2. html.unescape - decode HTML entities (&#34; -> ", &amp; -> &)
    #   3. strip HTML    - remove <br />, <p>, etc. left from review formatting
    #   4. strip Markdown emphasis (*word* -> word)
    #   5. collapse whitespace
    def _clean(t: str) -> str:
        t = ftfy.fix_text(t)
        t = html.unescape(t)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\*+([^*]+)\*+", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    df["clean_text"] = df[text_col].astype(str).apply(_clean)

    # Drop reviews below minimum word count
    df = df[df["clean_text"].str.split().str.len() >= MIN_WORDS]
    print(f"  After MIN_WORDS={MIN_WORDS} filter: {len(df):,} reviews to match")

    # Simple matching: assign each review to the first product_id whose name
    # appears as a keyword in the review. For small catalogs this is fast enough.
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT product_id, name FROM rag_products WHERE product_id = ANY(%s)", (product_ids,))
            catalog = {pid: name.lower() for pid, name in cur.fetchall()}
    finally:
        conn.close()

    results: list[dict] = []
    per_product_texts: dict[str, list[str]] = {pid: [] for pid in catalog}

    # Build inverted index: keyword → [product_ids that contain it]
    #
    # Two-sided discriminativeness filter (both must pass):
    #
    #   Product-side  (MAX_KEYWORD_PRODUCTS): drop keywords shared by many products.
    #                 "blue" appears in 800 product names → useless for matching.
    #
    #   Review-side   (MAX_REVIEW_FREQUENCY): drop keywords that appear in many reviews.
    #                 "pocket" appears in 30k reviews but only 1 product name → passes
    #                 the product-side filter but causes 1,700 false matches.
    #                 A keyword appearing in >2% of reviews is too generic to be
    #                 discriminative regardless of how specific it is to a product.
    #
    # Together these approximate TF-IDF: keywords rare in both the product catalog
    # AND the review corpus carry the most signal.
    MAX_KEYWORD_PRODUCTS = 20
    MAX_REVIEW_FREQUENCY = int(len(df) * 0.02)  # 2% of review corpus

    # Count how many reviews each keyword appears in (document frequency)
    print("  Computing review-side keyword frequencies...")
    review_word_counts: dict[str, int] = {}
    for text in df["clean_text"]:
        for word in set(text.lower().split()):   # set() counts once per review
            if len(word) > 4:
                review_word_counts[word] = review_word_counts.get(word, 0) + 1

    raw_index: dict[str, list[str]] = {}
    for pid, name in catalog.items():
        for word in name.split():
            if len(word) > 4:
                raw_index.setdefault(word, []).append(pid)

    keyword_to_pids = {
        kw: pids for kw, pids in raw_index.items()
        if len(pids) <= MAX_KEYWORD_PRODUCTS                          # product-side
        and review_word_counts.get(kw, 0) <= MAX_REVIEW_FREQUENCY    # review-side
    }
    dropped_product_side = sum(1 for kw, pids in raw_index.items() if len(pids) > MAX_KEYWORD_PRODUCTS)
    dropped_review_side  = sum(1 for kw, pids in raw_index.items()
                               if len(pids) <= MAX_KEYWORD_PRODUCTS
                               and review_word_counts.get(kw, 0) > MAX_REVIEW_FREQUENCY)
    print(f"  Inverted index: {len(keyword_to_pids)} discriminative keywords "
          f"(dropped {dropped_product_side} high product-fanout, "
          f"{dropped_review_side} high review-frequency like 'pocket'/'pants')")

    total = len(df)
    matched = 0
    for i, text in enumerate(df["clean_text"]):
        if i % 100_000 == 0:
            print(f"  Matching: {i:,}/{total:,} reviews processed, {matched} matched so far...")

        words = set(text.lower().split())

        # Count how many product-name keywords appear in this review per product.
        # Assigning to the highest-hit product (rather than first-in-dict-order)
        # is strictly more accurate: "Biba floral kurta" review scores higher on
        # the "Biba Floral Kurta" product than on a generic "kurta" product.
        hit_counts: dict[str, int] = {}
        for word in words:
            if word in keyword_to_pids:
                for pid in keyword_to_pids[word]:
                    hit_counts[pid] = hit_counts.get(pid, 0) + 1

        # Require at least 2 keyword hits before assigning.
        # A single generic keyword hit ("pocket", "cotton") is coincidence;
        # 2+ hits from the same product name is genuine signal.
        strong_hits = {pid: count for pid, count in hit_counts.items() if count >= 2}
        if strong_hits:
            best_pid = max(strong_hits, key=lambda p: strong_hits[p])
            per_product_texts[best_pid].append(text)
            matched += 1

    n_products_matched = sum(1 for t in per_product_texts.values() if t)
    print(f"  Matching complete: {matched:,} reviews matched to {n_products_matched} products")

    # Cap reviews per product before deduplication.
    # Without this, a product matching "kurta" accumulates tens of thousands of reviews
    # and the O(n²) Jaccard dedup takes hours. 200 candidates is enough signal;
    # we'll keep ~50 after deduplication anyway.
    MAX_CANDIDATES = 200
    for pid, texts in per_product_texts.items():
        if not texts:
            continue
        candidates = texts[:MAX_CANDIDATES]
        deduped = _deduplicate(candidates)
        print(f"  product {pid}: {len(texts):,} matched → {len(candidates)} sampled → {len(deduped)} kept after dedup")
        for text in deduped:
            results.append({"product_id": pid, "text": text, "source": "kaggle"})

    return results


def write_to_postgres(reviews: list[dict]) -> int:
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.executemany(INSERT_SQL, reviews)
        print(f"Inserted {len(reviews)} reviews into rag_reviews.")
        return len(reviews)
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to Amazon Fashion reviews CSV")
    args = parser.parse_args()

    # Use all product IDs in catalog
    conn = psycopg2.connect(POSTGRES_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT product_id FROM rag_products LIMIT 5000")
        ids = [r[0] for r in cur.fetchall()]
    conn.close()

    reviews = scrape_reviews(ids, reviews_csv=args.csv)
    if reviews:
        write_to_postgres(reviews)
    else:
        print("No reviews to write.")
