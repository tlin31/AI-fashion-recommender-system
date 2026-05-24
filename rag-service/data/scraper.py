# Lightweight review scraper to augment the Kaggle catalog with ~500-1000 real reviews.
# Not load-bearing — the pipeline works on product descriptions alone if scraping fails.
# Run manually: python data/scraper.py
#
# Strategy: use the Kaggle "Amazon Fashion Reviews" dataset as a clean fallback
# rather than live scraping, which risks rate-limiting and ToS issues.
# Match reviews to catalog products by product name keyword overlap.

from __future__ import annotations

import os
import re

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

    df = pd.read_csv(reviews_csv, low_memory=False)

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Use reviewText or body column
    text_col = next((c for c in ["reviewtext", "body", "review_text", "text"] if c in df.columns), None)
    if text_col is None:
        print(f"Could not find review text column in {reviews_csv}. Columns: {list(df.columns)}")
        return []

    df = df.dropna(subset=[text_col])
    df["clean_text"] = df[text_col].astype(str).apply(
        lambda t: re.sub(r"\s+", " ", t).strip()
    )
    # Drop reviews below minimum word count
    df = df[df["clean_text"].str.split().str.len() >= MIN_WORDS]

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

    for text in df["clean_text"]:
        text_lower = text.lower()
        for pid, name in catalog.items():
            # Match if any word from the product name (>4 chars) appears in the review
            keywords = [w for w in name.split() if len(w) > 4]
            if keywords and any(kw in text_lower for kw in keywords):
                per_product_texts[pid].append(text)
                break  # assign to first matching product only

    for pid, texts in per_product_texts.items():
        deduped = _deduplicate(texts)
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
