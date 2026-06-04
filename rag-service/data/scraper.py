# Loads McAuley 2023 Amazon Fashion *reviews* JSONL, joins to rag_products by
# parent_asin, deduplicates, and writes clean review text to rag_reviews.
#
# Reviews source: McAuley-Lab/Amazon-Reviews-2023
#   raw/review_categories/Amazon_Fashion.jsonl
#   Fields used: parent_asin, text, title, rating
#
# Run:
#   python data/scraper.py --reviews /path/to/Amazon_Fashion.jsonl
#
# If --reviews is omitted the script exits cleanly — the pipeline works without
# reviews (product descriptions alone are sufficient for RAG retrieval).

from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ftfy
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL     = os.environ.get("POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse")
MIN_WORDS        = 20      # drop reviews shorter than this many tokens
MAX_PER_PRODUCT  = 50      # keep at most this many reviews per product after dedup
MAX_OVERLAP      = 0.30    # Jaccard threshold — drop review if >30% overlap with a kept one
CHUNK_SIZE       = 100_000 # read JSONL in chunks to avoid loading multi-GB files at once

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_reviews (
    id          SERIAL PRIMARY KEY,
    product_id  TEXT NOT NULL REFERENCES rag_products(product_id),
    text        TEXT NOT NULL,
    rating      SMALLINT,
    source      TEXT DEFAULT 'amazon',
    created_at  TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS rag_reviews_product_id_idx ON rag_reviews (product_id);
"""

UPSERT_SQL = """
INSERT INTO rag_reviews (product_id, text, rating, source)
VALUES (%(product_id)s, %(text)s, %(rating)s, %(source)s)
ON CONFLICT DO NOTHING;
"""


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _clean(text: str) -> str:
    """Fix encoding, strip HTML/Markdown, collapse whitespace."""
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedup(reviews: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    """Drop reviews with Jaccard overlap > MAX_OVERLAP against any already-kept review."""
    kept: list[tuple[str, int | None]] = []
    for candidate in reviews:
        if all(_jaccard(candidate[0], k[0]) <= MAX_OVERLAP for k in kept):
            kept.append(candidate)
        if len(kept) >= MAX_PER_PRODUCT:
            break
    return kept


def load_product_ids() -> set[str]:
    """Return the set of product_ids currently in rag_products."""
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT product_id FROM rag_products")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def process_reviews(reviews_jsonl: Path, product_ids: set[str]) -> dict[str, list[tuple[str, int | None]]]:
    """
    Stream the reviews JSONL in chunks.  For each review:
      - skip if parent_asin not in rag_products
      - clean and length-filter the text
      - accumulate up to MAX_PER_PRODUCT * 4 candidates per product
        (we'll dedup down to MAX_PER_PRODUCT afterwards)

    Returns: {product_id: [(clean_text, rating), ...]} — pre-dedup
    """
    per_product: dict[str, list[tuple[str, int | None]]] = {pid: [] for pid in product_ids}
    total_read = matched = skipped_short = 0
    t0 = time.time()

    file_mb = reviews_jsonl.stat().st_size / 1_048_576
    print(f"[{_ts()}] Streaming {reviews_jsonl.name}  ({file_mb:,.0f} MB, chunk={CHUNK_SIZE:,} rows) ...")
    print(f"  Catalog size: {len(product_ids):,} products to match against")
    print()

    for chunk_n, chunk in enumerate(pd.read_json(reviews_jsonl, lines=True, chunksize=CHUNK_SIZE), 1):
        total_read += len(chunk)

        # Normalise column names
        chunk.columns = chunk.columns.str.lower().str.strip()

        # Resolve which column has the review body
        text_col = next(
            (c for c in ["text", "reviewtext", "body", "review_text"] if c in chunk.columns),
            None,
        )
        if text_col is None:
            print(f"  WARNING: no review text column found in chunk (columns: {list(chunk.columns)})")
            continue

        # parent_asin is the join key to rag_products.product_id
        if "parent_asin" not in chunk.columns:
            print("  WARNING: parent_asin column missing — skipping chunk")
            continue

        chunk = chunk.dropna(subset=["parent_asin", text_col])
        in_catalog = chunk[chunk["parent_asin"].isin(product_ids)]

        if in_catalog.empty:
            elapsed = time.time() - t0
            rate = total_read / elapsed if elapsed > 0 else 0
            print(f"  [{_ts()}] Chunk {chunk_n}: {total_read:,} rows read  "
                  f"({rate:,.0f} rows/s)  matched={matched:,}  0 hits this chunk")
            continue

        chunk_matched = 0
        for _, row in in_catalog.iterrows():
            pid = str(row["parent_asin"]).strip()
            raw = str(row.get(text_col, ""))

            # Optionally prepend review title for extra context
            title = str(row.get("title", "")).strip()
            if title and title.lower() not in raw.lower():
                raw = f"{title}. {raw}"

            clean = _clean(raw)
            if len(clean.split()) < MIN_WORDS:
                skipped_short += 1
                continue

            rating_raw = row.get("rating") or row.get("overall")
            try:
                rating = int(float(rating_raw)) if rating_raw is not None else None
            except (ValueError, TypeError):
                rating = None

            # Cap candidates at 4× MAX_PER_PRODUCT to keep dedup tractable
            if len(per_product[pid]) < MAX_PER_PRODUCT * 4:
                per_product[pid].append((clean, rating))
                matched += 1
                chunk_matched += 1

        elapsed = time.time() - t0
        rate = total_read / elapsed if elapsed > 0 else 0
        products_covered = sum(1 for v in per_product.values() if v)
        print(f"  [{_ts()}] Chunk {chunk_n}: {total_read:,} rows read  "
              f"({rate:,.0f} rows/s)  "
              f"matched={matched:,}  +{chunk_matched} this chunk  "
              f"products covered={products_covered:,}/{len(product_ids):,}")

    elapsed = time.time() - t0
    products_covered = sum(1 for v in per_product.values() if v)
    print(f"\n[{_ts()}] Streaming done in {elapsed:.0f}s")
    print(f"  Total read   : {total_read:,} reviews")
    print(f"  Matched      : {matched:,} reviews ({skipped_short:,} dropped — too short)")
    print(f"  Products hit : {products_covered:,} / {len(product_ids):,}")
    return per_product


def build_records(per_product: dict[str, list[tuple[str, int | None]]]) -> list[dict]:
    """Deduplicate and flatten to insert records."""
    records: list[dict] = []
    products_with_reviews = 0
    total_products = sum(1 for v in per_product.values() if v)
    done = 0
    t0 = time.time()

    print(f"\n[{_ts()}] Deduplicating reviews for {total_products:,} products ...")
    for pid, candidates in per_product.items():
        if not candidates:
            continue
        deduped = _dedup(candidates)
        products_with_reviews += 1
        done += 1
        for text, rating in deduped:
            records.append({"product_id": pid, "text": text, "rating": rating, "source": "amazon"})

        # Log every 500 products
        if done % 500 == 0 or done == total_products:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_products - done) / rate if rate > 0 else 0
            print(f"  [{_ts()}] {done:,}/{total_products:,} products deduped  "
                  f"({rate:.0f}/s  ETA {eta:.0f}s)  reviews kept so far: {len(records):,}")

    print(f"\n[{_ts()}] Dedup complete: {len(records):,} reviews kept across "
          f"{products_with_reviews:,} products")
    return records


def write_to_postgres(records: list[dict]) -> int:
    total = len(records)
    print(f"\n[{_ts()}] Writing {total:,} reviews to rag_reviews ...")
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                # Write in batches so we can log progress
                WRITE_BATCH = 2000
                written = 0
                for i in range(0, total, WRITE_BATCH):
                    batch = records[i:i + WRITE_BATCH]
                    psycopg2.extras.execute_batch(cur, UPSERT_SQL, batch, page_size=500)
                    written += len(batch)
                    print(f"  [{_ts()}] Written {written:,}/{total:,} reviews")
        print(f"[{_ts()}] ✓ All {total:,} reviews inserted into rag_reviews.")
        return total
    finally:
        conn.close()


def run(reviews_jsonl: Path | None) -> None:
    if reviews_jsonl is None or not reviews_jsonl.exists():
        print("No reviews JSONL provided or file not found — skipping.")
        print("To add reviews, download Amazon_Fashion.jsonl from:")
        print("  huggingface_hub.hf_hub_download(")
        print('    repo_id="McAuley-Lab/Amazon-Reviews-2023",')
        print('    filename="raw/review_categories/Amazon_Fashion.jsonl",')
        print('    repo_type="dataset")')
        return

    t_start = time.time()
    print(f"[{_ts()}] Loading product IDs from rag_products ...")
    product_ids = load_product_ids()
    print(f"  {len(product_ids):,} products in catalog\n")

    per_product = process_reviews(reviews_jsonl, product_ids)
    records = build_records(per_product)

    if records:
        write_to_postgres(records)
        total_elapsed = time.time() - t_start
        print(f"\n[{_ts()}] ✓ Done in {total_elapsed:.0f}s total.")
    else:
        print("No reviews matched the current catalog — nothing written.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=None,
        help="Path to Amazon_Fashion.jsonl (reviews, not metadata)",
    )
    args = parser.parse_args()
    run(args.reviews)
