# One-time script to seed the Milvus index from the rag_products PostgreSQL table.
# Run this after normalize.py has populated the DB. Kafka consumer handles ongoing updates;
# this script is only for the initial bulk load.
# Usage: python data/run_ingest.py [--limit N]

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

from ingestion.chunker import chunk_description
from ingestion.embedder import embed_batch
from ingestion.indexer import build_milvus_collection, upsert_chunks

POSTGRES_URL = os.environ["POSTGRES_URL"]
_EMBED_BATCH = 64  # embed this many chunks per OpenAI call to stay within rate limits


def _chunk_id(product_id: str, index: int) -> str:
    raw = f"{product_id}:description:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:24]


async def run(limit: int | None = None) -> None:
    await build_milvus_collection()

    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            query = """
                SELECT product_id, description, price_range, category, occasion, brand
                FROM rag_products
                WHERE description IS NOT NULL AND length(description) > 30
                ORDER BY rating_count DESC NULLS LAST
            """
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Loaded {len(rows)} products from PostgreSQL.")

    all_chunks: list[dict] = []
    for product_id, description, price_range, category, occasion, brand in rows:
        texts = chunk_description(description)
        for i, text in enumerate(texts):
            all_chunks.append({
                "id": _chunk_id(product_id, i),
                "product_id": product_id,
                "chunk_type": "description",
                "text": text,
                "price_range": price_range or "",
                "category": category or "",
                "occasion": occasion or "",
                "brand": brand or "",
            })

    print(f"Generated {len(all_chunks)} chunks from {len(rows)} products.")

    # Embed in batches to respect OpenAI rate limits
    total_upserted = 0
    for i in range(0, len(all_chunks), _EMBED_BATCH):
        batch = all_chunks[i : i + _EMBED_BATCH]
        texts = [c["text"] for c in batch]
        vectors = await embed_batch(texts)
        for chunk, vector in zip(batch, vectors):
            chunk["vector"] = vector

        upserted = await upsert_chunks(batch)
        total_upserted += upserted

        if (i // _EMBED_BATCH) % 10 == 0:
            print(f"  Progress: {total_upserted}/{len(all_chunks)} chunks indexed...")

    print(f"Done. {total_upserted} chunks upserted into Milvus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max products to ingest (default: all)")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit))
