# Manages the Milvus collection: creates it with the required schema on first run
# and upserts embedded chunks with their metadata fields. Schema changes here
# require a full collection rebuild and re-embed.

from __future__ import annotations


async def upsert_chunks(chunks: list[dict]) -> int:
    # Upsert chunk vectors + metadata into Milvus collection.
    # Each chunk dict must have: id, product_id, vector, chunk_type,
    # price_range, category, occasion, brand, text.
    # Returns number of chunks inserted.
    raise NotImplementedError


async def build_milvus_collection() -> None:
    # Create collection if it doesn't exist, with the schema defined in CLAUDE.md.
    # Safe to call on every startup — no-ops if collection already exists.
    raise NotImplementedError
