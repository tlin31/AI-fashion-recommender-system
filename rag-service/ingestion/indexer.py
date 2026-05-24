# Manages the Milvus collection: creates it with the required schema on first run
# and upserts embedded chunks with their metadata fields. Schema changes here
# require a full collection rebuild and re-embed.

from __future__ import annotations

import os

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "fashion_rag")
VECTOR_DIM = 1536  # text-embedding-3-small output dimension
_BATCH_SIZE = 1000  # Milvus recommended max per upsert call


def _connect() -> None:
    connections.connect(
        host=os.environ.get("MILVUS_HOST", "localhost"),
        port=int(os.environ.get("MILVUS_PORT", 19530)),
    )


async def build_milvus_collection() -> None:
    """Create the collection if it doesn't exist.

    Safe to call on every startup — no-ops when the collection is already present.
    Metadata fields (price_range, category, occasion, brand) are indexed from day one
    so Phase 2 multi-constraint pre-filtering drops in without a schema rebuild.
    """
    _connect()

    if utility.has_collection(COLLECTION_NAME):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64),
        # "description" or "review"
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        # Metadata fields — indexed for pre-filter queries at retrieval time
        FieldSchema(name="price_range", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="occasion", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields=fields, description="Fashion RAG product chunks")
    col = Collection(name=COLLECTION_NAME, schema=schema)

    # HNSW: append-friendly (no rebuild on insert), strong recall at moderate ef values.
    # IVF_FLAT would need periodic retraining as the catalog grows.
    col.create_index(
        field_name="vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.load()


async def upsert_chunks(chunks: list[dict]) -> int:
    """Upsert chunk vectors and metadata into Milvus.

    Each chunk dict must contain:
        id, product_id, vector, chunk_type, text,
        price_range, category, occasion, brand

    Uses deterministic IDs (product_id + chunk_index) so re-running the ingest
    script is idempotent — existing chunks are overwritten, not duplicated.
    Returns the number of chunks upserted.
    """
    _connect()
    col = Collection(COLLECTION_NAME)

    total = 0
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        data = [
            [c["id"] for c in batch],
            [c["vector"] for c in batch],
            [c["product_id"] for c in batch],
            [c["chunk_type"] for c in batch],
            [c["text"][:2048] for c in batch],  # guard against oversized text
            [c.get("price_range", "") for c in batch],
            [c.get("category", "") for c in batch],
            [c.get("occasion", "") for c in batch],
            [c.get("brand", "") for c in batch],
        ]
        col.upsert(data)
        total += len(batch)

    col.flush()
    return total
