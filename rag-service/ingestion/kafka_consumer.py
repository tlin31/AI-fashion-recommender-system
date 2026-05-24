# Async Kafka consumer for the ingestion write path. Decouples product/review
# ingest events from the query path by consuming from KAFKA_INGEST_TOPIC and
# running chunk → embed → Milvus upsert for each event.

from __future__ import annotations

import hashlib
import json
import os

from kafka import KafkaConsumer

from ingestion.chunker import chunk_description, chunk_review
from ingestion.embedder import embed_batch
from ingestion.indexer import upsert_chunks

TOPIC = os.environ.get("KAFKA_INGEST_TOPIC", "rag-ingest")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def _make_chunk_id(product_id: str, chunk_type: str, index: int) -> str:
    """Deterministic ID so re-ingesting the same product is idempotent."""
    raw = f"{product_id}:{chunk_type}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:24]


async def start_consumer() -> None:
    """Consume from KAFKA_INGEST_TOPIC and index each event into Milvus.

    Expected message payload (JSON):
        {
            "product_id": str,
            "text": str,
            "chunk_type": "description" | "review",
            "metadata": {
                "price_range": str,
                "category": str,
                "occasion": str,
                "brand": str
            }
        }

    commit() is called only after a successful Milvus upsert — at-least-once
    delivery ensures no events are silently dropped if the upsert fails.
    """
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda b: json.loads(b.decode()),
        enable_auto_commit=False,
        group_id="rag-indexer",
    )

    for message in consumer:
        event = message.value
        product_id = event["product_id"]
        text = event["text"]
        chunk_type = event.get("chunk_type", "description")
        meta = event.get("metadata", {})

        # Chunk the text based on type
        if chunk_type == "review":
            raw_chunks = [chunk_review(text)]
            raw_chunks = [c for c in raw_chunks if c is not None]
        else:
            raw_chunks = chunk_description(text)

        if not raw_chunks:
            consumer.commit()
            continue

        # Embed all chunks for this event in one batch call
        vectors = await embed_batch(raw_chunks)

        chunks = [
            {
                "id": _make_chunk_id(product_id, chunk_type, i),
                "product_id": product_id,
                "vector": vectors[i],
                "chunk_type": chunk_type,
                "text": raw_chunks[i],
                "price_range": meta.get("price_range", ""),
                "category": meta.get("category", ""),
                "occasion": meta.get("occasion", ""),
                "brand": meta.get("brand", ""),
            }
            for i in range(len(raw_chunks))
        ]

        await upsert_chunks(chunks)
        # Commit offset only after successful upsert (at-least-once delivery)
        consumer.commit()
