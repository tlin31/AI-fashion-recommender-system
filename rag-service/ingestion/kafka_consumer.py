# Async Kafka consumer for the ingestion write path. Decouples product/review
# ingest events from the query path by consuming from KAFKA_INGEST_TOPIC and
# running chunk → embed → Milvus upsert for each event.

from __future__ import annotations


async def start_consumer() -> None:
    # Consumes from KAFKA_INGEST_TOPIC.
    # Each message: raw product/review event → chunk → embed → upsert to Milvus.
    # Write path is fully decoupled from the query path.
    raise NotImplementedError
