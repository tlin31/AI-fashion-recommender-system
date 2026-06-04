# Async Kafka consumer for the ingestion write path.
#
# Production hardening applied:
#   1. aiokafka (truly async) — kafka-python's sync iterator blocked the event loop
#   2. Dead Letter Queue — poison pills are forwarded to rag-ingest-dlq after MAX_RETRIES
#      and committed so the topic never stalls on a single bad message
#   3. max_poll_interval_ms tuned up + max_poll_records kept small so Kafka never
#      declares the consumer dead during slow OpenAI embedding calls
#   4. Consumer lag logged after every commit so silent falling-behind is visible
#   5. Topic setup script (see data/setup_kafka_topics.py) creates rag-ingest with
#      4 partitions so multiple consumer instances can run in parallel

from __future__ import annotations

import hashlib
import json
import logging
import os

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from ingestion.chunker import chunk_description, chunk_review
from ingestion.embedder import embed_batch
from ingestion.indexer import upsert_chunks

logger = logging.getLogger(__name__)

TOPIC = os.environ.get("KAFKA_INGEST_TOPIC", "rag-ingest")
DLQ_TOPIC = f"{TOPIC}-dlq"          # dead letter queue for unprocessable messages
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
GROUP_ID = "rag-indexer"
MAX_RETRIES = 3                      # attempts before a message is sent to DLQ


def _make_chunk_id(product_id: str, chunk_type: str, index: int) -> str:
    """Deterministic ID so re-ingesting the same product is idempotent."""
    raw = f"{product_id}:{chunk_type}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:24]


async def _process_event(event: dict) -> None:
    """Chunk → embed → upsert one ingest event. Raises on any failure."""
    product_id = event["product_id"]           # KeyError here → caught by caller
    text = event["text"]
    chunk_type = event.get("chunk_type", "description")
    meta = event.get("metadata", {})

    if chunk_type == "review":
        raw_chunks = [chunk_review(text)]
        raw_chunks = [c for c in raw_chunks if c is not None]
    else:
        raw_chunks = chunk_description(text)

    if not raw_chunks:
        logger.warning("product_id=%s produced no chunks — skipping", product_id)
        return

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
    logger.info("product_id=%s indexed %d chunks", product_id, len(chunks))


async def _log_consumer_lag(consumer: AIOKafkaConsumer) -> None:
    """Log per-partition consumer lag after each successful commit.

    Lag = log-end-offset − committed-offset. Persistent lag > 500 for more than
    a few minutes in production should page the on-call engineer (wire into
    Prometheus/Grafana via a metrics exporter rather than logging in prod).
    """
    try:
        partitions = consumer.assignment()
        end_offsets = await consumer.end_offsets(partitions)
        for tp, end in end_offsets.items():
            committed = await consumer.committed(tp)
            lag = end - (committed or 0)
            logger.info("consumer_lag partition=%s-%d lag=%d", tp.topic, tp.partition, lag)
    except KafkaError as exc:
        # Non-fatal — lag logging should never crash the consumer
        logger.warning("Could not read consumer lag: %s", exc)


async def start_consumer() -> None:
    """Consume from KAFKA_INGEST_TOPIC and index each event into Milvus.

    Fix 1 — aiokafka replaces kafka-python so `async for` truly yields control
    between messages, keeping the FastAPI event loop responsive.

    Fix 2 — messages that fail MAX_RETRIES times are published to DLQ_TOPIC and
    committed so downstream messages are never blocked.

    Fix 3 — max_poll_interval_ms=600_000 (10 min) prevents Kafka from declaring
    the consumer dead during slow OpenAI calls; max_poll_records=10 keeps each
    poll batch small so we re-poll frequently enough to hold the session.

    Fix 4 — consumer lag is logged after every commit.

    Fix 5 — run multiple instances of this consumer against a multi-partition
    topic (see data/setup_kafka_topics.py) for parallel ingest throughput.
    """
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        value_deserializer=lambda b: json.loads(b.decode()),
        # Fix 3: give the consumer 10 minutes between polls before Kafka
        # considers it dead. Default is 5 min — too tight for OpenAI p99.
        max_poll_interval_ms=600_000,
        # Process at most 10 messages per poll so we re-poll frequently
        # even during heavy embedding work.
        max_poll_records=10,
    )

    # Fix 2: DLQ producer — started alongside the consumer
    dlq_producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode(),
    )

    await consumer.start()
    await dlq_producer.start()
    logger.info("Kafka consumer started. topic=%s group=%s dlq=%s", TOPIC, GROUP_ID, DLQ_TOPIC)

    retry_counts: dict[str, int] = {}   # key: "{topic}-{partition}-{offset}"

    try:
        async for message in consumer:
            msg_key = f"{message.topic}-{message.partition}-{message.offset}"

            # Safely deserialize — value_deserializer already ran; guard field access
            event = message.value
            if not isinstance(event, dict):
                logger.error("Unparseable message at offset %d — sending to DLQ", message.offset)
                await dlq_producer.send(DLQ_TOPIC, value={"raw": str(message.value), "error": "not a dict"})
                await consumer.commit()
                continue

            attempt = retry_counts.get(msg_key, 0) + 1

            try:
                await _process_event(event)
                retry_counts.pop(msg_key, None)   # clear on success
                await consumer.commit()
                await _log_consumer_lag(consumer)  # Fix 4: log lag after each commit

            except Exception as exc:
                logger.warning(
                    "Processing failed (attempt %d/%d) product_id=%s error=%s",
                    attempt, MAX_RETRIES,
                    event.get("product_id", "unknown"),
                    exc,
                )

                if attempt >= MAX_RETRIES:
                    # Fix 2: give up — send to DLQ and move on so the topic
                    # never stalls on a single bad message
                    logger.error(
                        "MAX_RETRIES exceeded for product_id=%s — publishing to DLQ %s",
                        event.get("product_id", "unknown"),
                        DLQ_TOPIC,
                    )
                    await dlq_producer.send(DLQ_TOPIC, value={
                        "original_event": event,
                        "error": str(exc),
                        "attempts": attempt,
                    })
                    retry_counts.pop(msg_key, None)
                    await consumer.commit()       # skip it, unblock the topic
                else:
                    retry_counts[msg_key] = attempt
                    # Do NOT commit — Kafka will re-deliver this message after
                    # the consumer rejoins or on the next poll cycle

    finally:
        await consumer.stop()
        await dlq_producer.stop()
        logger.info("Kafka consumer stopped.")
