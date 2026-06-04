# One-time Kafka topic setup script. Run this before starting the consumer.
#
# Creates:
#   rag-ingest       2 partitions — derived from SLA + throughput (see comment below)
#   rag-ingest-dlq   1 partition  — dead letter queue; low volume, no parallelism needed
#
# WHY 2 PARTITIONS (not 4, not 1):
#   Throughput per consumer  : ~60 msg/min (dominated by OpenAI embed latency ~1s/product)
#   Peak demand              : 500 products in one seasonal launch
#   SLA                      : products searchable within 15 min of being added
#   Consumers needed         : ceil(500/15 / 60) = ceil(0.56) = 1
#   Future-proofed (2×)      : 2 partitions
#
#   4 partitions would only pay off if throughput per consumer jumped 4× (e.g. local GPU
#   embeddings) or peak demand grew to 2,000+ products per launch.
#
#   IMPORTANT: partition count cannot be reduced after creation without deleting and
#   recreating the topic (all unprocessed messages are lost). Set it intentionally.
#
# Usage: python data/setup_kafka_topics.py [--bootstrap localhost:9092]
#
# Uses aiokafka (same library as the consumer) — kafka-python is not in requirements.

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOPICS = [
    NewTopic(
        name="rag-ingest",
        num_partitions=2,
        replication_factor=1,   # single broker for local dev; raise to 3 in prod
    ),
    NewTopic(
        name="rag-ingest-dlq",
        num_partitions=1,
        replication_factor=1,
    ),
]


async def setup(bootstrap: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        results = await admin.create_topics(TOPICS)
        for topic, error_code, error_message in results:
            if error_code == 0:
                logger.info("Topic %-20s created", topic)
            elif error_code == 36:          # TopicExistsException error code
                logger.info("Topic %-20s already exists", topic)
            else:
                logger.error("Topic %-20s FAILED code=%d: %s", topic, error_code, error_message)
    finally:
        await admin.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    args = parser.parse_args()
    asyncio.run(setup(args.bootstrap))
