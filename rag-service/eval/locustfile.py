# Locust load test definition. Simulates concurrent users hitting POST /query with
# representative fashion queries to document the real throughput degradation point.

from __future__ import annotations

from locust import HttpUser, between, task


SAMPLE_QUERIES = [
    "blue linen trousers under $80",
    "something minimalist for a job interview",
    "flowy dress for a beach wedding",
    "Nike running shoes size 10",
    "cozy oversized sweater for autumn",
]


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query(self) -> None:
        import random
        self.client.post("/query", json={"query": random.choice(SAMPLE_QUERIES)})
