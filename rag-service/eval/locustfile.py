# Locust load test definition. Simulates concurrent users hitting POST /query with
# representative fashion queries to document the real throughput degradation point.
#
# Query source: all 100 queries from eval/golden_queries.json, stratified across
# navigational / attribute / exploratory / edge-case types. Using the full golden
# set (rather than a handful of fixed strings) prevents Redis cache inflation —
# with only 5 fixed queries every request after the first few would be a cache hit,
# producing artificially fast numbers that hide real pipeline bottlenecks.
#
# Run each concurrency level for 3 minutes and record p50/p95/p99:
#   locust -f eval/locustfile.py --host http://localhost:8002 \
#     --users 1  --spawn-rate 1  --run-time 3m --headless --csv eval/load_test_u1
#   locust -f eval/locustfile.py --host http://localhost:8002 \
#     --users 5  --spawn-rate 5  --run-time 3m --headless --csv eval/load_test_u5
#   locust -f eval/locustfile.py --host http://localhost:8002 \
#     --users 10 --spawn-rate 10 --run-time 3m --headless --csv eval/load_test_u10
#   locust -f eval/locustfile.py --host http://localhost:8002 \
#     --users 20 --spawn-rate 20 --run-time 3m --headless --csv eval/load_test_u20

from __future__ import annotations

import json
import pathlib
import random

from locust import HttpUser, between, task

_golden_path = pathlib.Path(__file__).parent / "golden_queries.json"
SAMPLE_QUERIES: list[str] = [
    q["query"]
    for q in json.loads(_golden_path.read_text())
]


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query(self) -> None:
        self.client.post("/query", json={"query": random.choice(SAMPLE_QUERIES)})
