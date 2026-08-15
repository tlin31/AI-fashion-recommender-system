# AI Fashion Recommender System

A fashion recommendation platform that combines collaborative filtering with LLM-powered
natural-language search. CF handles behavioural signals; a retrieval-augmented search
service handles the semantic and cold-start queries CF cannot answer — evaluated against a
hand-labelled query set with a regression gate, not by inspection.

> Built on **[Gorse](https://github.com/gorse-io/gorse)** (Apache 2.0), an open-source
> recommender engine. The Go engine at the repository root is Gorse; the three subsystems
> below are this project's own work.

---

## Components

| Subsystem | Stack | What it does |
|---|---|---|
| **[rag-service/](rag-service/)** | FastAPI · Milvus · BM25 · GPT-4o-mini | Natural-language product search: guardrail → hybrid retrieval (BM25 + dense, RRF-fused) → CRAG corrective loop → cross-encoder reranker → grounded answer with citations |
| **[python-agent/](python-agent/)** | LangGraph · FastAPI · PostgreSQL | ReAct agent with multi-turn memory and human-in-the-loop approval before any user preference is written |
| **[fashion-recommend/](fashion-recommend/)** | Go · Gin · React · PostgreSQL | Domain API over Gorse — auth, social features, LLM trait extraction, and the customer-facing SPA |

```
React SPA ──┬── fashion-recommend :5001 (Go/Gin) ──┬── Gorse :8088  (CF engine)
            │                                      ├── python-agent :8001 (LangGraph)
            │                                      └── PostgreSQL
            └── rag-service :8002 (FastAPI) ───────┬── Milvus (HNSW) + BM25
                                                   ├── Redis (embedding cache)
                                                   └── OpenAI (embed + generate)
```

---

## What makes it non-trivial

**A real evaluation harness.** 100 golden queries stratified across navigational /
exploratory / attribute / edge cases, with graded relevance judgments (0/1/2) produced by
TREC-style pooling rather than by labelling only what the system happened to return.
NDCG@10 and Recall@10 are gated against a locked baseline in CI.

**Thresholds calibrated, not guessed.** The CRAG routing thresholds were selected via a
50-combination offline grid with 1,000-resample bootstrap validation. A cache-and-replay
design — the thresholds affect routing but never retrieval, so retrieval runs once and every
combination is replayed offline — cut the cost from 12 evaluation runs to ~1.2. The
calibration **confirmed** the existing thresholds rather than improving on them, and is
reported that way.

**Negative results kept.** The CRAG query rewrite degrades retrieval quality on 75 of 100
queries: rephrasing searches the same closed catalogue and introduces no new information,
unlike the web-search fallback in the original CRAG paper. The loop is retained with the
finding documented rather than quietly deleted.

**A metric bug found and fixed.** Duplicate product IDs in ranked lists let six queries
score above the theoretical maximum of 1.0. Fixing it cost 7 points of reported Recall@10;
the baseline was re-measured and re-locked accordingly.

Full write-up: **[rag-service/README.md](rag-service/README.md)**

---

## Results

Retrieval quality over 100 golden queries and 1,481 graded relevance judgments:

| NDCG@10 | Recall@10 |
|---:|---:|
| 0.8468 | 0.6993 |

One caveat stated up front: 89% of pooled candidates were judged relevant, well above the
10–30% typical of TREC pools, so the system is partly measured against a standard it
defined — a strict re-scoring (top grade only) gives NDCG 0.8033 and Recall 0.7785, which
is the check that the conclusion does not rest on a lenient boundary.

Load: 0% failures at 1/5/10/20 concurrent users, p50 3.3 s and 1.81 RPS at 10 users.
Latency is dominated by OpenAI round-trips, not local infrastructure — p50 stays flat as
concurrency rises.

---

## Running it

```bash
cd fashion-recommend && make docker-up    # PostgreSQL, Redis, Milvus, Kafka, Gorse
```

Per-service setup: [rag-service](rag-service/README.md) · [everything else](CLAUDE.md)

Licence: Apache 2.0 — see [LICENSE](LICENSE). Gorse is Apache 2.0, © the Gorse authors.
