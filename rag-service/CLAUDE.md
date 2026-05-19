# rag-service — CLAUDE.md

A standalone FastAPI microservice that adds natural language product search to the fashion
recommendation system. Users submit free-text queries and receive grounded product
recommendations with cited sources. Complements (does not replace) the Gorse CF system:
CF handles behavioral signals, this service handles semantic/cold-start queries.

---

## Common Commands

```bash
cd rag-service

# First-time setup
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY at minimum

# Run the API server (port 8002 — python-agent owns 8001)
uvicorn main:app --reload --port 8002

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_retrieval.py -v

# Seed the BM25 index + Milvus from PostgreSQL (required before first query)
python ingestion/indexer.py --seed

# Run the eval harness against the golden query set
python eval/run_eval.py --golden-set eval/golden_queries.json

# Check regression against stored baseline
python eval/check_regression.py --threshold 0.05

# Load test
locust -f eval/locustfile.py --host http://localhost:8002
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Used for `text-embedding-3-small` embeddings + GPT-4o-mini generation + Ragas LLM-as-judge |
| `MILVUS_HOST` | `localhost` | Milvus host (existing Docker Compose service) |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `fashion_rag` | Milvus collection name — never change after index is built without a full re-embed |
| `POSTGRES_URL` | `postgresql://gorse:gorse_pass@localhost:5432/gorse` | Reuses existing DB; BM25 index built from `products` table at startup |
| `REDIS_URL` | `redis://localhost:6379` | TTL cache for frequent RAG query results |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Async ingestion write path |
| `PINECONE_API_KEY` | — | Phase 2 only — leave unset in Phase 1 |
| `CRAG_HIGH_THRESHOLD` | `0.75` | Cosine score above this → synthesize directly |
| `CRAG_LOW_THRESHOLD` | `0.45` | Cosine score below this → fall back to trending |
| `CRAG_MAX_RETRIES` | `2` | Hard cap on CRAG retry attempts |
| `CRAG_TIME_BUDGET_S` | `3.5` | Hard wall-clock cap on CRAG loop (seconds) |

---

## Pipeline Flow

Each inbound `POST /query` request traverses these stages in order:

```
POST /query
  │
  ├─ 1. Guardrail (pipeline/guardrail.py)
  │      GPT-4o-mini zero-shot classifier: is this a fashion query?
  │      Off-topic → return redirect response, skip retrieval entirely.
  │
  ├─ 2. Hybrid Retrieval (pipeline/retrieval.py)
  │      BM25 over product titles/descriptions (in-memory, built at startup from Postgres)
  │      + Milvus HNSW semantic search with metadata pre-filter
  │      → fused via RRF (k=60), returns top-20 candidates
  │
  ├─ 3. CRAG Corrective Loop (pipeline/crag.py)
  │      Grade top candidates by cosine similarity to query embedding.
  │      score > CRAG_HIGH_THRESHOLD  → proceed to reranker
  │      score in [LOW, HIGH]         → rewrite query (GPT-4o-mini) + retry retrieval
  │      score < CRAG_LOW_THRESHOLD
  │        or max retries hit         → fall back to non-personalized trending products
  │
  ├─ 4. Cross-Encoder Reranker (pipeline/reranker.py)
  │      ms-marco-MiniLM-L-6-v2 scores all 20 candidates in one batch forward pass.
  │      Business rules applied as score adjustments after cross-encoder:
  │        boost: days_since_added < 30
  │        demote: stock_level = low
  │      Returns top-5 to generator.
  │
  └─ 5. Generator (pipeline/generator.py)
         GPT-4o-mini synthesises a grounded answer from top-5 chunks.
         System prompt is pinned — never modified at runtime.
         Response includes cited product IDs.
```

---

## Milvus Schema

Collection: `fashion_rag` (set via `MILVUS_COLLECTION`)

| Field | Type | Purpose |
|---|---|---|
| `id` | VARCHAR | chunk UUID |
| `product_id` | VARCHAR | foreign key to PostgreSQL `products` |
| `vector` | FLOAT_VECTOR[1536] | `text-embedding-3-small` embedding |
| `chunk_type` | VARCHAR | `"description"` or `"review"` |
| `price_range` | VARCHAR | `"budget"` / `"mid"` / `"premium"` |
| `category` | VARCHAR | e.g. `"tops"`, `"trousers"`, `"dresses"` |
| `occasion` | VARCHAR | e.g. `"casual"`, `"formal"`, `"sport"` |
| `brand` | VARCHAR | normalized brand name |
| `text` | VARCHAR | raw chunk text (returned with results) |

**Schema constraint:** `price_range`, `category`, `occasion`, `brand` are indexed metadata fields
used for pre-filtering before ANN search. Adding new filter fields requires rebuilding the
collection and re-embedding. Do not add ad-hoc filter fields — modify the schema in
`ingestion/indexer.py` and document the change.

---

## Chunking Rules

- **Product descriptions:** sentence-aware sliding window, 256 tokens, ~50-token overlap.
  Uses `nltk.sent_tokenize()` + `tiktoken` for counting.
- **Customer reviews:** one chunk per review (reviews are semantically atomic). Reviews
  shorter than 20 words are dropped at ingestion.
- Chunking logic lives entirely in `ingestion/chunker.py`. The retrieval layer never
  re-chunks — it only queries.

---

## Startup Dependency Order

The service requires these to be running before a query succeeds:

1. PostgreSQL (BM25 index built from `products` table at startup)
2. Milvus (vector search)
3. Redis (query cache — startup proceeds if unavailable, caching silently disabled)
4. OpenAI API reachable (embeddings + generation)

Kafka is only needed for the async ingestion write path (`POST /ingest`), not for queries.

---

## Test Layout

```
tests/
├── conftest.py                  # shared fixtures, mock OpenAI client
├── test_guardrail.py            # classifier: fashion vs off-topic inputs
├── test_retrieval.py            # hybrid retrieval + RRF fusion logic
├── test_crag.py                 # grader thresholds, retry logic, fallback path
├── test_reranker.py             # score ordering, business rule adjustments
├── test_generator.py            # grounding prompt, cited_sources field
└── test_api.py                  # end-to-end POST /query with mocked pipeline
```

Tests use mocked OpenAI and Milvus clients — no external services required to run the suite.

---

## Port Assignment

| Service | Port |
|---|---|
| fashion-recommend (Go/Gin) | 5001 |
| python-agent (LangGraph) | 8001 |
| **rag-service (this service)** | **8002** |
| Gorse master HTTP | 8088 |
| Milvus gRPC | 19530 |
