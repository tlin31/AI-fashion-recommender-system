# rag-service

A standalone FastAPI microservice that adds natural language product search to the fashion recommendation system. Users submit free-text queries and receive grounded product recommendations with cited sources.

**Role in the system:** Complements the Gorse collaborative-filtering engine. Gorse handles behavioural signals (likes, views, purchases). This service handles semantic and cold-start queries — "show me casual summer dresses under $50" — where there are no prior interactions to learn from.

**Port:** `8002` (Gorse master: 8088 · python-agent: 8001 · fashion-recommend: 5001)

---

## How it works

Each `POST /query` request passes through five stages in order:

```
POST /query
  │
  ├─ 1. Guardrail
  │      GPT-4o-mini checks whether the query is fashion-related.
  │      Off-topic queries (e.g. "who won the World Cup?") are rejected
  │      immediately without touching the index.
  │
  ├─ 2. Hybrid Retrieval
  │      BM25 keyword search over 5,000 product names/descriptions (built
  │      at startup from PostgreSQL) is fused with Milvus HNSW semantic
  │      search using Reciprocal Rank Fusion (k=60). Returns top-20 chunks.
  │
  ├─ 3. CRAG Corrective Loop
  │      Grades the top candidates by cosine similarity to the query.
  │      High score  → proceed directly to reranking.
  │      Medium score → GPT-4o-mini rewrites the query and retries retrieval
  │                     (up to CRAG_MAX_RETRIES times).
  │      Low score   → falls back to non-personalized trending products.
  │
  ├─ 4. Cross-Encoder Reranker
  │      ms-marco-MiniLM-L-6-v2 rescores all 20 candidates in one batch
  │      forward pass. Business rule adjustments applied after scoring:
  │        + boost for products added in the last 30 days
  │        - demote for low-stock items
  │      Returns top-5 to the generator.
  │
  └─ 5. Generator
         GPT-4o-mini synthesises a grounded answer from the top-5 chunks.
         Response includes cited product IDs so the frontend can render
         product cards alongside the answer.
```

---

## Setup

**Prerequisites:** Python 3.11+, Docker

### 1. Start infrastructure

Redis, PostgreSQL, Milvus, Kafka, and Gorse all run in Docker via the shared compose file in `fashion-recommend/`:

```bash
cd fashion-recommend
docker-compose up -d
cd ../rag-service
```

Verify the containers are healthy:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

You should see `fashion-redis`, `fashion-postgres`, `fashion-milvus`, and `fashion-kafka` all with status `healthy`.

### 2. Install Python dependencies

```bash
cd rag-service
pip install -r requirements.txt
```

> **Conda users:** use the full path to ensure the right environment:
> ```bash
> /opt/homebrew/anaconda3/bin/pip install -r requirements.txt
> ```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
OPENAI_API_KEY=sk-...   # required — used for embeddings and generation
```

All other defaults (`localhost` ports, Postgres credentials) match the Docker Compose setup and work without changes for local development.

### 4. Seed the vector index

This only needs to be run once. It reads products from PostgreSQL, chunks descriptions, embeds them with `text-embedding-3-small`, and writes vectors to the Milvus `fashion_rag` collection:

```bash
python ingestion/indexer.py --seed
```

Seeding 5,000 products takes approximately 10–15 minutes (OpenAI API rate-limited). The BM25 index is rebuilt automatically from PostgreSQL on every service startup — no separate seeding step needed for that.

### 5. Run the service

```bash
uvicorn main:app --reload --port 8002
```

On startup you should see:

```
Cross-encoder reranker ready
BM25 index ready — 5000 products indexed
Milvus collection 'fashion_rag' loaded from http://localhost:19530
rag-service ready on port 8002
```

---

## API

### `POST /query`

Submit a natural language product search query.

**Request:**
```json
{
  "query": "casual blue summer dress under budget",
  "filters": { "price_range": "budget", "category": "dresses" },
  "top_k": 5
}
```

`filters` is optional. Supported filter fields: `price_range` (`budget` / `mid` / `premium`), `category`, `occasion`, `brand`, `chunk_type`.

**Response:**
```json
{
  "answer": "Here are some casual blue summer dresses that fit your budget...",
  "products": [
    {
      "product_id": "prod_0042",
      "name": "GRACE KARIN Women's Floral Sundress",
      "category": "dresses",
      "price": 28.99,
      "score": 0.87,
      "chunk_text": "Lightweight cotton blend, perfect for warm weather..."
    }
  ],
  "cited_sources": ["prod_0042", "prod_0107"],
  "retrieval_path": "synthesize",
  "latency_ms": {
    "guardrail_ms": 480,
    "retrieval_ms": 22,
    "crag_ms": 0,
    "rerank_ms": 95,
    "generation_ms": 1340,
    "total_ms": 1937
  }
}
```

`retrieval_path` indicates which CRAG branch was taken: `"synthesize"` (high confidence), `"retry"` (query was rewritten), or `"fallback"` (low confidence, trending products returned).

### `POST /ingest`

Queue products for async re-ingestion into the vector index via Kafka.

**Request:**
```json
{ "product_ids": ["prod_0042", "prod_0107"] }
```

Omit `product_ids` to re-ingest all products. Returns `202 Accepted` immediately; actual embedding and Milvus write happen asynchronously.

### `GET /health`

Returns the status of each downstream dependency.

**Response:**
```json
{
  "status": "ok",
  "milvus": true,
  "postgres": true,
  "redis": true
}
```

`status` is `"degraded"` if any dependency is unreachable. The service starts in degraded mode if Milvus is unavailable (all `/query` requests return empty results); Redis unavailability silently disables the embedding cache only.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Used for `text-embedding-3-small` embeddings, GPT-4o-mini generation, and Ragas evaluation. |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `fashion_rag` | Collection name. Do not change after the index is built without a full re-embed. |
| `POSTGRES_URL` | `postgresql://gorse:gorse_pass@localhost:5432/gorse` | Reuses the existing fashion-recommend database. BM25 index is built from `rag_products` at startup. |
| `REDIS_URL` | `redis://localhost:6379` | Embedding cache (TTL 1 hour). Service starts without Redis — caching is silently disabled. |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Only needed for `POST /ingest`. Not required for queries. |
| `KAFKA_INGEST_TOPIC` | `rag-ingest` | Kafka topic for async ingestion events |
| `CRAG_HIGH_THRESHOLD` | `0.75` | Cosine score above this → synthesize directly (no retry) |
| `CRAG_LOW_THRESHOLD` | `0.45` | Cosine score below this → fall back to trending products |
| `CRAG_MAX_RETRIES` | `2` | Maximum query-rewrite retries before falling back |
| `CRAG_TIME_BUDGET_S` | `3.5` | Wall-clock cap on the CRAG loop in seconds |

---

## Running tests

All 54 tests run in under 3 seconds with no external services required — OpenAI and Milvus are mocked.

```bash
# Run the full suite
pytest tests/ -v

# Run a single file
pytest tests/test_retrieval.py -v

# Run a single test by name
pytest tests/test_guardrail.py::test_timeout_defaults_to_true -v
```

**Test layout:**

```
tests/
├── conftest.py          # shared fixtures and mock OpenAI client
├── test_guardrail.py    # fashion vs off-topic classification (16 tests)
├── test_retrieval.py    # hybrid BM25 + Milvus + RRF fusion (38 tests)
└── test_chunker.py      # sentence-aware chunking logic
```

---

## Startup dependency order

The service requires these to be running before queries succeed:

1. **PostgreSQL** — BM25 index built at startup. Hard failure if unreachable (service won't start). Retries 3× with exponential backoff before aborting.
2. **Milvus** — vector search. Logs `CRITICAL` and starts in degraded mode if unreachable.
3. **Redis** — embedding cache. Optional. Silently disabled if unreachable; all queries fall through to OpenAI.
4. **OpenAI API** — embeddings and generation. Needed per-request, not at startup. Each `embed()` call has a 5-second timeout.

Kafka is only needed for the `POST /ingest` write path, not for queries.

---

## Embedding cache

`embed()` in `pipeline/utils.py` caches query vectors in Redis:

- **Key:** `emb:` + first 24 hex chars of `SHA256(query_text)`
- **TTL:** 1 hour
- **Cache hit:** ~3ms (Redis lookup, no OpenAI call)
- **Cache miss:** ~300–3000ms depending on network latency to OpenAI

**Important:** the cache is keyed on the exact query string. Two queries that mean the same thing but use different words will each miss the cache and call OpenAI separately. The cache pays off for repeated identical searches (user refreshes, retries, autocomplete debounce).

---

## Data sources

The vector index is seeded from the [Amazon Fashion dataset](https://amazon-reviews-2023.github.io/) (~800K reviews, 5,000 product subset). Raw data lives in `amazon_data/raw/`. Chunking rules:

- **Product descriptions:** sentence-aware sliding window, 256 tokens, ~50-token overlap (`ingestion/chunker.py`)
- **Customer reviews:** one chunk per review; reviews shorter than 20 words are dropped

---

## Running in Docker

```bash
# Build the image
docker build -t rag-service .

# Run (assumes infrastructure already up via docker-compose)
docker run --env-file .env \
  --network fashion-recommend_default \
  -p 8002:8002 \
  rag-service
```

NLTK tokenizer data (`punkt`) is downloaded at image build time, so there is no network call on cold start.

---

## Common commands

```bash
# Start infrastructure
cd fashion-recommend && docker-compose up -d && cd ../rag-service

# Install dependencies
pip install -r requirements.txt

# Seed vector index (first time only, ~10–15 min)
python ingestion/indexer.py --seed

# Run service
uvicorn main:app --reload --port 8002

# Run tests (no external services needed)
pytest tests/ -v

# Evaluate retrieval quality against the golden query set
python eval/run_eval.py --golden-set eval/golden_queries.json

# Check for quality regression against the stored baseline
python eval/check_regression.py --threshold 0.05

# Load test
locust -f eval/locustfile.py --host http://localhost:8002
```
