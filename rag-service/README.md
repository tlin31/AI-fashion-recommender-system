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
  │      High score  → proceed directly to reranking ("synthesize").
  │      Medium score → GPT-4o-mini rewrites the query and retries retrieval.
  │      Low score   → returns the best real candidates ("best_effort").
  │      Trending products are used ONLY when retrieval returns nothing at all.
  │
  ├─ 4. Cross-Encoder Reranker
  │      ms-marco-MiniLM-L-6-v2 rescores all 20 candidates in one batch
  │      forward pass. Business rule applied after scoring:
  │        + 0.05 boost for description chunks over review chunks
  │      Then collapses to one chunk per product (highest scorer wins) so
  │      top_k counts distinct products. Returns top-k to the generator.
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

`retrieval_path` indicates which CRAG branch was taken: `"synthesize"` (high confidence), `"retry"` (query was rewritten), `"best_effort"` (low confidence — real candidates, weak signal), or `"fallback"` (retrieval returned nothing; trending products).

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
| `OPENAI_API_KEY` | — | **Required.** Used for `text-embedding-3-small` embeddings, GPT-4o-mini generation, and the faithfulness judge. |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `fashion_rag` | Collection name. Do not change after the index is built without a full re-embed. |
| `POSTGRES_URL` | `postgresql://gorse:gorse_pass@localhost:5432/gorse` | Reuses the existing fashion-recommend database. BM25 index is built from `rag_products` at startup. |
| `REDIS_URL` | `redis://localhost:6379` | Embedding cache (TTL 1 hour). Service starts without Redis — caching is silently disabled. |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Only needed for `POST /ingest`. Not required for queries. |
| `KAFKA_INGEST_TOPIC` | `rag-ingest` | Kafka topic for async ingestion events |
| `CRAG_HIGH_THRESHOLD` | `0.45` | Grade above this → synthesize directly (no retry) |
| `CRAG_LOW_THRESHOLD` | `0.43` | Grade below this → return `best_effort` candidates (real results, weak signal — **not** trending) |
| `CRAG_MAX_RETRIES` | `2` | Maximum query-rewrite retries before returning best effort |
| `CRAG_TIME_BUDGET_S` | `3.5` | Wall-clock cap on the CRAG loop in seconds |

CRAG thresholds were selected empirically — see [CRAG Threshold Calibration](#crag-threshold-calibration-2026-08-06).

---

## Running tests

All 116 tests run in about 10 seconds with no external services required — OpenAI and Milvus are mocked.

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

# Load test (headless, 3 min per level, results saved to eval/load_test_u*.csv)
locust -f eval/locustfile.py --host http://localhost:8002 --users 1  --spawn-rate 1  --run-time 3m --headless --csv eval/load_test_u1
locust -f eval/locustfile.py --host http://localhost:8002 --users 5  --spawn-rate 5  --run-time 3m --headless --csv eval/load_test_u5
locust -f eval/locustfile.py --host http://localhost:8002 --users 10 --spawn-rate 10 --run-time 3m --headless --csv eval/load_test_u10
locust -f eval/locustfile.py --host http://localhost:8002 --users 20 --spawn-rate 20 --run-time 3m --headless --csv eval/load_test_u20
```

---

## Load Test Results (baseline, 2026-07-11)

Tested against 100 golden queries (prevents Redis cache inflation from a small fixed set).
All runs: 0% failure rate. Full pipeline active: BM25 + Milvus + cross-encoder + GPT-4o-mini.

| Concurrent Users | Requests | Fail% | p50 (ms) | p95 (ms) | p99 (ms) | Avg (ms) | RPS  |
|-----------------:|---------:|------:|---------:|---------:|---------:|---------:|-----:|
| 1                | 27       | 0%    | 4,100    | 9,700    | 12,000   | 4,570    | 0.15 |
| 5                | 153      | 0%    | 3,300    | 6,800    | 9,800    | 3,787    | 0.86 |
| 10               | 323      | 0%    | 3,300    | 6,400    | 8,200    | 3,509    | 1.81 |
| 20               | 605      | 0%    | 3,600    | 7,000    | 8,700    | 3,865    | 3.40 |

**Observations:**
- **Bottleneck is OpenAI API latency (~3–4 s/call), not local infra.** p50 stays flat across all concurrency levels because requests fan out to OpenAI independently.
- **p99 improves as concurrency rises** (12 s → 8.7 s) due to Redis cache hits warming up across the 100-query pool.
- **RPS scales linearly** (0.15 → 3.40 for 1 → 20 users) — no saturation point reached at 20 users.
- Degradation point was not hit; push to 50–100 users to find the ceiling in a production environment.

---

## Evaluation (2026-08-11)

Measured against a hand-labelled golden query set: 100 queries stratified across
navigational / exploratory / attribute / edge cases, with graded relevance judgments
(0 / 1 / 2) built by TREC-style pooling over three adjudication rounds — **1,481
(query, product) judgments, 14.8 per query.**

| Metric | Value | Target | |
|---|---:|---:|---|
| NDCG@10 | 0.8468 | ≥ 0.50 | pass |
| Recall@10 | 0.6993 | ≥ 0.70 | **miss by 0.0007** |
| Faithfulness | 0.9580 | ≥ 0.85 | pass |

By query type:

| Type | n | NDCG@10 | Recall@10 | Faithfulness |
|---|---:|---:|---:|---:|
| navigational | 20 | 0.9393 | 0.8249 | 1.0000 |
| attribute | 30 | 0.8998 | 0.7837 | 0.9779 |
| exploratory | 30 | 0.8157 | 0.5990 | 0.9368 |
| edge | 20 | 0.7214 | 0.5977 | 0.9180 |

### Why Recall@10 is structurally capped at 0.7615

Recall@10 misses its target, and the reason is arithmetic rather than retrieval quality.

`Recall@10 = hits in top-10 / total relevant`. The denominator counts **every** product
judged relevant for that query, including ones that cannot fit in ten slots. After round 3
the golden set averages **13.4 relevant products per query** (median 14, max 21), and
**81 of 100 queries have more than 10 relevant products.**

Ten slots cannot hold 13.4 items, so the maximum attainable score is:

```
max Recall@10 = mean( min(10, |relevant|) / |relevant| ) = 0.7615
```

Against that ceiling, 0.6993 is **91.8% of what is mathematically achievable**. The ≥0.70
target was set when the label set averaged 5.8 judgments per query and the ceiling was
close to 1.0; denser labelling lowered the ceiling without changing retrieval at all.

The target is deliberately left at 0.70 rather than lowered to match the result — moving a
threshold to fit an outcome is not a fix. The honest statements are that Recall@10 is
0.6993, that its ceiling is 0.7615, and that **Recall@k is the wrong headline metric once
the number of relevant items routinely exceeds k**. NDCG@10 does not have this problem: it
normalises by the ideal ranking truncated at k, so it stays interpretable.

### Faithfulness judge — what actually runs

The scorer is **not** Ragas, despite the original filename. `ragas` is deliberately left
uninstalled (commented out in `requirements.txt`), so `eval/faithfulness_judge.py` runs its
own two-step GPT-4o-mini judge: extract the factual claims from the answer, then verify
each against the retrieved context. Score = supported claims / total claims.

Installing `ragas` silently switches judges and makes new numbers incomparable to the
locked baseline — hence the comment rather than a deletion.

Known bias: the judge returns 1.0 when claim extraction yields nothing or the response
fails to parse. Short answers therefore skew high — all 20 navigational queries scored
exactly 1.0000 — so 0.9580 is somewhat optimistic.

### Labelling honesty note

Round 3 judged **89% of pooled candidates as relevant** (474 of 903 at the top grade),
against the 10–30% typical of a TREC pool. Pooling only from this system's own output and
then grading generously means the system is partly measured against a standard it defined.
A strict re-scoring — counting only grade 2 as relevant — gives NDCG@10 **0.8033** and
Recall@10 **0.7785**, so the conclusion does not depend on the lenient 0-versus-1 boundary.

---

## CRAG Threshold Calibration (2026-08-06)

The CRAG thresholds were not chosen by intuition. This section records the method, the
results, and — importantly — the negative findings, because the honest conclusion is that
**the corrective loop delivers very little on this catalog and the calibration confirmed
the existing thresholds rather than improving on them.**

### Method: cache-and-replay

CRAG thresholds affect **routing only**, never retrieval. Given a query, the candidates
`hybrid_search()` returns and their cosine scores are identical whether the threshold is
0.45 or 0.65. So the expensive half runs once and every threshold combination is replayed
against a cached snapshot:

```bash
python eval/calibrate_crag.py --build-cache   # 100 queries, needs Postgres + Milvus + OpenAI
python eval/calibrate_crag.py --grid          # pure offline CPU, free, rerunnable
```

This is the same principle as sweeping a classifier's decision threshold over cached scores
to draw an ROC curve: **score once, threshold many times.** Cost dropped from 12 full eval
runs to ~1.2. `eval/calibration_cache.json` is committed, so anyone can re-derive the
decision without an API key, and adding a new candidate threshold later costs nothing.

The cache carries a **fingerprint** (embedding model, RRF k, candidate pool, per-product
chunk cap, retrieval top-k, Milvus row count). `--grid` refuses to run if the live config
has drifted — a stale cache is worse than no cache, because it fails silently and produces
confident, wrong thresholds.

### The grid: 50 combinations

Three graders × five HIGH cutoffs × three-to-four LOW cutoffs, minus invalid pairs where
`LOW >= HIGH`:

| Grader | HIGH candidates | LOW candidates | Valid combos |
|---|---|---|---:|
| `mean20` (production) | 0.45, 0.50, 0.55, 0.60, 0.65 | 0.10, 0.32, 0.38, 0.43 | 20 |
| `max` | 0.5124, 0.6007, 0.6925, 0.7406, 0.8174 | 0.3818, 0.4531, 0.5064 | 15 |
| `top3` | 0.4942, 0.5703, 0.6415, 0.6747, 0.7514 | 0.3751, 0.4409, 0.4784 | 15 |

Each grader has a different natural score range, so the non-production graders reuse the
**same percentile positions** as the production anchors rather than the same absolute
numbers — combos are compared at equal routing rates, not at arbitrary cutoffs.

Objective: **NDCG@10**. Constraint: retry rate < 20%. Faithfulness was deliberately *not*
used as the objective — the generator is faithful to whatever chunks it receives, so it sits
near 0.95 regardless of routing and cannot discriminate between combos.

### Score distribution

Grading uses the mean Milvus cosine across all 20 retrieved candidates. Over the 100-query
golden set:

```
p0 0.3205   p25 0.4289   p50 0.5125   p75 0.6030   p90 0.6470   p100 0.7091
```

**A HIGH threshold of 0.75 is unreachable on this catalog.** An earlier configuration
documented 0.75/0.45; it would have produced zero `synthesize` decisions and routed every
single query through a rewrite.

The distribution is also tight: moving HIGH by 0.05 reroutes roughly 16–18 of 100 queries.
That brittleness is worth knowing — modest data drift changes routing behaviour with no
code change.

### Results

Scored against the completed label set — 1,481 graded (query, product) judgments over 100
queries, after a third pooling round took coverage from 5.8 to 14.8 labels per query. The
grid was built *before* that labeling, deliberately: pooling over the union of both
candidate sets (`eval/pull_adjudication_round3.py`) removes a bias that would otherwise
penalise combos for surfacing unjudged products, since retry returns rewritten-query
results that are likelier to be unlabelled.

| Config | NDCG@10 | Recall@10 | Retry rate | Expected added latency |
|---|---:|---:|---:|---:|
| `mean20 / 0.45 / 0.10` (pre-calibration) | 0.8459 | 0.6983 | 0.30 | 309.6 ms |
| **`mean20 / 0.45 / 0.43` (selected)** | **0.8468** | 0.6993 | **0.05** | **51.6 ms** |
| `mean20 / 0.50 / 0.43` | 0.8483 | 0.6999 | 0.23 | 237.4 ms |

Bootstrap, 1,000 resamples of the query set with threshold selection re-run on each:

- Improvement over the pre-calibration config: **+0.0058, 95% CI [+0.0009, +0.0140]** —
  the interval **excludes zero**.
- Selection stability: the winning combo took only **21.2%** of resamples.

**Conclusion: the selected config is a real improvement, but the precise optimum is not
identified.** The direction is solid — raising LOW to 0.43 beats the previous setting
outside the noise band while cutting the retry rate from 30% to 5% and expected added
latency from 310 ms to 52 ms. Which combo is *best*, however, is near-tied: several
configurations sit within 0.002 NDCG of each other, and no single one wins a majority of
resamples. Reported as such rather than presenting a point estimate as settled.

The higher-NDCG alternatives all violate the retry-rate constraint — `mean20 / 0.50 / 0.43`
buys +0.0015 NDCG for 4.6× the retry rate and 185 ms of added latency.

**Labeling honesty note.** Round 3 judged 89% of pooled candidates as relevant (474 of 903
at the top grade), well above the 10–30% typical of TREC pools. Pooling only from this
system's own output and then grading generously means the system is partly measured against
a standard it defined. A sensitivity check holds the conclusion: scoring strictly (only
grade 2 counts as relevant) gives NDCG@10 0.8033 and Recall@10 0.7785, so the result does
not depend on the lenient 0-versus-1 boundary.

*Historical note:* an earlier version of this section reported NDCG 0.6931 / Recall 0.6740
against an incomplete label set, where ~60% of returned slots were unjudged and therefore
scored as irrelevant. Those figures were pessimistic by construction.

### What changed, and why

`CRAG_LOW_THRESHOLD` was raised from `0.10` to `0.43`.

The decision was made on the incomplete label set, where the quality difference read as
−0.0044 NDCG — inside the noise band, and therefore an unprovable loss traded against a
deterministic 258 ms latency saving and a retry-rate drop from 30% to 5%. That was the
right call on the evidence available.

Re-scoring against the completed labels reversed the sign: the change is now **+0.0009
NDCG with a 95% CI excluding zero**, so it improves quality *and* latency. Worth noting
that the original justification did not depend on this — the trade was defensible when the
quality term was believed to be slightly negative.

Confirmation run against the live service after the change (incomplete label set, so the
absolute values are lower than those reported above — the point is the agreement, not the
level):

| | Simulated | Live | Diff |
|---|---:|---:|---:|
| NDCG@10 | 0.6931 | 0.6930 | −0.0001 |
| Recall@10 | 0.6740 | 0.6740 | 0.0000 |

The offline replay reproduces the live pipeline to four decimal places, which validates the
cache-and-replay methodology itself.

### Negative findings (kept deliberately)

CRAG is retained in the codebase and this section documents why it underperforms, rather
than deleting the experiment.

1. **The corrective action is net harmful.** Across 100 queries, the query rewrite *improved*
   the retrieval grade for 24 (mean +0.0246) and *degraded* it for 75 (mean −0.0275).

2. **Root cause: the rewrite introduces no new information.** In the original CRAG paper the
   corrective action on a failed retrieval is a *web search* — new knowledge from outside the
   corpus. Here it rephrases the query and searches the same 5,000-product index. If the
   product is not in the catalogue, no rephrasing will find it. The user's original phrasing
   is usually the most accurate expression of intent, so LLM rewriting mostly adds semantic drift.

3. **The grader is a weak signal.** Rank correlation between the grade and true NDCG@10,
   measured against the golden set:

   | Grader | Pearson | Spearman |
   |---|---:|---:|
   | `mean20` (production) | 0.393 | 0.443 |
   | `max` | 0.490 | 0.543 |
   | `top3` | 0.457 | 0.513 |

   `max` correlates better because averaging across all 20 candidates lets the weak tail —
   which is weak for good *and* bad retrievals, and therefore carries almost no information —
   pull every query toward the middle. **But `max` did not win the grid**: better correlation
   with the proxy did not translate into better end-to-end NDCG. The production grader was
   kept on that evidence.

4. **The retry loop cannot improve on its second attempt.** `run_crag()` rewrites the
   *original* query on every attempt and never updates it, so with `temperature=0` attempt 2
   reproduces attempt 1 exactly and burns roughly 1 s for nothing. Retained as a known issue.

5. **Structural limit.** CRAG pays off when the corpus is large and heterogeneous and an
   external knowledge source can be consulted. On a single-domain closed catalogue of 5,000
   products, a product either exists or it does not, and the headroom for a corrective loop is
   small. On this project the larger gains are in retrieval itself — chunking, embeddings,
   query understanding — not in correction.

### Future work

**Reconsider the grader.** `_grade()` averages cosine over all 20 candidates, so the weak tail
compresses the achievable range to 0.32–0.71, which is why HIGH had to be tuned down to 0.45
rather than the 0.75 the design assumed. Grading on the max, or on the top-5 mean, would give a
more separable signal and let the threshold be set on the quality of the *best* evidence rather
than the average of everything retrieved — but it invalidates the calibration cache and turns
threshold tuning into a grader redesign, so it belongs in its own session.

**Move the grader after the cross-encoder.** The pipeline already runs a cross-encoder, which is
a far stronger relevance signal than cosine. Reordering to `retrieve → rerank → grade → retry`
would cost ~120 ms on the retry path versus the 886 ms the rewrite currently costs, and the
forward pass is already paid for on the happy path.

**Correct by failure mode.** One action (rewrite) is currently applied to every failure. Relaxing
metadata filters, decomposing multi-constraint queries, or honestly reporting absence are
different corrections for different causes.

**Report CRAG on the subset where it fires.** Whole-set NDCG understates it: the loop affects
~30% of queries, so a real improvement there is diluted roughly threefold in the headline number.
