# rag-service

A standalone FastAPI microservice that adds natural-language product search to a fashion
recommendation platform. Users submit free-text queries and receive grounded product
recommendations with cited sources.

**Role in the system:** complements the Gorse collaborative-filtering engine rather than
replacing it. Gorse learns from behavioural signals — likes, views, purchases. This service
handles the queries CF structurally cannot answer: semantic intent and cold start. *"Show me
casual summer dresses under $50"* has no interaction history to learn from, and a product
added yesterday has no interactions at all.

**Port:** `8002` · **Indexed:** 5,000 products → 6,337 description vectors
(64,744 customer reviews are ingested into PostgreSQL but **not** indexed — see
[Reviews are not indexed](#reviews-are-not-indexed))

---

## Results

Measured against a hand-labelled golden query set — 100 queries, 1,481 graded relevance
judgments built by TREC-style pooling across three adjudication rounds.

| Metric | Value | Target | |
|---|---:|---:|---|
| NDCG@10 | **0.8468** | ≥ 0.50 | pass |
| Recall@10 | **0.6993** | ≥ 0.70 | miss by 0.0007 — [structurally capped at 0.7615](#why-recall10-is-structurally-capped-at-07615) |
| Faithfulness | **0.9580** | ≥ 0.85 | pass |

Latency, measured p50 over warm queries:

| | `/query` | `/query/stream` |
|---|---:|---:|
| First visible result | 2424 ms | **527 ms** |
| First token of prose | — | 1276 ms |
| Complete answer | 2424 ms | 2583 ms |

Streaming does not make generation faster — it stops the caller waiting on work
that already finished. Products are ranked once reranking completes and are sent
then, rather than being held until the last token is written. **78% cut in
time-to-first-content.** See
[Latency: where the time goes](#latency-where-the-time-goes).

Load: 0% failures at 1/5/10/20 concurrent users; 1.81 RPS at 10 users. Latency is
dominated by OpenAI round-trips, not local infrastructure.

127 unit tests, fully mocked. A regression gate blocks metric drops over 5%.

**Three things worth knowing before reading further**, because they are the parts most
projects leave out:

- Recall@10 misses its target for **arithmetic** reasons, not quality ones, and the target
  was deliberately not lowered to match the result.
- The CRAG corrective loop **does not work well here**, and the section explaining why is
  kept rather than deleted.
- The threshold calibration **confirmed** the existing values instead of improving them.

---

## Architecture

Two endpoints run the identical pipeline and differ only in how the answer is
delivered — `POST /query` returns it complete, `POST /query/stream` streams it.

```
POST /query  ·  POST /query/stream
  │
  ├─ Guardrail STARTED, not awaited ────────────────────────┐
  │    GPT-4o-mini zero-shot: is this a fashion query?      │
  │    Redis guard:{sha256}, TTL 24h — hit is ~0 ms         │  runs concurrently
  │    Fails OPEN: a timeout lets the query through         │  with everything
  │                                                         │  below it
  ├─ 1. Hybrid Retrieval          ~20 ms                    │
  │    BM25 (in-memory, built from Postgres at startup)     │
  │    ∥ Milvus HNSW  → RRF fusion (k=60) → top-20 chunks   │
  │    Metadata pre-filter applied inside the ANN search.   │
  │    Dense path unavailable → BM25-only, degraded[] set   │
  │                                                         │
  ├─ 2. CRAG Corrective Loop      ~0.2 ms                   │
  │    Mean cosine over candidates — no extra API call.     │
  │    ≥ HIGH → reranker ("synthesize")                     │
  │    middle → rewrite + retry ("retry")                   │
  │    < LOW  → best real candidates ("best_effort")        │
  │    Trending only when retrieval returns nothing at all. │
  │                                                         │
  ├─ 3. Cross-Encoder Reranker    ~230 ms                   │
  │    ms-marco-MiniLM-L-6-v2, all 20 candidates in one     │
  │    batch forward pass. +0.05 description boost (a no-op │
  │    today — every indexed chunk is a description).       │
  │    Collapses to one chunk per product, so top_k counts  │
  │    distinct products.                                   │
  │                                                         │
  ├─ 4. Guardrail GATE ⇦─────────────────────────────────────┘
  │    Residual wait 0–131 ms against a ~511 ms call.
  │    Last point where rejecting is free: everything above is
  │    local, everything below costs an LLM call. NO → 400.
  │
  │    /query/stream: products emitted HERE, ~527 ms
  │
  └─ 5. Generator                 ~1700 ms
       GPT-4o-mini, temperature 0, pinned system prompt.
       Answers only from supplied context.
       Failure → products returned without prose, degraded[] set.
```

Every response carries a per-stage latency breakdown (`guardrail_ms`, `retrieval_ms`,
`crag_ms`, `rerank_ms`, `generation_ms`, `total_ms`) and a `degraded` list naming any
component that failed. `guardrail_ms` is the **residual wait**, not the call duration —
the call overlaps the stages above it.

That instrumentation is what located the bottleneck; the section below is what it found.

---

## Design decisions and tradeoffs

| Decision | Alternative | Why | What it costs |
|---|---|---|---|
| **Hybrid BM25 + dense** | Dense only | Brand and SKU queries need exact lexical match. `"NELEUS women's 3 pack compression tank"` is a keyword problem, not a semantic one | Two indexes to keep in sync; BM25 rebuilt from Postgres at every startup |
| **RRF fusion (k=60)** | Weighted sum `α·dense + (1−α)·sparse` | Parameter-free and scale-invariant — see below | Discards score magnitude: a runaway top hit contributes the same as a marginal one |
| **HNSW index** | IVF_FLAT | Append-friendly — no retraining as the catalogue grows | Higher memory; `ef` must be tuned at query time |
| **Metadata pre-filter** | Post-filter after ANN | Filtering inside the ANN search preserves recall; post-filtering can empty the result set when the filtered subset never reaches the top-K | Filter fields must exist in the Milvus schema at index-build time — adding one means a full re-embed |
| **Cross-encoder rerank** | Bi-encoder, or no rerank | Joint `(query, doc)` encoding captures interaction terms a bi-encoder cannot | ~120 ms CPU — affordable only because it runs over 20 candidates, not the index |
| **Cosine grader in CRAG** | LLM-as-grader | Zero additional latency and zero API cost | Weak signal (Spearman 0.44 vs true NDCG). Measured and documented rather than assumed |
| **Sentence-aware chunking** | Fixed-size character split | Embedding quality degrades badly on truncated sentences | Slower; needs `nltk` + `tiktoken` at ingest |
| **Reviews as atomic chunks** *(implemented, not wired up)* | Split long reviews | Splitting mid-review destroys sentiment context — the thing that makes a review useful | `chunk_review()` exists but the seeding script never calls it, so no review is currently indexed |
| **`temperature=0`** | Non-zero for variety | Grounded synthesis should be deterministic; non-zero makes faithfulness evaluation irreproducible | Repetitive phrasing across similar queries |
| **Fail-open Redis and guardrail** | Fail-closed | A search box should degrade, not 500. Redis down ⇒ cache disabled; classifier timeout ⇒ query proceeds | Off-topic queries can slip through when the classifier times out |
| **Kafka for `/ingest`** | Synchronous HTTP ingest | Decouples write throughput from query latency — bulk re-indexing cannot degrade live search | Extra infrastructure, unnecessary for the read path |
| **Custom eval metrics** | An off-the-shelf eval library | Full control over gain function and discount, and invariant assertions inside the metric | Must implement and test NDCG/Recall correctly — a duplicate-ID bug got through until asserts were added |

### Why RRF rather than a weighted sum

BM25 scores are unbounded and corpus-dependent; cosine similarity lives in `[-1, 1]`. They
are not on comparable scales, so any `α·dense + (1−α)·sparse` needs `α` retuned per corpus,
and arguably per query type — navigational queries want sparse weighted heavily, exploratory
queries want dense.

RRF sidesteps this by using **rank only**:

```
RRF(d) = Σ  1 / (k + rank_i(d)),   k = 60
```

Parameter-free, immune to scale differences, and robust when one retriever returns garbage —
a bad retriever contributes small reciprocals rather than a large raw score. The cost is
real: RRF throws away *how much* better the top hit was. For this catalogue that trade is
worth it; for a corpus where score magnitude is meaningful it might not be.

### Why the cross-encoder is affordable

A cross-encoder is the expensive way to rank — it runs a full forward pass per
`(query, document)` pair, which is why nobody uses one over a whole index. It is affordable
here purely because **retrieval has already narrowed the field to 20 candidates**. Batched
into a single forward pass, that is ~120 ms on CPU.

The architectural point is that the bi-encoder (embeddings) and cross-encoder do different
jobs: the bi-encoder is a cheap recall device over 6,337 vectors, the cross-encoder is an
expensive precision device over 20. Using either for the other's job would be a mistake.

### `best_effort` rather than a trending fallback

The original design fell back to trending products when retrieval quality was low. That
was wrong, and the eval caught it: **trending products have zero overlap with the labelled
relevant set, so those queries scored NDCG = 0** — worse than returning the weak candidates
that had actually been retrieved. 26 queries were affected.

Now a low grade returns the real candidates tagged `best_effort`. Trending fires only when
retrieval returns *nothing at all*, which is the one case where there is genuinely nothing
better to show. Honest weak results beat confident irrelevant ones.

### Calibration by cache-and-replay

CRAG thresholds affect **routing only, never retrieval** — given a query, the candidates and
their cosine scores are identical whether the threshold is 0.45 or 0.65. So retrieval runs
once, and all 50 threshold combinations are replayed offline against a cached snapshot.
Cost fell from 12 full evaluation runs to ~1.2.

Same idea as sweeping a classifier's decision threshold over cached scores to draw an ROC
curve: **score once, threshold many times.** The cache is committed, so the decision is
re-derivable without an API key and adding a new candidate threshold later is free.

The precondition matters and is enforced: the cache is only valid while the cached stage is
independent of the swept parameter. It carries a fingerprint (embedding model, RRF `k`,
candidate pool, chunk cap, retrieval top-k, Milvus row count) and the grid **refuses to run**
if the live configuration has drifted. A stale cache is worse than no cache — it fails
silently and produces confident, wrong answers.

---

## Latency: where the time goes

The per-stage instrumentation exists so this section can be written from measurement
rather than intuition. Every number below was measured on the golden query set.

### Decomposing generation

Generation is ~69% of a request. Fitting the two query classes — navigational at 108
output tokens / 1817 ms, exploratory at 200 tokens / 2581 ms (n=36, correlation between
output length and `generation_ms` = 0.738):

```
generation ≈ 920 ms  +  8.3 ms × output_tokens
             └ TTFT      └ decode, ~120 tok/s
```

Median answer is 115 tokens and **nothing hits the 300-token cap** — `max_tokens=300`
is currently inert. That decomposition rules several options in and out:

| Option | Saving | Why |
|---|---|---|
| Lower `max_tokens` | **0 ms** | No query reaches the cap; the parameter does nothing today |
| Shrink input context (`top_k` 5→3) | **≈0 ms** | Prefill is fast — this is a *cost* lever, not a latency one |
| Halve output length | ~480 ms | Cuts decode only; the 920 ms TTFT floor is untouched |
| **Stream the answer** | **1900 ms perceived** | Turns TTFT into the number the user experiences |
| Skip generation entirely | 1817 ms | Only for classes where prose adds nothing |

### What was done

**Guardrail moved off the critical path.** It is a full GPT-4o-mini round-trip (~511 ms
isolated) whose verdict is independent of retrieval, so it now starts first and is awaited
just before the generator — overlapping retrieval, CRAG and rerank. Residual wait is
0–131 ms.

*The first attempt at this was wrong and is worth recording.* Gathering the guardrail with
retrieval saved only **19 ms**, because retrieval is ~19 ms and **you can only hide the
smaller of two concurrent tasks**. Moving the await later, so it overlaps ~250 ms of
rerank instead, is what actually recovered the cost.

**Guardrail verdicts cached.** `guard:{sha256}` in Redis, TTL 24 h against the embedding
cache's 1 h, because a query's topic does not change. `guardrail_ms` drops to exactly 0.0
on a hit. Only real verdicts are cached — the fail-open `True` from the exception path is
never written, or one API blip would whitelist a query for a day.

**Answer streamed, products first.** `POST /query/stream` emits Server-Sent Events:

```
event: products   products, retrieval_path, degraded      p50   527 ms
event: token      one text delta, many of these           p50  1276 ms
event: done       cited_sources, latency_ms               p50  2583 ms
event: error      generation failed after products sent
```

| | `/query` | `/query/stream` |
|---|---:|---:|
| First visible result | 2424 ms | **527 ms** (−78%) |
| Complete answer | 2424 ms | 2583 ms (unchanged) |

Total time is deliberately unchanged. Products are ranked the moment reranking finishes
and were previously held until the last token was written; the endpoint stops the caller
waiting on work that already completed.

Three implementation points that were not obvious:

- Retrieval runs **before** `StreamingResponse` is returned, so a guardrail rejection is
  still a clean `400`. Once bytes are on the wire the status code is committed and
  failures can only be reported as events — hence `event: error` followed by `done`
  rather than an aborted connection.
- `cited_sources` ships in `done`, not with `products`: `[n]` references cannot be
  resolved from a partial answer.
- `X-Accel-Buffering: no` is set, because a buffering proxy would silently undo the
  entire endpoint.

A separate endpoint rather than a flag on `/query` keeps the JSON contract untouched —
`eval/run_eval.py`, `eval/locustfile.py` and the regression gate all parse `/query` and
none needed changing.

### An honest limit on the measurement

**End-to-end `total_ms` could not be shown to improve** by the guardrail work.
`generation_ms` spans p10 1342 ms to p90 3317 ms on this machine, and even local CPU
`rerank_ms` swings 227–771 ms between runs. A ~400 ms effect is not separable from that
at n=30 — one run showed the cached condition as *slower*, which is plainly noise.

The mechanism is confirmed at stage level (`guardrail_ms` falls from ~511 ms to 0–131 ms).
The total is bounded by generation, and no amount of work elsewhere changes that.

### What is left

Skipping generation for navigational queries would save the full ~1817 ms rather than
hiding it, and the data supports the premise: navigational NDCG@10 is 0.9359, so retrieval
is near-perfect and the products *are* the answer. The obstacle is runtime classification —
query `type` exists only as evaluation metadata.

The first candidate signal, BM25 score concentration, was tested offline and **rejected**
(`eval/probe_navigational_router.py`; see *What didn't work* item 8). A viable version needs
an explicit intent signal rather than a retrieval-score heuristic: the guardrail already
makes an LLM call that runs concurrently and is Redis-cached, so returning an `intent` field
alongside `is_fashion` would cost no extra latency on the critical path. That is a session,
not a patch, and it is not started.

---

## Evaluation (2026-08-11)

100 queries stratified across navigational / exploratory / attribute / edge cases, with
graded judgments (0 / 1 / 2) from TREC-style pooling over three rounds — **1,481
(query, product) judgments, 14.8 per query.**

| Type | n | NDCG@10 | Recall@10 | Faithfulness |
|---|---:|---:|---:|---:|
| navigational | 20 | 0.9393 | 0.8249 | 1.0000 |
| attribute | 30 | 0.8998 | 0.7837 | 0.9779 |
| exploratory | 30 | 0.8157 | 0.5990 | 0.9368 |
| edge | 20 | 0.7214 | 0.5977 | 0.9180 |

The gradient is the expected one: exact-match navigational queries are easiest, deliberately
ambiguous edge cases hardest.

### Why Recall@10 is structurally capped at 0.7615

Recall@10 misses its target, and the reason is arithmetic rather than retrieval quality.

`Recall@10 = hits in top-10 / total relevant`. The denominator counts **every** product
judged relevant, including ones that cannot fit in ten slots. After round 3 the golden set
averages **13.4 relevant products per query** (median 14, max 21), and **81 of 100 queries
have more than 10 relevant products.**

Ten slots cannot hold 13.4 items, so:

```
max Recall@10 = mean( min(10, |relevant|) / |relevant| ) = 0.7615
```

Against that ceiling, 0.6993 is **91.8% of what is mathematically achievable**. The ≥ 0.70
target was set when the label set averaged 5.8 judgments per query and the ceiling was near
1.0; denser labelling lowered the ceiling without changing retrieval at all.

The target is deliberately left at 0.70 rather than lowered to match the result — moving a
threshold to fit an outcome is not a fix. The honest statements are that Recall@10 is
0.6993, that its ceiling is 0.7615, and that **Recall@k stops being a meaningful headline
metric once the number of relevant items routinely exceeds k**. NDCG@10 does not have this
problem: it normalises by the ideal ranking truncated at k, so it stays interpretable at
any label density.

### Faithfulness judge — what actually runs

The scorer is **not** Ragas, despite the original filename. `ragas` is deliberately left
uninstalled (commented out in `requirements.txt`), so `eval/faithfulness_judge.py` runs its
own two-step GPT-4o-mini judge: extract the factual claims from the answer, then verify each
against the retrieved context. Score = supported claims / total claims.

Installing `ragas` silently switches judges and makes new numbers incomparable to the locked
baseline — hence the comment rather than a deletion.

Known bias: the judge returns 1.0 when claim extraction yields nothing or the response fails
to parse. Short answers therefore skew high — all 20 navigational queries scored exactly
1.0000 — so 0.9580 is somewhat optimistic.

### Labelling honesty note

Round 3 judged **89% of pooled candidates as relevant** (474 of 903 at the top grade),
against the 10–30% typical of a TREC pool. Pooling only from this system's own output and
then grading generously means the system is partly measured against a standard it defined.
A strict re-scoring — counting only grade 2 as relevant — gives NDCG@10 **0.8033** and
Recall@10 **0.7785**, so the conclusion does not depend on the lenient 0-versus-1 boundary.

Single annotator throughout. A second annotator and a Cohen's kappa is the remaining gap;
below roughly 0.6 agreement the metric is not trustworthy regardless of how it is computed.

---

## CRAG Threshold Calibration (2026-08-06)

The thresholds were not chosen by intuition. This section records the method, the results,
and the negative findings — the honest conclusion is that **the corrective loop delivers
very little on this catalogue, and the calibration confirmed the existing thresholds rather
than improving them.**

```bash
python eval/calibrate_crag.py --build-cache   # needs Postgres + Milvus + OpenAI
python eval/calibrate_crag.py --grid          # pure offline CPU, free, rerunnable
```

### The grid: 50 combinations

Three graders × five HIGH cutoffs × three-to-four LOW cutoffs, minus invalid pairs where
`LOW >= HIGH`:

| Grader | HIGH candidates | LOW candidates | Valid combos |
|---|---|---|---:|
| `mean20` (production) | 0.45, 0.50, 0.55, 0.60, 0.65 | 0.10, 0.32, 0.38, 0.43 | 20 |
| `max` | 0.5124, 0.6007, 0.6925, 0.7406, 0.8174 | 0.3818, 0.4531, 0.5064 | 15 |
| `top3` | 0.4942, 0.5703, 0.6415, 0.6747, 0.7514 | 0.3751, 0.4409, 0.4784 | 15 |

Each grader has a different natural range, so the non-production graders reuse the **same
percentile positions** as the production anchors rather than the same absolute numbers —
combos are compared at equal routing rates, not at arbitrary cutoffs.

Objective: **NDCG@10**. Constraint: retry rate < 20%. Faithfulness was deliberately *not*
the objective — the generator is faithful to whatever chunks it receives, so it sits near
0.95 regardless of routing and cannot discriminate between combos.

### Score distribution

Grading uses mean Milvus cosine across all 20 retrieved candidates. Over the golden set:

```
p0 0.3205   p25 0.4289   p50 0.5125   p75 0.6030   p90 0.6470   p100 0.7091
```

**A HIGH threshold of 0.75 is unreachable on this catalogue.** An earlier configuration
documented 0.75/0.45; it would have produced zero `synthesize` decisions and routed every
query through a rewrite. The lesson generalises: *compute the achievable range before
setting a threshold.*

The distribution is also tight — moving HIGH by 0.05 reroutes 16–18 of 100 queries. That
brittleness matters: modest data drift changes routing behaviour with no code change.

### Results

| Config | NDCG@10 | Recall@10 | Retry rate | Expected added latency |
|---|---:|---:|---:|---:|
| `mean20 / 0.45 / 0.10` (pre-calibration) | 0.8459 | 0.6983 | 0.30 | 309.6 ms |
| **`mean20 / 0.45 / 0.43` (selected)** | **0.8468** | 0.6993 | **0.05** | **51.6 ms** |
| `mean20 / 0.50 / 0.43` | 0.8483 | 0.6999 | 0.23 | 237.4 ms |

Bootstrap, 1,000 resamples with threshold selection re-run on each:

- Improvement over the pre-calibration config: **+0.0058, 95% CI [+0.0009, +0.0140]** — the
  interval **excludes zero**.
- Selection stability: the winning combo took only **21.2%** of resamples.

**The selected config is a real improvement, but the precise optimum is not identified.**
The direction is solid — raising LOW to 0.43 beats the previous setting outside the noise
band while cutting the retry rate from 30% to 5% and expected added latency from 310 ms to
52 ms. Which combo is *best* is near-tied: several sit within 0.002 NDCG of each other and
none wins a majority of resamples. Reported as such rather than presenting a point estimate
as settled.

The higher-NDCG alternatives all violate the retry-rate constraint — `mean20 / 0.50 / 0.43`
buys +0.0015 NDCG for 4.6× the retry rate and 185 ms of latency.

The grid was built *before* the final labelling round, deliberately: pooling over the union
of both candidate sets removes a bias that would otherwise penalise combos for surfacing
unjudged products, since retry returns rewritten-query results that are likelier to be
unlabelled.

Confirmation against the live service reproduced the offline replay to four decimal places
(0.6931 simulated / 0.6930 live, on the label set current at the time), validating the
cache-and-replay method itself.

---

## What didn't work

CRAG is retained in the codebase and this section documents why it underperforms, rather
than quietly deleting the experiment.

**1. The corrective action is net harmful.** Across 100 queries the rewrite *improved* the
retrieval grade for 24 (mean +0.0246) and *degraded* it for 75 (mean −0.0275).

**2. Root cause: the rewrite introduces no new information.** In the original CRAG paper the
corrective action on failed retrieval is a *web search* — new knowledge from outside the
corpus. Here it rephrases the query and searches the same 5,000-product index. If the
product is not in the catalogue, no rephrasing will find it. The user's original phrasing is
usually the most accurate expression of intent, so LLM rewriting mostly adds semantic drift.

**3. The grader is a weak signal.** Rank correlation against true NDCG@10:

| Grader | Pearson | Spearman |
|---|---:|---:|
| `mean20` (production) | 0.393 | 0.443 |
| `max` | 0.490 | 0.543 |
| `top3` | 0.457 | 0.513 |

`max` correlates better because averaging across all 20 candidates lets the weak tail —
weak for good *and* bad retrievals, therefore nearly information-free — pull every query
toward the middle. **But `max` did not win the grid.** Better correlation with the *proxy*
did not translate into better end-to-end NDCG, so the production grader was kept on that
evidence. Optimising a proxy and optimising the target are different things.

**4. The retry loop cannot improve on its second attempt.** `run_crag()` rewrites the
*original* query every attempt and never updates it, so with `temperature=0` attempt 2
reproduces attempt 1 exactly and burns ~1 s for nothing. Retained as a known issue.

**5. Structural limit.** CRAG pays off when the corpus is large and heterogeneous and an
external knowledge source can be consulted. On a single-domain closed catalogue of 5,000
products, a product either exists or it does not, and the headroom for a corrective loop is
small. Here the larger gains are in retrieval itself — chunking, embeddings, query
understanding — not in correction.

**6. A metric bug that survived code review.** `_MAX_CHUNKS_PER_PRODUCT = 2` let one product
occupy several result slots, and nothing collapsed chunks back to one row per product. Six
queries scored **above the theoretical maximum of 1.0** (one reported Recall 2.0, NDCG 1.5)
because relevance was double-counted in the numerator while the denominator counted it once.
Fixed by max-pooling chunk scores up to the product; `eval/metrics.py` now raises on
duplicate IDs. **Impossible values are the cheapest bug detector available** — the assertion
found what review had not.

**7. Reviews were built but never indexed.** `ingestion/chunker.py` implements
`chunk_review()` with a 20-word floor and one-chunk-per-review policy, and the Milvus schema
carries a `chunk_type` field for `"description"` or `"review"`. But `data/run_ingest.py` —
the seeding script — reads only `rag_products`, calls only `chunk_description()`, and
hardcodes `chunk_type: "description"`. The only caller of `chunk_review()` is the Kafka
consumer, which no review has ever been published to.

**8. BM25 score concentration is not a usable query router.** A high AUC hid an unusable
operating point, and the negative control showed the hypothesised mechanism was wrong.
Details below.

### Reviews are not indexed

All 6,337 vectors are product descriptions. Verified three ways: the seeding script's code
path, the vector count (6,337 / 5,000 products ≈ 1.27 chunks each, consistent with
descriptions alone), and the 4,000 retrieved candidates recorded in
`eval/calibration_cache.json`, of which **4,000 are `chunk_type: "description"` and zero are
reviews**.

Two consequences worth stating plainly:

- **The reranker's business rule is currently a no-op.** `_DESCRIPTION_BOOST = 0.05` is meant
  to rank descriptions above reviews at equal cross-encoder score. With no review chunks in
  the index it adds a constant to every candidate, which changes no ordering at all. The
  pipeline's one business rule does nothing in production.
- **It is a plausible cause of the weakest numbers.** Exploratory and edge queries score
  worst (NDCG 0.8157 / 0.7214 against 0.9393 for navigational). Questions like *"does it run
  small?"* or *"is the fabric see-through?"* are answerable from reviews and unanswerable
  from a marketing description — so the missing corpus and the weak query classes line up.

64,744 reviews sit in the `rag_reviews` PostgreSQL table, ingested and unused.

### BM25 concentration cannot route navigational queries

Generation is 69% of request latency, and navigational queries score NDCG@10 0.9359 — the
ranked products already answer them. Skipping the generator for that class would remove
~1817 ms outright instead of hiding it behind streaming. The service cannot see a query's
`type` at request time, so it needs a proxy.

**Hypothesis:** a navigational query names one specific product, so its BM25 scores should
concentrate on a few documents; an exploratory query matches many products weakly.

Tested against the golden set's `type` labels (20 navigational / 80 other) with a
2000-sample bootstrap, mirroring the CRAG calibration. Unlike the CRAG thresholds, these
labels were assigned when the set was written, not derived from the system under test.

| Feature | AUC | 95% CI |
|---|---:|---|
| `max_score` — raw top-1 BM25 | **0.924** | [0.866, 0.969] |
| `top1_ratio` — top-1 share of the top-50 mass | 0.863 | [0.787, 0.926] |
| `entropy` over top-50 (negated) | 0.844 | [0.757, 0.919] |
| `top1_over_next9` | 0.834 | [0.740, 0.919] |
| `gap_ratio` — (s₁ − s₂)/s₁ | 0.807 | [0.685, 0.904] |
| `n_tokens` — *negative control* | 0.416 | [0.287, 0.552] |

**AUC 0.924 looks deployable. It is not.** Three reasons, in order of how much they matter:

**Every concentration feature loses to raw magnitude.** If the mechanism were "one document
dominates," `top1_ratio` and `gap_ratio` would lead. They trail. The best feature is the
absolute score, which is a function of term rarity (IDF) — so what separates the classes is
*rare words*, not concentration. The hypothesis was wrong even though the numbers looked
right.

**BM25 cannot distinguish a rare brand from a rare product noun.** The one false positive
above the best usable threshold is `silky durag for 360 waves with long tail straps`
(`max_score` 62.9) — an *attribute* query outranking all 20 navigational ones. `durag` is as
rare a token as `MEROKEETY`. At the level of term statistics they are the same object, so no
BM25 feature can separate them; the distinction is semantic and lives above the retrieval
score.

**The base rate destroys the precision.** At 20% prevalence the best threshold holding
precision ≥ 0.80 is `max_score ≥ 54.10`: recall 0.30, skipping generation on 7 queries per
100 of which 1 is wrong — a mean saving of **127 ms** across all traffic. Pushing to full
recall drops precision to 0.56. Precision is the metric that matters here because the two
errors are not symmetric: a false negative forfeits a saving, a false positive silently
drops the prose from a query that needed it. A 1-in-7 chance of the second, for 127 ms, is
not a trade worth making.

The negative control is load-bearing. BM25 sums over query terms, so a longer query scores
higher mechanically, and the whole result could have been an artefact of navigational
queries being wordier. `n_tokens` at AUC 0.416 (and `max_score` per token still at 0.915)
rules that out — the signal is real, it is simply the wrong signal.

Reproduce with `python eval/probe_navigational_router.py`; it needs only PostgreSQL and
takes a few seconds. Full per-query output in `eval/navigational_router_results.json`.

---

## Scale and cost

The system runs at laptop scale — 5,000 products, 6,337 vectors, no production traffic.
This section states what that costs today and what breaks first when it grows, because
"it works on 5,000 products" and "it works" are different claims.

### Cost per query

Token counts are measured from the pipeline (`tiktoken`, `cl100k_base`) against the golden
set; **unit prices change and should be re-checked before quoting a bill.**

| Stage | Calls | Tokens | Notes |
|---|---|---|---|
| Embedding | 1 | ~15 in | `text-embedding-3-small`; skipped on a Redis cache hit |
| Guardrail | 1 | ~40 in / 5 out | `max_tokens=5` — "YES"/"NO" is one token |
| Generation | 1 | ~735 in / ≤300 out | Context is top-5 chunks; measured mean 131 tokens per chunk |
| Query rewrite | 0.05 | ~30 in / 60 out | Retry path only — 5% of queries after calibration |

**~775 input and ≤305 output tokens per query, across three LLM round-trips.**

Two properties worth noting. Generation dominates by an order of magnitude, so any cost work
starts there — `top_k` is the lever, since context scales linearly with it. And the embedding
cache only helps on *exact* repeat queries; it does nothing for the generation cost, which is
where the money is. A response-level cache (future work) is a cost optimisation before it is
a latency one.

### What breaks first, in order

**1. The BM25 index, at roughly 10⁵–10⁶ products.** It is a `BM25Okapi` object held in
process memory and **rebuilt from PostgreSQL on every startup**. Today that is 3.5 MB of raw
text over 5,000 products and a sub-second build. At a million products it is both a slow
cold start and a per-replica memory cost that scales with the catalogue rather than with
traffic — every replica holds a full copy. This is the first thing to break and the first
thing to move out of process: Elasticsearch/OpenSearch, or Postgres full-text search, both
of which make the sparse index a shared service instead of a per-instance liability.

**2. Milvus memory, at roughly 10⁷ vectors.** 1536-dimensional `float32` vectors are 6 KB
each: 6,337 vectors is 39 MB, but 10 million is **61 GB of raw vectors** before HNSW graph
overhead, which adds materially at `M=16`. Mitigations in increasing order of disruption:
scalar quantisation or `float16` (halves it), an IVF-family index (trades recall for
memory), dimensionality reduction via Matryoshka-style truncation of the embedding, or
partitioning by category so each search touches a slice.

**3. The golden query set, at any scale.** 100 queries is already thin — most per-type
breakdowns rest on 20–30 queries, so a per-type difference of a few points is inside the
noise. It does not break loudly; it silently stops being able to detect regressions as the
catalogue diversifies. Growing the catalogue without growing the golden set is the failure
mode nobody notices.

**4. Cost, well before infrastructure.** Nothing about the architecture prevents scaling to
100 QPS, but at three LLM calls per query the bill scales linearly with traffic while
infrastructure cost stays roughly flat. At meaningful volume the correct move is not more
replicas — it is to stop calling an LLM on every request: cache responses, and skip
generation entirely for navigational queries where the product list *is* the answer. The
cheap version of that routing decision has already been tested and rejected — BM25 score
concentration does not identify the class precisely enough (*What didn't work*, item 8) —
so it needs a real intent signal, which is why it is future work rather than a quick win.

**What does *not* break early:** the cross-encoder reranker always sees exactly 20
candidates regardless of catalogue size, so its ~120 ms is constant. Retrieval is
`O(log n)` through HNSW. The pipeline's expensive stages are bounded by design, which is
the point of narrowing before reranking.

### Stability and failure modes

Current behaviour under dependency failure, and where it is wrong:

| Dependency | On failure | Assessment |
|---|---|---|
| Redis | Cache silently disabled; every query calls OpenAI | Correct — degrade, don't fail |
| Guardrail LLM | 3 s timeout, **fails open** — query proceeds | Correct for a search box; off-topic queries can slip through |
| Milvus | Service starts degraded; `/query` returns empty | Loud in `/health`, but a **dense-retrieval outage silently becomes a BM25-only search** rather than an error |
| PostgreSQL | Hard failure at startup after 3 retries with backoff | Correct — BM25 cannot be built without it |
| **OpenAI generation** | **No fallback — the request fails** | **The real gap.** Retrieval succeeded; the answer is what failed. Returning ranked products without prose would be strictly better than a 5xx |
| Kafka | `/ingest` unavailable | Fine — queries are unaffected |

The load test found **0% failures at 1/5/10/20 concurrent users**, but that was against a
healthy OpenAI. The untested case is partial degradation — slow rather than absent
upstream — which is the failure mode that actually happens in production.

Two known gaps beyond that: there is **no rate limiting or per-caller quota**, so a single
client can run up an unbounded LLM bill; and **p95 is 6.4 s at 10 concurrent users**, which
is not shippable for interactive search regardless of how the infrastructure scales. Users
abandon search well before that. The fix is not capacity — it is streaming the answer, or
skipping generation for the query classes that do not need it.

---

## Future work

### Retrieval and evaluation

**Index the reviews.** The largest single piece of unused capability in the project — 64,744
reviews already sitting in PostgreSQL, with the chunker written and the schema field in
place. Wiring `chunk_review()` into `data/run_ingest.py` is a few lines; the reason it is a
session rather than a patch is everything downstream. The index grows roughly 10×, which
changes ANN recall behaviour and rerank candidate mix; **every metric moves, so the locked
baseline, the calibration cache and its fingerprint are all invalidated**, and the golden set
needs another adjudication round because reviews will surface products no round has judged.
It would also make the description boost meaningful for the first time, and is the most
likely lever on the exploratory and edge queries that score worst today. Sequence it the same
way as round 3: rebuild the cache first, pool over the union, then re-lock.


**Reconsider the grader.** `_grade()` averages cosine over all 20 candidates, so the weak
tail compresses the achievable range to 0.32–0.71 — which is why HIGH had to be tuned to
0.45 rather than the 0.75 the design assumed. Grading on the max, or the top-5 mean, would
give a more separable signal. It invalidates the calibration cache and turns threshold
tuning into a grader redesign, so it belongs in its own session.

**Move the grader after the cross-encoder.** The pipeline already runs a cross-encoder,
which is a far stronger relevance signal than cosine. Reordering to
`retrieve → rerank → grade → retry` would cost ~120 ms on the retry path against the 886 ms
the rewrite currently costs, and the forward pass is already paid for on the happy path.

**Correct by failure mode.** One action (rewrite) is currently applied to every failure.
Relaxing metadata filters, decomposing multi-constraint queries, and honestly reporting
absence are different corrections for different causes.

**Report CRAG on the subset where it fires.** Whole-set NDCG understates it — the loop
affects ~30% of queries, so a real improvement there is diluted roughly threefold.

**Second annotator + Cohen's kappa.** The largest methodological gap, and the one no
statistical technique fixes.

### Cloud migration

**Response-level cache.** Redis currently caches embeddings only (`emb:{sha256}`, TTL 1 h).
A cache on the full `QueryResponse`, keyed on canonicalised `(query, filters, top_k)` and
checked *before* the guardrail, makes a repeat query nearly free — caching after the
guardrail still pays ~50 ms and an API call. Adds cache hit rate as a second metric. Half a
day.

**Skip generation where prose adds nothing.** Navigational queries score NDCG@10 0.9359 —
retrieval is near-perfect and the ranked products *are* the answer, so the ~1817 ms
generation call buys little. Streaming hides that cost; skipping it removes it. The
obstacle is runtime classification: query `type` exists only as evaluation metadata. Next
step is an offline test of whether BM25 score concentration separates navigational from
exploratory queries, scored against the golden set's `type` labels — which, unlike the CRAG
relevance judgments, were not self-assigned. Same method as the threshold calibration: test
the signal before shipping it.

**Deploy to AWS.** App Runner for the container, RDS for Postgres, ElastiCache for Redis,
and **Zilliz Cloud** for vectors. Zilliz is the managed version of Milvus built by the same
team, so `pymilvus` and `hybrid_search()` are untouched — the migration is a URI and a token.
The alternative, OpenSearch Serverless, would mean rewriting retrieval against the k-NN API
*and* carries a minimum capacity floor that runs to hundreds of dollars a month regardless
of traffic. Two known obstacles: `sentence-transformers` downloads the ~90 MB cross-encoder
at startup, so it must be baked into the image or cold starts run 30 s+; and App Runner's
VPC connector routes *all* outbound traffic through the VPC, so reaching OpenAI and Zilliz
needs a NAT Gateway — roughly $32/month, which would be the largest line item in the
deployment. Kafka is dropped from the deployed profile: it serves `/ingest` only, and MSK
costs more than the feature is worth here.

**Reranker to a SageMaker endpoint.** The latency breakdown already points at the
cross-encoder as the largest local cost, which makes this an evidence-driven move rather
than a résumé-driven one. The honest expectation is that a SageMaker CPU endpoint will be
*slower* than local CPU — a network hop added to save nothing. The real case for a separate
inference service is independent scaling and GPU access, not single-user latency, and the
result will be reported whichever way it lands.

**Tiered CI regression gate.** The full harness is ~4 LLM calls per query across 100
queries. Running it on every push scales cost and CI time with push frequency, not with
signal. Instead: a 25-query stratified smoke subset per push asserting on **hard failures
only** — exceptions, empty results, schema violations, latency blowout — and the full 100
with faithfulness nightly and on release, where the 5% metric gate lives. On 25 queries a 5%
NDCG delta is inside the noise band, so gating on it there would generate false alarms. The
principle: **match the assertion to the sample size.** Small fast samples catch breakage;
only large samples catch quality regressions.

**pgvector consolidation.** Longer term, folding the vector store into the RDS instance
already required removes a managed service and a vendor. pgvector is a mainstream production
choice and Postgres full-text search could in principle absorb the BM25 index too, collapsing
three components into one. Framed correctly this is cost- and operations-motivated
consolidation, which is a better reason than swapping vector databases to prove it can be
done.

---
---

# Reference

Everything below is operational detail rather than design discussion.

## Setup

**Prerequisites:** Python 3.11+, Docker

### 1. Start infrastructure

Redis, PostgreSQL, Milvus, Kafka and Gorse run via the shared compose file:

```bash
cd fashion-recommend && docker-compose up -d && cd ../rag-service
docker ps --format "table {{.Names}}\t{{.Status}}"
```

`fashion-redis`, `fashion-postgres`, `fashion-milvus` and `fashion-kafka` should all be
`healthy`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Conda users:** use the full path to hit the right environment —
> `/opt/homebrew/anaconda3/bin/pip install -r requirements.txt`

### 3. Configure

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` at minimum. All other defaults match the Docker Compose setup.

### 4. Seed the vector index

Once only. Reads products from PostgreSQL, chunks descriptions, embeds with
`text-embedding-3-small`, writes to the Milvus `fashion_rag` collection:

```bash
python ingestion/indexer.py --seed
```

~10–15 minutes for 5,000 products (OpenAI rate-limited). The BM25 index rebuilds from
PostgreSQL on every startup — no separate seeding step.

### 5. Run

```bash
uvicorn main:app --reload --port 8002
```

---

## API

### `POST /query`

```json
{
  "query": "casual blue summer dress under budget",
  "filters": { "price_range": "budget", "category": "dresses" },
  "top_k": 5
}
```

`filters` is optional. Supported: `price_range` (`budget` / `mid` / `premium`), `category`,
`occasion`, `brand`, `chunk_type`.

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
  "degraded": [],
  "latency_ms": {
    "guardrail_ms": 480, "retrieval_ms": 22, "crag_ms": 0,
    "rerank_ms": 95, "generation_ms": 1340, "total_ms": 1937
  }
}
```

`retrieval_path` is `"synthesize"` (high confidence), `"retry"` (query rewritten),
`"best_effort"` (low confidence — real candidates, weak signal), or `"fallback"` (retrieval
returned nothing; trending products).

`degraded` lists components that failed during the request — empty on the healthy path.
Degradations are surfaced rather than raised, so a partial outage returns partial results
instead of a 5xx:

| Value | Meaning |
|---|---|
| `milvus_unavailable` | No Milvus client at startup — dense retrieval skipped |
| `milvus_search_failed` | Milvus raised mid-query (collection unloaded, network drop) |
| `embedding_failed` | `embed()` timed out or the API errored |
| `bm25_only` | Results came from the sparse index alone; no semantic matching |
| `generation_failed` | Products are real and ranked; the prose is a fallback sentence |

### `POST /query/stream`

Identical request body and pipeline as `/query`; the answer arrives as Server-Sent
Events instead of a single JSON body. Use it when a client can render incrementally —
products appear at ~527 ms rather than ~2.4 s.

```
event: products
data: {"products": [...], "retrieval_path": "synthesize", "degraded": []}

event: token
data: {"text": "Here are"}

event: done
data: {"cited_sources": ["prod_0042"], "degraded": [], "latency_ms": {...}}
```

An off-topic query is still rejected with a plain `400` — retrieval runs before streaming
begins, so the status code is not yet committed. If generation fails *after* products have
been sent, an `event: error` is emitted and the stream still ends with `done` carrying
`degraded: ["generation_failed"]`.

`cited_sources` cannot be sent with `products`: resolving `[n]` references requires the
complete answer text.

### `POST /ingest`

Queues products for async re-indexing via Kafka. `{ "product_ids": [...] }`, or omit to
re-ingest everything. Returns `202 Accepted` immediately.

### `GET /health`

Per-dependency status. `"degraded"` if any is unreachable — the service starts without
Milvus (queries return empty) and without Redis (caching silently disabled).

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Embeddings, generation, and the faithfulness judge |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `fashion_rag` | Do not change after the index is built without a full re-embed |
| `POSTGRES_URL` | `postgresql://gorse:gorse_pass@localhost:5432/gorse` | BM25 index is built from `rag_products` at startup |
| `REDIS_URL` | `redis://localhost:6379` | Embedding cache (`emb:`, TTL 1 h) and guardrail verdict cache (`guard:`, TTL 24 h). Optional |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Only for `POST /ingest` |
| `KAFKA_INGEST_TOPIC` | `rag-ingest` | Async ingestion topic |
| `CRAG_HIGH_THRESHOLD` | `0.45` | Grade at or above this → synthesize directly |
| `CRAG_LOW_THRESHOLD` | `0.43` | Grade below this → `best_effort` (real results, **not** trending) |
| `CRAG_MAX_RETRIES` | `2` | Max query-rewrite retries |
| `CRAG_TIME_BUDGET_S` | `3.5` | Wall-clock cap on the CRAG loop |

Thresholds were selected empirically — see
[CRAG Threshold Calibration](#crag-threshold-calibration-2026-08-06).

---

## Tests

127 tests in about 10 seconds, no external services — OpenAI and Milvus are mocked.

```bash
pytest tests/ -v
pytest tests/test_retrieval.py -v
pytest tests/test_guardrail.py::test_timeout_defaults_to_true -v
```

```
tests/
├── conftest.py          # shared fixtures, mock OpenAI client
├── test_guardrail.py    # fashion vs off-topic classification
├── test_retrieval.py    # hybrid BM25 + Milvus + RRF fusion
├── test_crag.py         # path routing, time budget, retry logic
├── test_reranker.py     # score ordering, per-product dedup
├── test_generator.py    # grounding prompt, cited_sources
├── test_chunker.py      # sentence-aware chunking
├── test_metrics.py      # NDCG/Recall correctness + duplicate guard
└── test_api.py          # end-to-end with mocked pipeline
```

---

## Startup dependency order

1. **PostgreSQL** — BM25 index built at startup. Hard failure if unreachable; retries 3× with exponential backoff before aborting.
2. **Milvus** — vector search. Logs `CRITICAL` and starts degraded if unreachable.
3. **Redis** — embedding cache. Optional; silently disabled if unreachable.
4. **OpenAI** — per-request, not at startup. Each `embed()` call has a 5 s timeout.

Kafka is only needed for `POST /ingest`.

---

## Embedding cache

`embed()` in `pipeline/utils.py` caches query vectors in Redis. Key `emb:` + first 24 hex
chars of `SHA256(query_text)`, TTL 1 hour. Hit ~3 ms; miss 300–3000 ms.

Keyed on the **exact** query string — two queries meaning the same thing in different words
each miss and call OpenAI separately. It pays off for repeated identical searches: refreshes,
retries, autocomplete debounce.

---

## Data sources

Seeded from the [Amazon Fashion dataset](https://amazon-reviews-2023.github.io/) (~800K
reviews, 5,000-product subset). Raw data in `amazon_data/raw/`.

- **Descriptions:** sentence-aware sliding window, 256 tokens, ~50-token overlap —
  **this is the only content currently in the vector index**
- **Reviews:** `chunk_review()` produces one chunk per review and drops anything under 20
  words, but nothing calls it during seeding. See
  [Reviews are not indexed](#reviews-are-not-indexed).

---

## Docker

```bash
docker build -t rag-service .
docker run --env-file .env --network fashion-recommend_default -p 8002:8002 rag-service
```

NLTK `punkt` data is downloaded at build time, so there is no network call on cold start.

---

## Load test results (2026-07-11)

Run against the 100-query golden set — a small fixed query set would inflate results through
the Redis embedding cache. 0% failures at every level.

| Users | Requests | Fail% | p50 | p95 | p99 | RPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 0% | 4,100 ms | 9,700 ms | 12,000 ms | 0.15 |
| 5 | 153 | 0% | 3,300 ms | 6,800 ms | 9,800 ms | 0.86 |
| 10 | 323 | 0% | 3,300 ms | 6,400 ms | 8,200 ms | 1.81 |
| 20 | 605 | 0% | 3,600 ms | 7,000 ms | 8,700 ms | 3.40 |

- **The bottleneck is OpenAI latency (~3–4 s/call), not local infrastructure.** p50 stays
  flat across concurrency because requests fan out independently.
- **p99 improves as concurrency rises** (12 s → 8.7 s) as embedding-cache hits warm up
  across the query pool.
- **RPS scales linearly** — no saturation at 20 users. The degradation point was never
  reached; finding it needs 50–100 users.

---

## Common commands

```bash
# Infrastructure
cd fashion-recommend && docker-compose up -d && cd ../rag-service

# Seed (first time, ~10-15 min)
python ingestion/indexer.py --seed

# Run
uvicorn main:app --reload --port 8002

# Tests
pytest tests/ -v

# Evaluate against the golden query set
python eval/run_eval.py --golden-set eval/golden_queries.json

# Regression gate
python eval/check_regression.py --threshold 0.05

# Threshold calibration
python eval/calibrate_crag.py --build-cache
python eval/calibrate_crag.py --grid

# Load test
locust -f eval/locustfile.py --host http://localhost:8002 \
  --users 10 --spawn-rate 10 --run-time 3m --headless --csv eval/load_test_u10
```
