# fashion-recommend

A recommendation stack for Amazon Fashion, built on [Gorse](https://github.com/gorse-io/gorse).

This README leads with a measurement rather than an architecture diagram, because the
measurement is what determined the architecture.

---

## 1. The constraint this system is designed around

Before building anything, the interaction graph was measured. Over the full McAuley
Amazon Fashion dump — 2,500,939 reviews, 2,035,490 distinct users:

| | |
|---|---|
| Mean interactions per user | **1.23** |
| Users appearing exactly once | **86.18%** |
| Iterative 5-core | **empty** |
| Iterative 3-core | 2,223 users (0.11%) |

The 5-core is not small. It is empty. There is no subgraph in which every user has five
interactions and every item has five interactions.

**Collaborative filtering is undefined for roughly 98% of this population.** Two users can
only be similar through a co-interacted item, and at 1.23 interactions per user those
overlaps barely exist. This is not a tuning problem; it is the shape of the domain.

That single fact sets the direction:

> The system is not organised around making CF better. It is organised around the
> **cold-start path** — content labels, semantic retrieval, and LLM-extracted user
> traits — and every metric is reported for warm and cold cohorts separately, because a
> single average would conceal the split entirely.

### A measurement detail that changes the answer

k-core must be computed over **deduplicated `(user, item)` edges**, not raw events. The
corpus contains 26,564 duplicate edges, and counting them shifts every degree threshold.
An earlier pass that counted events reported a 5-core of 428 users; on distinct edges the
same computation returns empty.

Reproduce with:

```bash
python3 data/build_interactions.py --stats
```

---

## 2. Dataset and split protocol

### Two catalogues, deliberately not shared

| table | rows | purpose | consumer |
|---|---|---|---|
| `rag_products` | 5,000 | retrieval corpus | rag-service (BM25 + Milvus) |
| `reco_products` | 95,335 | evaluation catalogue | Gorse |
| `reco_interactions` | 558,940 | evaluation interactions | Gorse |

The obvious simplification — grow `rag_products` to 95k and use one table — was rejected.
That table is rag-service's retrieval corpus, and BM25 is rebuilt from it at startup.
Growing it 19× would change retrieval for every query and, worse, make 1,481 locked
relevance judgments incomplete: products entering the corpus unjudged count as
non-relevant and depress NDCG for reasons unrelated to retrieval quality. That damage is
not repairable by re-running the evaluation; it requires re-adjudication.

### Why not leave-last-out

Leave-last-out is the textbook protocol for implicit feedback. It was measured and
rejected:

| protocol | train events | items with any training signal | evaluable warm test users |
|---|---|---|---|
| **global temporal cutoff** (q=0.70) | 391,258 | 62,006 (**65%**) | 859 |
| leave-last-out | 30,798 | 17,644 (**18.5%**) | 3,420 |

On a corpus where 86% of users appear once, leave-last-out sends 95% of events to test and
leaves **81.5% of the catalogue with no training signal at all**. An item with no training
interaction cannot be recommended by CF, popularity, or item-to-item — so catalog coverage
would be structurally capped at 18.5%, and Gini computed over a truncated universe. The
beyond-accuracy metrics would be measuring the protocol, not the recommender.

A single global cutoff is also **more** leak-proof: leave-last-out will train on a 2022
event while testing a 2015 event belonging to a different user. One cutoff makes every
training event older than every test event, and `--build` asserts it.

The cutoff q=0.70 (2020-11-23) was chosen because the evaluable warm cohort peaks there
(704 → 859 → 843 across q=0.40…0.75) while item coverage keeps rising.

> The warm cohort is ~859 users under either protocol. That is the corpus, not a bug.
> Cold-start is where the volume is: **156,935 cold test users**.

### Feedback taxonomy

Star ratings and verification status are mapped to intent-graded feedback types rather
than Gorse's default star schema:

| condition | Gorse type | trained on |
|---|---|---|
| verified, rating ≥ 4 | `purchase` | positive |
| verified, rating = 3 | `view` | read |
| rating ≤ 2 | `dislike` | **neither** |
| unverified | `view` | read |

**The explicit-negative signal is stored but inert.** `config.toml` lists
`positive_feedback_types = ["purchase", "favorite", "add_to_cart"]` and
`read_feedback_types = ["view"]`; `dislike` is in neither. After the push Gorse reports
276,194 positive + 51,280 read = 327,474 against 391,258 events sent — the ~64k difference
is exactly the dislikes. Whether including them helps is the knob the trait ablation turns;
it is not a bug to fix silently.

Only the train split is loaded into Gorse. `--verify-gorse` samples users and asserts that
no test event reached it.

---

## 3. System composition

| component | role | store |
|---|---|---|
| Gorse master | trains CF and CTR models, computes similarity, writes caches | — |
| Gorse server | serves recommendations — **reads cache only, never computes** | — |
| Go API (Gin) | auth, social features, proxy to the agent | host Postgres |
| React SPA | UI | — |
| PostgreSQL (host) | traits, conversations, social data, both catalogues | source of truth |
| PostgreSQL (Docker) | Gorse's own users / items / feedback | source of truth |
| Redis Stack | Gorse's computed caches (RediSearch required) | derived, recomputable |
| Milvus | product embeddings for rag-service | derived |

**There are two Postgres instances.** Gorse's data store is the Docker one, which
deliberately does not publish 5432; the application and evaluation tables live on the host.
Same credentials, different databases. Counting Gorse's tables from a host connection
returns zero rows and looks like data loss.

### Algorithms

Trained on this data:

- **Matrix factorization** (collaborative filtering) — measured as ineffective on this
  corpus, which is the expected result given §1 and is reported rather than hidden
- **Factorization machine** (CTR ranker) — learns crosses between user labels and item labels
- **Tags item-to-item** — IDF-weighted label overlap, HNSW-indexed. The primary cold-start
  recall source
- **Non-personalized** — trending / popular / new arrivals, the fallback when nothing else applies

`user-to-user` is currently disabled. It sits sequentially ahead of item-to-item in the
master's task chain, carries 1,484,868 units of work, and appears in none of the evaluation
arms — and per §1, user-based similarity has almost no signal to find here.

### The online path does no inference

A recommendation request reads a precomputed list out of Redis. All computation happens in
the master's offline task cycle: the CF model retrains every 24h, the recommendation caches
regenerate hourly.

> That hourly regeneration is a hazard for evaluation, not a feature: a harness reading
> those caches can have them change mid-run. Stop the task loop before an evaluation run.

---

## 4. Evaluation

**Not yet run.** The harness is designed and its disciplines are fixed; the numbers are not
in yet. What is decided:

- Metrics: NDCG@10, Recall@10/20, HitRate@10, MRR, plus catalog coverage, Gini,
  intra-list diversity, novelty
- Arms: Random, MostPopular, Gorse CF, Gorse item-to-item, and item-to-item with trait labels
- Every metric reported for warm and cold cohorts **separately** — no single average
- **Random and MostPopular are computed by the harness from the full interaction table,
  not from Gorse.** The evaluation cohort has to be sampled to fit a memory ceiling
  (§5), and a popularity baseline computed from a sampled recommender would move with
  the sampling — a baseline whose value depends on the independent variable is not a baseline
- `evicted_keys == 0` is a hard precondition, not a log line: cache eviction silently
  depresses retrieval metrics without raising an error

For comparison, the sibling rag-service is fully evaluated and locked: NDCG@10 0.8468,
Recall@10 0.6993 against a structural ceiling of 0.7615, faithfulness 0.9580.

---

## 5. What does not work, and why

This section is the point of the README.

### 5.1 Seven components were written, shipped, and had never run

Fixing them was one day's work. Finding them was the hard part, because each failed
silently:

| defect | why nothing surfaced |
|---|---|
| Trait score embedded in the label string | Gorse indexes a label only on its **second** occurrence across users; scores made every label a singleton, so all were dropped |
| Emission threshold above the reachable score range | The extractor max-normalises then merges at `keyword*0.4 + ai*0.6`, so without the LLM leg the ceiling is 0.4 — under a `> 0.5` gate |
| Go client's JSON tags were snake_case | Gorse reads Go field names; `labels` matched case-insensitively but `user_id` never reached `UserId`, so **every user ever inserted landed under one empty id** |
| `column = "Labels"` in two similarity configs | `column` is an expr expression, not a column name. Failed to compile, per entity, while the dashboard reported the task `Complete` in `0.0s` |
| `duration('7d')` / `duration('30d')` | Go's `time.ParseDuration` has no `d` unit. **Compiled cleanly**, failed at evaluation |
| `float(now() - item.Timestamp)` | The subtraction yields a `time.Duration`, which expr's `float()` rejects. Also compiled cleanly |
| `product_likes` table never created | The migration file exists; nothing runs it. Every like returned 500 before reaching the feedback call |

Four disciplines came out of this, and they are enforced in tests:

1. **Verify what the store holds, not what the API returned.** A `RowAffected: 1` is
   compatible with a row that has no key.
2. **A round-trip mock cannot test a wire contract.** `api/server_test.go` encoded its
   response with the same struct the client decoded with, so a tag error cancelled itself
   out and 285 lines of tests stayed green. `models/models_wire_test.go` asserts literal
   JSON keys instead.
3. **Configuration expressions must be regression-tested by *evaluation*, not compilation.**
   Of the three expression bugs, only one failed at compile time. `config/fashion_config_test.go`
   evaluates every expression in the config; it found the third bug before the logs did.
4. **Task status, CPU usage, and health checks are not evidence of work.** A failing task
   reported `Complete 383,460/383,460`. A master whose Redis had died reported `Running`
   at 98% CPU. Redis answers `PING` with `PONG` while loading an RDB and rejecting every
   real command.

### 5.2 item-to-item returns neighbours, but its scores cannot rank them

Two separate findings, reported separately:

| | |
|---|---|
| Neighbours exist | 33 of 50 randomly sampled items |
| Scores usable for ranking | **No** |

Across 330 scores: global range 0.500689–0.510655 (span 0.00997), sd 0.002123, and the
**median spread inside a single top-10 is 0.000339**. Every score lives in 1% of [0,1].

The label vocabulary explains it. Sampling 3,000 items (5.49 labels each):

| prefix | occurrences | distinct | consequence |
|---|---|---|---|
| `item_name` | 2,999 | 2,186 | near-unique — never matches |
| `brand` | 2,993 | 1,492 | near-unique |
| `price_range` | 2,999 | **4** | ~80% are `mid` |
| `avg_rating` | 2,999 | 25 | coarse, **and semantically unrelated to similarity** |
| `style` | 1,533 | 6 | 51% coverage |
| `color` | 870 | 14 | 29% coverage |

**The labels that discriminate never match; the labels that match do not discriminate.**
Two arbitrary items typically share `price_range:mid` and `avg_rating:4.5`, and two items
both rated 4.5 are not similar in any sense that matters — so `commonCount` and `commonSum`
come out nearly equal for every pair.

The first hypothesis was the `(commonCount+100)` shrinkage term in Gorse's distance
function. **The data overturned it**: that term varies with overlap (k=2 → 0.020, k=10 →
0.091), so it creates spread rather than removing it. The cause is in this repo's label
generation, not upstream.

**Deliberately not fixed yet.** The fix is scheduled after the evaluation baseline is
locked, so the change can be reported as a before/after rather than made blind. Changing
labels inside a locked baseline round would make the two numbers incomparable — the same
trap as changing rag-service's corpus under its locked judgments.

### 5.3 A single machine bounds the evaluation cohort

The full per-user recommendation cache does not fit. With `cache_size` at its floor of 30:

```
383,460 items × 30 ≈ 11.5M entries    item-to-item completes
371,218 users × 30 ≈ 11.1M entries    does not fit
```

`cache_size` cannot go lower: evaluation includes Recall@20, and the recommender filters
the user's already-seen items first, so a cache of 20 can return fewer than 20 after
filtering and depress the metric **without erroring**.

Two properties of Redis make the ceiling lower than it looks:

- **RediSearch runs a fork-based GC.** Copy-on-write cost was measured at ~1.9 GB against a
  ~1.9 GB dataset — roughly 1:1 — so the container must be sized for **twice** `maxmemory`.
  Disabling RDB snapshots does not stop this; the module forks on its own schedule.
- **LRU eviction is designed for serving, not for batch builds.** While a job rewrites the
  whole cache, LRU discards what was just written, so a working set above the ceiling never
  converges — silently. One overnight run logged 13.1M evictions and completed nothing.

The consequence for the evaluation is scope, not correctness: the cohort is stratified
(all 859 evaluable warm users plus a sampled cold population) and the sampling is stated
in the results.

### 5.4 In-app feedback is plumbing, not a training signal

Frontend interactions reach Gorse, verified end to end:

| type | count | mechanism |
|---|---|---|
| `favorite` | 1 | heart button |
| `add_to_cart` | 1 | Add button |
| `view` | 6 | `IntersectionObserver`, 50% visible for 1 continuous second |

The `view` timestamps are the informative part: 6 impressions arrived in 3 pairs across 25
seconds on a page holding 20 cards. Without the dwell gate a single scroll would have
stamped all 20 at once, so the batching is evidence the throttle works rather than merely
that the request fires. `view` is a read feedback type, and unthrottled impressions dilute
the relative weight of positive signal.

**The models train on the 391,258 train-split events. A handful of demo clicks changes
nothing**, and the loop is presented as a wired pipeline, not as a data source.

---

## 6. Running it

```bash
# infrastructure
cd fashion-recommend && docker compose up -d

# apply migrations — nothing does this automatically
for f in database/migrations/*.sql; do
  PGPASSWORD=gorse_pass psql -h localhost -U gorse -d gorse -f "$f"
done

# API — the default port is 5000, but the frontend proxy expects 5001,
# and macOS usually holds 5000 for AirPlay Receiver
PORT=5001 go run main.go

# frontend
cd frontend && npm install && npm run dev
```

Tests:

```bash
go test ./...                          # from fashion-recommend/
go test ./config/ -run TestFashion     # from the repo root — config expression evaluation
python3 -m pytest data/ -v
```

Before any batch run on a memory-constrained host:

```bash
docker compose stop milvus kafka minio etcd            # frees ~650 MB
docker exec fashion-redis redis-cli CONFIG SET save ""  # resets on container recreate
```

After it:

```bash
docker exec fashion-redis redis-cli INFO stats | grep evicted_keys   # must be 0
```

---

## 7. Related

- [`../python-agent/`](../python-agent/) — LangGraph ReAct agent, HITL trait approval,
  two-model tiering with per-turn cost accounting
- [`../rag-service/`](../rag-service/) — natural-language product search; evaluated and
  baseline-locked
- [`CLAUDE.md`](../CLAUDE.md) — the operational companion to this document: traps,
  measured failure modes, and what not to repeat
