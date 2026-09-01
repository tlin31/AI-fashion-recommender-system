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

Two rounds, both over the same locked cohort of **162,562 evaluable users** (156,935 cold,
5,627 warm) and the same 95,335-item catalogue. Round 1 locked the baseline; round 2
measured a deliberate change to the item label schema (§5.2).

### 4.1 The headline: no personalized arm beats popularity

NDCG@10, cold and warm reported separately — never averaged, because a single number over
a population that is 96.5% cold is a report about cold-start with a personalization label
on it.

| arm | cold (156,935) | warm (5,627) |
|---|---|---|
| Random | 0.0000 | 0.0000 |
| **MostPopular** | **0.0192** | **0.0142** |
| Gorse CF (`/api/recommend`) | 0.0183 | 0.0046 |
| Gorse item-to-item | 0.0000 | 0.0015 |

Popularity wins in both cohorts. That is the result, not a placeholder for a better one.

Two things make it interpretable rather than merely disappointing:

**On cold users, Gorse CF *is* the fallback.** Its coverage (0.0002), Gini (0.0000) and
novelty (8.250 vs 8.237) match MostPopular digit-for-digit — it is serving the popularity
list through `[recommend.fallback]`. It then scores *below* our MostPopular, which is
precisely the contamination that motivated computing the baselines outside Gorse: a
popularity baseline read back from a sampled recommender measures the sampling.

**Recall is not ceiling-limited here.** Unlike the sibling rag-service, where Recall@10 is
structurally capped at 0.7615, this cohort averages 1.02–1.20 relevant items per user, so
`recall@10_ceiling` is 0.9995+. Nothing about the protocol is holding these numbers down.

### 4.2 What the label restructure moved, and what it did not

§5.2 rebuilt the item label schema. The comparison is trustworthy for one specific reason:
**Random and MostPopular reproduced bit-for-bit** across the two rounds. They are computed
in-harness from the host tables, so their invariance proves the cohort, exclude sets,
relevance data and scoring code did not shift — any movement elsewhere is attributable.

Warm cohort, before → after:

| arm | NDCG@10 | catalog coverage | users with ≥1 hit |
|---|---|---|---|
| Gorse CF | 0.0034 → **0.0046** | 0.2636 → **0.3341** | **38 → 55** |
| Gorse item-to-item | 0.0011 → 0.0015 | 0.3294 → **0.4154** | 14 → 17 |

Cold users did not move at all — every arm identical to four decimal places.

**The effect landed on the endpoint that serves users, not on the arm that was changed.**
`/api/recommend` gained 17 warm users with a hit (38 → 55, +45%); the item-to-item arm
itself gained 3 (14 → 17). The path is `[recommend.fallback]`, which lists
`item-to-item/style_similarity`: better labels reach the main endpoint through the fallback
chain. Coverage corroborates it — both arms rose by almost exactly the same proportion
(+26.8% and +26.1%), which is what a shared cause looks like.

**This is not cleanly attributable to labels alone, and is not claimed to be.** Redis lost
its cache before round 2 (§5.5), so the CF model retrained from scratch — two variables
moved on the warm cohort. One piece of evidence favours the label contribution: cold users
are identical digit-for-digit, meaning they never reach CF at all, so the retrain is a
no-op for them. On warm users the two causes cannot be separated by this experiment.

**item-to-item itself remains unusable, and now we know why it is not the labels.**
157,473 of 162,562 users get an empty list: they are cold, so there is no seed item whose
neighbours could be returned. Content similarity answers *"what resembles this item"*; the
task asks *"what will this person buy next"*. On a corpus where 86% of users appear exactly
once there is almost no bridge between the two questions. The value of the label work is
that it eliminates "the labels were broken" as an explanation — the representation is now
provably sound (§5.2) and the arm is still weak.

### 4.3 Reading numbers this small

The whole item-to-item arm rests on **14 users with a hit before and 17 after**, out of
162,562. Percentages over counts like that are theatre, so the harness reports the implied
user count alongside the rate and refuses to gate on strata that cannot support a
comparison.

That rule exists because this round exposed the gap. The gate initially FAILED on five
metrics in `warm/train=3-4` — NDCG −68.9%, Recall@20 −84.8% — which at n=212 is **three
users hitting becoming one**. `ABS_FLOOR = 1e-4` could not catch it: it floors the metric's
*value*, and 0.0142 is far above 1e-4 while representing three people. The missing guard
was on the metric's *support*. Strata below 30 hits are now reported in their own section
with their user counts, and never silenced — a genuine collapse looks identical at that
size and still needs a human read.

### 4.4 The disciplines, and what each one caught

| discipline | what it caught here |
|---|---|
| Baselines computed **outside** Gorse | Cold-user CF is the popularity fallback, scoring below our own MostPopular |
| Warm cohort **stratified** by training history | 4,768 of 5,627 "warm" users have ONE training event — near-cold, and aggregating them hides it |
| Denominators from the **source of truth** | Coverage is over the 95,335 seeded items, never Gorse's own count |
| `evicted_keys == 0` as a **precondition** | Eviction depresses retrieval silently, with no error |
| Metrics whose **definition changed** get no verdict | `FEATURE_LABEL_PREFIXES` gained `type:`/`cat:`, so ILD moved 0.4028 → 0.3475 over a different feature space. Reported without a direction; the comparable fact is `ild_item_coverage` 88.4% → **100%** |
| Minimum **support** before gating | The five false failures above |

`recall@10_ceiling` is reported every run for the same reason it mattered in rag-service:
a target above the ceiling is unreachable, and discovering that after the fact looks like a
quality problem rather than a measurement one.

For comparison, the sibling rag-service is fully evaluated and locked: NDCG@10 0.8468,
Recall@10 0.6993 against a structural ceiling of 0.7615, faithfulness 0.9580.

> The baseline is locked at `eval/baseline_metrics.json` (2026-08-31) and is **cohort-bound**.
> Day 5 changes the cohort for the trait ablation, which requires a fresh lock — the two
> re-locks are deliberately merged rather than done twice.

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

### 5.2 item-to-item's scores could not rank, and the cause was the vocabulary

Two separate findings, reported separately, because "it returns something" and
"what it returns is usable" are different claims:

| | |
|---|---|
| Neighbours exist | 33 of 50 randomly sampled items |
| Scores usable for ranking | **No** |

Across 330 scores: global range 0.500689–0.510655 (span 0.00997), sd 0.002123, and the
**median spread inside a single top-10 is 0.000339**. Every score lives in 1% of [0,1].

`Score = 1/(1+distance)` (`logics/item_to_item.go:127`), so 0.5007 means a distance of
0.997 — as close to "these two items share nothing" as the function goes. Reading the
conversion is what makes 0.50 legible as a floor rather than as a middling similarity.

**The labels that discriminate never match; the labels that match do not discriminate.**
Of the **107,747** distinct label strings the flat schema put on the similarity path,
**107,686 were `item_name` / `brand` / `price`** — near-unique per item, so they entered
the union of every pair and the intersection of none. Sixty-one strings were features.
Meanwhile the labels that *did* match were `price_range` (4 values, ~80% `mid`) and
`avg_rating` (25 coarse buckets) — and two items both rated 4.5 are not similar in any
sense a shopper would recognise.

The carriers were not merely inert. `item_to_item.go:337` divides the shared-tag weight
by `sqrt` of the weighted sum over *all* of an item's tags, and a near-unique label's IDF
is `log(95335)` ≈ 11.5 against ≈ 1.9 for a genuinely shared `cat:`. Two carriers therefore
dominated a norm the real features could not move.

The first hypothesis was the `(commonCount+100)` shrinkage term in Gorse's distance
function. **The data overturned it**: that term varies with overlap (k=2 → 0.020, k=10 →
0.091), so it creates spread rather than removing it. The cause was in this repo's label
generation, not upstream.

#### The fix, and the second failure it exposed

Labels moved from a flat array to Gorse's map form, so the two consumers can be
separated: `Labels.f` is the only branch tags item-to-item sees
(`column = "item.Labels.f"`), the rest of the map still reaches the CTR model, and the
carriers moved to `Comment` — the one field that `logics/`, `master/`, `model/` and
`dataset/` provably never read.

The first attempt (`type` + `cat` + `style` + `color`) removed the compression and
produced **exact ties** instead: the median item landed in an equivalence class of 66
byte-identical feature sets, so a top-10 was one class at distance 0, ordered arbitrarily
by the ANN index. Six attribute dimensions read out of the product title fixed that.

Replaying Gorse's own distance offline over all 95,335 eval products:

| | flat schema | map schema |
|---|---|---|
| items with an empty feature set | 21.0% | **0.0%** |
| distinct feature strings on the similarity path | 16 | **182** |
| equivalence classes | 1,564 | **39,915** |
| median item's class size | 1,585 items | **4 items** |
| distinct scores in a top-10 (median) | **1** of 10 | **7** of 10 |

That table is an offline replay of Gorse's scoring code, not Gorse's output — the distance
function is 20 lines and the IDF formula is one, so the whole arm can be simulated on the
real catalogue before touching the cluster. It found the tie problem in seconds instead of
after a two-hour run, and that is the habit worth keeping.

Confirmed afterwards against the live index, same 50-anchor protocol as the original
measurement:

| | before | after |
|---|---|---|
| anchors with ≥1 neighbour | 33 / 50 | **46 / 50** |
| global score range | 0.500689–0.510655 | 0.502549–0.523777 |
| median within-list spread | 0.000339 | **0.002044** |
| distinct scores in a top-10 (median) | **1** of 10 | **5** of 10 |

Three notes on honesty:

- **The offline replay was optimistic.** It predicted 7 distinct scores; live gives 5. The
  simulation brute-forced a 20,000-item pool while the real HNSW index holds 95,335, so
  equivalence classes are ~4.8× larger there and ties are correspondingly more common.
- **The absolute band is still ~[0.50, 0.52], and that is not a leftover defect.**
  `Score = 1/(1+d)` with the `(commonCount+100)` shrinkage puts even a strong match near
  d ≈ 0.96. Ranking ability, not absolute range, was the thing to fix.
- **Earlier drafts of this section quoted `style` 51% / `color` 29% coverage.** Those came
  from an unordered `LIMIT 3000` against Gorse's items table, which returns the
  first-inserted batch rather than a random sample. Measured over the full catalogue the
  figures are **48.7%** and **62.1%**.

The end-to-end effect is in §4.2, and it did not land where this section was aiming: the
arm's own NDCG stayed at noise level (14 → 17 users with a hit), while `/api/recommend`
gained 38 → 55 on the warm cohort through the fallback chain.

What was measured and rejected, since "why not also add X" is the obvious question:

| candidate | measured | why not |
|---|---|---|
| `brand`, floored at ≥10 items | 55.3% coverage, 1,733 values; median class 3→2 | IDF ≈ 7.5 reproduces the carrier dynamic, for one extra distinct score |
| price deciles | **11.9%** of the catalogue has a price at all | not viable |
| `avg_rating` | 100% coverage | **coverage is not relevance** |

Sequencing was deliberate: the baseline was locked *first*, so this is a before/after
rather than a blind change. Changing labels inside a locked baseline round would make the
two numbers incomparable — the same trap as changing rag-service's corpus under its locked
relevance judgments. One metric does not survive the change and is called out rather than
quietly re-reported: ILD is now computed over a different feature space, so
`ild_item_coverage` (88.4% → 100%) is the comparable fact, not ILD itself.

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

### 5.5 The cache is not the database, until you need it to be

Redis came back empty after a Docker restart, several times. The obvious diagnosis is
wrong, and the wrong one is expensive: it sends you to raise memory limits.

Snapshotting was never broken. Measured on a healthy container:

```
Background saving started ... Fork CoW for RDB: peak 37-69 MB ... terminated with success
total_forks:10   latest_fork_usec:7046   rdb_last_bgsave_status:ok
/data/dump.rdb   203 MB   dbsize 1,019,868
```

Saves fire every ~60s, take about a second, and copy-on-write peaks in the tens of MB —
nowhere near a memory ceiling. What actually happened is that `CONFIG SET save ""` had been
issued at runtime, to suppress the RediSearch fork GC. **A runtime `CONFIG SET` is not
written back to the config file**, so the running container silently persisted nothing for
13 hours while a freshly started one reports a perfectly healthy `save 3600 1 300 100 60
10000`. That asymmetry is the whole reason it survived several restarts undiagnosed: every
post-mortem inspection happens on a *new* container, which looks fine.

Nothing unrecoverable is lost — Redis is a cache and Postgres holds items, feedback and
users — but the rebuild (Load Dataset → train CF → item-to-item → per-user recommendations)
costs hours, and it silently invalidated a control in the evaluation above (§4.2).

Three needs, three different tools, which are easy to conflate:

| need | tool |
|---|---|
| planned restart | `redis-cli SAVE` — blocking, forks nothing, cannot hit the CoW path |
| unplanned death | periodic RDB gives ≤60s exposure; `appendonly yes` closes it further |
| a restorable known-good state | `SAVE` then `docker cp fashion-redis:/data/dump.rdb ./snapshot.rdb` |

The third is the one worth adopting before locking an evaluation baseline: it turns
"reproduce the locked state" from a multi-hour recompute into a file copy.

A diagnostic footnote, because it cost more than the bug did: `docker logs --since
2026-09-01T11:13:49` interprets a timestamp **without a timezone as local time**. On a
UTC+1 host that silently widened the window by an hour and swept in the pre-restart period,
which made a clean migration look like 89,865 broken items. The corrected filter
(`...11:13:49Z`) showed 530 — exactly the legacy items, all of them outside the eval
catalogue. Same failure shape as `grep -o 'loading:[01]'` matching `async_loading:0`
earlier in the project: **a filter looser than the thing it is meant to catch does not fail
loudly, it returns a confident wrong answer.**

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

### Order matters when resetting Redis

`FLUSHALL` deletes the RediSearch index along with the data, and Gorse only
issues `FT.CREATE` during `Init()`. Flushing while the master is running leaves
every index-backed operation failing with `No such index documents` — including
`POST /api/items`, so a subsequent push returns 500 with no obvious connection
to the flush.

Flush **before** the master starts, or restart it afterwards:

```bash
docker compose up -d postgres redis          # master NOT yet
until docker exec fashion-redis redis-cli INFO persistence | grep -qE '^loading:0'; do sleep 5; done
docker exec fashion-redis redis-cli FLUSHALL
docker exec fashion-redis redis-cli CONFIG SET save ""
docker compose up -d gorse-master gorse-server gorse-worker
```

Note the wait condition: `redis-cli PING` answers `PONG` while an RDB is still
loading and every real command is being rejected, so a health check based on
`PING` reports ready too early. `INFO persistence` anchored to `^loading:` is
the honest probe — `grep -o 'loading:[01]'` also matches `async_loading:0` and
returns two lines.

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
