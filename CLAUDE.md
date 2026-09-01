# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered fashion recommendation system built on top of the **Gorse** open-source recommendation engine. It has three main subsystems:

1. **Gorse core** (root of repo) — distributed recommendation engine (Master/Server/Worker nodes) written in Go
2. **fashion-recommend/** — a fashion-domain API layer built with Gin that integrates Gorse with LLM capabilities
3. **python-agent/** — a LangGraph ReAct agent (FastAPI) that replaces the Go agent with multi-turn memory, HITL trait approval, and a two-model architecture (Gemini router + Gemma finalizer)
4. **rag-service/** — a FastAPI microservice for natural-language product search; adds semantic/cold-start query handling on top of Gorse CF; pipeline: guardrail → hybrid retrieval (BM25 + Milvus) → CRAG loop → cross-encoder reranker → GPT-4o-mini generator

## Common Commands

### fashion-recommend (primary development area)

```bash
cd fashion-recommend

# Run API server
make run
# or directly:
go run main.go

# Method 3: flag-based startup (explicit port/db/gorse overrides)
go run main.go \
  -port 5001 \
  -db "host=localhost port=5432 user=gorse password=gorse_pass dbname=gorse sslmode=disable" \
  -gorse "http://localhost:8088"

# Build binaries
make build              # outputs to bin/fashion-api and bin/init-data

# Run tests
make test
# or a single package:
go test -v ./api/...

# Initialize sample data
make init-data
# or: go run data/init_data.go

# Seed Gorse from rag_products (items + demo feedback)
python3 data/seed_gorse.py

# Audit the item-label schema without writing anything (no Gorse needed).
# Prints per-prefix cardinality, how many labels Gorse's frequency filter will
# drop, and style/colour tagging coverage.
python3 data/seed_gorse.py --dry-run

# Seed a larger catalogue (default 1,000)
SEED_N_PRODUCTS=5000 python3 data/seed_gorse.py

# Label-schema regression tests (pure units, no DB/network)
python3 -m pytest data/test_seed_labels.py -v

# Go proxy (China mainland users — set before downloading dependencies)
export GOPROXY=https://goproxy.cn,direct
go mod download
go mod verify

# Start all infrastructure (Postgres, Redis, Gorse nodes)
make docker-up

# Stop infrastructure
make docker-down
```

#### Interaction dataset (`data/build_interactions.py`)

Measures the real interaction graph from the 2.5M-review McAuley Amazon Fashion
dump. Writes nothing; caches extracted columns to `data/cache/*.parquet`
(gitignored, ~4s to rebuild) so repeated analysis is instant.

```bash
# Sparsity analysis: degree distribution, iterative k-core, cohorts, taxonomy
python3 data/build_interactions.py --stats

# Restrict the join to a catalogue size
python3 data/build_interactions.py --stats --catalogue-size 5000

# How large must the catalogue be for the warm cohort to be evaluable?
python3 data/build_interactions.py --sweep-catalogue 1000,5000,20000,50000,all

# Build the eval dataset into Postgres (re-entrant: replaces, never appends)
python3 data/build_interactions.py --build             # ~24s
python3 data/build_interactions.py --build --dry-run   # report only

# Load the TRAIN split into Gorse (~67s), then check it landed
python3 data/build_interactions.py --push-gorse
python3 data/build_interactions.py --verify-gorse

# Graph-maths regression tests (pure units)
python3 -m pytest data/test_interactions.py -v
```

##### The eval dataset (`reco_products` + `reco_interactions`)

`--build` writes **95,335 products / 558,940 interactions**. Two decisions are
baked in and both are deliberate.

**The eval catalogue is its own table — `rag_products` is not touched.** The
obvious route (raise `SUBSAMPLE_SIZE` in `rag-service/data/normalize.py` and
regrow `rag_products` to 95k) was rejected: that table is rag-service's
**retrieval corpus**, BM25 is built from it at startup (`rag-service/main.py:98`,
`pipeline/retrieval.py:170`). Growing it 19× would change retrieval for every
query, require re-embedding 95k products into Milvus, shift the CRAG score
distribution the 0.45/0.43 thresholds were calibrated on, and — worst — make the
1,481 locked relevance judgments incomplete, since products entering the corpus
unjudged count as non-relevant and would depress NDCG/Recall for reasons
unrelated to retrieval quality. That damage is not repairable by re-running the
eval; it needs re-adjudication.

**The default split is a single global temporal cutoff, not leave-last-out.**

| protocol | train events | items with training signal | evaluable warm test users |
|---|---|---|---|
| `--split temporal` (default, q=0.70) | 391,258 | 62,006 (**65%**) | 859 |
| `--split leave-last-out` | 30,798 | 17,644 (**18.5%**) | 3,420 |

Leave-last-out is the textbook choice, but on a corpus where 86% of users appear
once it sends 95% of events to test and leaves 81.5% of the catalogue with *no
training signal at all*. An item with no training interaction cannot be
recommended by CF, popularity, or item-to-item, so catalog coverage would be
structurally capped at 18.5% and Gini computed over a truncated universe — the
beyond-accuracy metrics would be measuring the protocol, not the recommender.

A global cutoff is also **more** leak-proof, not less: leave-last-out will train
on a 2022 event while testing a 2015 event of a different user. One cutoff makes
every training event older than every test event, and `--build` asserts it. The
plan's prohibition was on a global *random* split, which this is not.

q=0.70 (2020-11-23) is chosen because the evaluable warm cohort peaks there
(704 → 859 → 843 across q=0.40…0.75) while item coverage keeps rising.

> The warm cohort is ~859 users either way — small, and that is the corpus, not a
> bug. Cold-start is where the volume is: 156,935 cold test users.

##### Pushing to Gorse

`--push-gorse` loads **only the train split** (391,258 events) plus all 95,335
items. Sending test events would put held-out interactions inside the model being
evaluated; `--verify-gorse` samples users and asserts no test event reached Gorse.

**Three gotchas, all of which look like failures but are not:**

1. **`/api/dashboard/stats` lags.** The master recomputes it on a schedule, so
   right after a push it still reports the old numbers. Verify by sampling
   entities through the API (what `--verify-gorse` does), not by reading counters.
2. **There are two Postgres instances.** Gorse's data store is the *Docker*
   Postgres (`fashion-postgres`, not published on the host); `rag_products` and
   `reco_*` live in the *host* Postgres. Same credentials, different databases.
   Check Gorse's side with
   `docker exec fashion-postgres psql -U gorse -d gorse -c "select count(*) from feedback"`.
3. **`fit_period = "24h"`** (`fashion-recommend/config/config.toml`, set in *both*
   `[recommend.collaborative]` and `[recommend.ranker]`), so the CF model does not
   retrain on newly pushed data for up to a day. `docker restart fashion-gorse-master`
   forces a reload.

   The master's task ticker is `min(collaborative.fit_period, ranker.fit_period)`
   (`master/master.go:125`), so the two values only mean anything together.
   Omitting the `ranker` one is not "inherit the other" — it falls back to **60
   minutes**, which silently overrides the 24h next to it and regenerates every
   user's offline cache hourly. An eval harness reading those caches can then have
   them change mid-run, with no error. Set both to `8760h` to freeze the loop for
   an evaluation (`RunTasksLoop` still primes once at startup, so caches are built
   exactly once), and set both back afterwards.

**`dislike` is stored but not trained on.** `config.toml:29-30` sets
`positive_feedback_types = ["purchase", "favorite", "add_to_cart"]` and
`read_feedback_types = ["view"]` — `dislike` is in neither. After the push Gorse
reports 276,194 positive + 51,280 read = 327,474 against 391,258 events sent; the
~64k difference is exactly the dislikes. The explicit-negative signal is present
in `reco_interactions` but inert in the model, which is the knob the Day 5
ablation turns, not a bug to fix silently.

Key measured facts (see `--stats` for the full table):

- 86.18% of users appear exactly once. The **iterative 5-core is empty** — not
  small, empty. 3-core is 2,223 users out of 2,035,490.
- k-core is computed on **distinct `(user, item)` edges**, not raw events;
  26,564 duplicate edges exist and counting them shifts every degree threshold.
- Under leave-last-out a 2-event user has **one** training event, and 72.7% of
  the "warm" cohort is exactly that. Report warm metrics stratified by training
  history or the warm number is measuring near-cold users.
- At the current 5,000-product catalogue only **370 users** have ≥2 training
  events. Do not lock an eval baseline there.
- `--sweep-catalogue` sizes the catalogue at **N ≈ 50,000**, and shows the hard
  ceiling: normalize.py's quality filters leave only 95,335 candidate products
  in total, capping the evaluable warm cohort at ~3,420 users.

> The sweep reports curve **shape**, not a predicted catalogue membership. A
> faithful re-implementation of normalize.py's published filter chain reproduces
> only 58.6% of the actual 5,000-row catalogue, so the ASIN-level composition of
> a hypothetical larger catalogue is not claimed. Two selection rules are shown;
> `dump review count` is selection-on-the-outcome (it picks items *because* they
> have many interactions, then reports how many interactions they have) and is
> included only as the optimistic bound. Size from `meta rating_count`.

#### Two catalogue sizes — do not conflate

`rag_products` holds 5,000 normalised Amazon products, but `seed_gorse.py` posts
only the top `SEED_N_PRODUCTS` (default **1,000**) by `rating_count`. Gorse never
sees the rest, so **catalog-coverage denominators must use the seeded count, not
the table size**. Every seed run prints both numbers.

#### Item label schema

Item labels are a **map**, not a flat array, and the shape is the whole point.
Gorse's `Labels` field is `any` and the two consumers read different parts:

| written to | read by | how |
|---|---|---|
| `Labels.f` | tags item-to-item **only** | `column = "item.Labels.f"`; `flatten` (`logics/item_to_item.go:379`) takes `[]dataset.ID` |
| the whole map | the CTR model | `ctr.ConvertLabels` (`model/ctr/data.go:39`) flattens `{"brand":"Nike"}` to the feature `brand.Nike` |
| `Comment` | nothing | grep `.Comment` across `logics/ master/ model/ dataset/` returns zero hits |

```json
Labels  = {"f": ["type:t-shirt", "cat:tops", "style:casual", "color:black",
                 "audience:women", "sleeve:short"],
           "brand": "zara", "price_range": "mid", "avg_rating": "4.5"}
Comment = "{\"name\": \"...\", \"price\": 12.34, \"desc\": \"...\"}"
```

**Off the similarity path is not the same as deleted.** `brand` and
`price_range` are legitimate CTR features — an FM can learn a brand embedding —
they were only ever wrong as *similarity* signals.

##### Why: the flat schema fed similarity 99.94% noise

Of the **107,747** distinct label strings the flat schema put on the similarity
path, **107,686 were `item_name` / `brand` / `price`** — near-unique carriers.
Sixty-one strings were actual features. Carriers have IDF ≈ `log(95335)` ≈ 11.5
each against ≈ 1.9 for `cat:`, and the distance function
(`logics/item_to_item.go:337`) divides by `sqrt` of the weighted sum over *all*
of an item's tags, so two carriers dominated a norm the real features could not
move. Result: every live score in **0.500689–0.510655**, within-list spread
0.000339.

> `Score = 1/(1+distance)` (`item_to_item.go:127`), so 0.5007 means distance
> 0.997 — as close to "no shared signal at all" as the function goes. That
> conversion is what makes the live numbers legible; without it 0.50 looks like
> a middling similarity rather than a floor.

##### Measured before/after, over all 95,335 eval products

Replaying Gorse's own distance offline (same IDF formula as
`dataset.GetItemColumnValuesIDF`, `dataset/dataset.go:196`):

| | flat schema | map schema |
|---|---|---|
| items with an empty feature set | 21.0% | **0.0%** |
| distinct feature strings | 16 | **182** |
| equivalence classes | 1,564 | **39,915** |
| median item's class size | 1,585 items | **4 items** |
| distinct scores in a top-10 (median) | **1** of 10 | **7** of 10 |

##### Two findings that changed the design mid-way

**1. The raw `category` column was being thrown away.** It has 106 values, 100%
coverage and 2 singletons — by far the best feature in the catalogue — and
`_normalise_category` collapsed it to 12 buckets for Gorse's `Categories` field
and then discarded the detail. Both levels are now labels (`type:` and `cat:`),
and keeping both is what lets Jaccard express "somewhat alike" at all: a shared
label is binary, so a flat vocabulary has two rungs and a two-level taxonomy has
three — same type > same category > unrelated.

**2. `type`+`cat`+`style`+`color` alone swapped one failure for another.** It
removed the compression and produced **exact ties**: the median item landed in an
equivalence class of 66 byte-identical feature sets, 80.5% of the catalogue sat
in a class of ≥10, so a top-10 was one class at distance 0 ordered arbitrarily by
the ANN index. Six attribute dimensions read out of the product title
(`audience` 82%, `sleeve` 29%, `pattern` 21%, `fit` 18%, `neck` 12%,
`length` 10% — see `ATTRIBUTE_KEYWORDS`) cut the median class to 4 and took a
top-10 from 1 distinct score to 7.

> This was found by replaying the distance function offline, not by a two-hour
> Gorse run. Worth keeping as a habit: the scoring code is 20 lines and the IDF
> formula is one line, so the whole arm can be simulated on the real catalogue
> before touching the cluster.

##### Rejected, with the measurement

| candidate | measured | why not |
|---|---|---|
| `brand`, floored at ≥10 items | 55.3% coverage, 1,733 values; median class 3→2, +1 distinct score | IDF ≈ 7.5 reproduces the carrier dynamic for almost nothing |
| price deciles | **11.9%** of the eval catalogue has a price at all | not viable |
| `avg_rating` | 100% coverage | **coverage is not relevance** — two items rated 4.5 are not similar |

##### Things that will silently break this

- **`"f"` is a literal string in three places** — `config.toml`'s `column`,
  `models.ItemLabels`'s json tag, and `seed_gorse.build_item_labels`. Renaming
  one makes item-to-item fail on every item while the dashboard reports
  `Complete`. `models_wire_test.go` asserts the key.
- **`avg_rating` must stay a string.** A JSON number under a map key reaches
  *neither* model: `ctr.convertLabels` has no `float64` branch (gorm's json
  serializer decodes into `float64`, not `json.Number`) and `flatten` ignores
  numeric leaves. It would look like a feature and be inert.
- **Flat labels fail at *runtime*, not compile time.** `item.Labels.f` against a
  `[]string` compiles fine and then errors per item — the Day 2 failure mode
  exactly. `config/fashion_config_test.go`'s `TestFashionColumnRejectsFlatLabels`
  pins this, which is what makes an incomplete re-seed loud instead of silent.
- **`_feature_labels` on a dict must not iterate keys.** `frozenset(l for l in
  some_dict)` yields `"f"`, `"brand"`, … — none match a prefix, so ILD returns a
  clean **0.0 at full-looking coverage**. Pinned in `eval/test_metrics.py`.

##### The two seeders had already drifted

`seed_gorse.py` emitted `occasion:` and `material:`; `build_interactions.py`
never did. Since the 95,335-product eval catalogue is built by the latter, the
harness's `FEATURE_LABEL_PREFIXES` named two prefixes that appeared on **zero**
of the items it scored. `build_interactions.py` now imports the schema rather
than restating it.

Two content-tagging assumptions remain heuristic rather than ground truth:

- **`style:`** — `rag_products` has no style column, so labels come from keyword
  matching over name + description. The six classes mirror `traits/extractor.go`
  so the user and item sides share a vocabulary; that file's keywords are Chinese
  and the catalogue is English, so it cannot be reused directly.
- **`color:`** — the `colour` column is empty for 941/1,000 items and the rest is
  uncontrolled free text (`cut vines`, `3x`, `smokey quartz`). Labels are mapped
  onto the same ten-colour palette the user side uses.

`price_range:unknown` exists because `rag-service/data/normalize.py:340` assigns
`"mid"` to every price-less product. 68% of the catalogue has no price, so 672 of
803 `mid` items at N=1,000 actually meant "no data". normalize.py belongs to
rag-service, whose eval baseline is locked, so this is corrected at seed time.

##### Re-seeding after a label change

Items only — the train events are unchanged and re-posting costs one Redis
round-trip each (`server/rest.go:1240`):

```bash
python3 data/build_interactions.py --push-gorse --skip-feedback
docker restart fashion-gorse-master     # forces a dataset reload
```

`--skip-feedback` refuses to run with `--reset-gorse`: that pair deletes the
feedback it then declines to restore, and the failure is silent — the push
reports success, every recommender falls back, and the eval reads it as a model
quality collapse rather than as missing data.

⚠️ **ILD is not comparable across this change.** `FEATURE_LABEL_PREFIXES` gained
`type:`/`cat:`, so the metric is computed over a different feature space. The
comparable fact is `ild_item_coverage` going 79% → 100%. Same rule as always:
never change labels inside a locked baseline round without saying which numbers
stopped being comparable.


#### User label schema — three bugs, not one

`NumUserLabels = 0` against `NumItemLabels = 15,332` was **three independent
defects stacked**, each of which alone was enough to zero the count. Any fix that
addressed only one would have looked like it failed.

**1. The score was baked into the label string** (`traits/gorse_sync.go`).
`fmt.Sprintf("style:%s:%.1f", …)` turned one preference into as many distinct
strings as there were distinct scores. Gorse indexes a label only on its **second
occurrence across users** (`master/tasks.go:295-330`), so every scored label was a
singleton and every singleton was dropped. Fixed: the score now gates only, and
the emitted string is a flat `prefix:value` — same shape the item side always had.

**2. The gate was above the reachable score range.** `score > 0.5`, but
`extractor.go:312` max-normalises the keyword pass (top-1 = 1.0) and
`extractor.go:197` merges as `keyword*0.4 + ai*0.6`. When the LLM leg is absent or
fails, the merged ceiling is **0.4** — so the gate silently discarded every
keyword-only extraction. Measured on live data: of 10 rows in `user_traits`, three
had any style/colour at all and only one cleared 0.5. Default is now **0.35**,
below that 0.4 ceiling and above the second tier.

**3. `models.User` serialised as `user_id`, Gorse reads `UserId`.**
`encoding/json` matches field names case-insensitively but not across
underscores, so `labels` happened to reach `Labels` while `user_id` never reached
`UserId`. Every user the Go service ever inserted landed in **one row with an
empty id**. Verified by hand:

```
POST /api/users [{"user_id":"x","labels":["style:minimalist"]}]
  → {"RowAffected":1}   and   GET /api/users → {"UserId":"", "Labels":[...]}
```

`models.Feedback` had the identical defect (`feedback_type` / `user_id` /
`item_id`), so feedback posted through the Go client was equally keyless. Both
structs now use Gorse's wire names, as `models.Item` always did.

> This one survived `api/server_test.go` because the mock Gorse encodes its
> response with the *same* struct the client decodes with — a tag error cancels
> itself out in a round trip. `models/models_wire_test.go` asserts literal JSON
> keys instead, which is the only shape of test that can catch it.

**Namespace alignment (user side → item side):** `price:low` → `price_range:budget`,
`price_preference:` → `price_range:`, `favorite_brand:` → `brand:`. The values are
mapped too, not just the prefixes — the item side's vocabulary is
`budget/mid/premium` while traits produce `low/medium/high` and
`data/init_data.go` had invented a third set (`mid-range/high-end/luxury`).
Renaming a prefix without mapping its values leaves the two sides just as
disjoint as before.

> Note the CTR model indexes user and item labels in **separate** feature spaces
> (`userLabelIndex` vs `itemLabelIndex`, `master/tasks.go`), so the FM does not
> *require* string equality to learn a cross. Alignment matters for the Day 5
> aggregation arm — where user labels are literally derived from item labels — and
> for tags user-to-user, not for making the FM work at all.

| env | default | meaning |
|---|---|---|
| `TRAIT_LABEL_MIN_SCORE` | `0.35` | style/colour score gate; the ablation sweeps this |
| `TRAIT_LABEL_MAX_PER_PREFIX` | `5` | cap per prefix, so label *count* is not a confound between ablation arms |

Labels are also sorted (score desc, then name) before being written. Go's map
iteration is randomised, so the previous implementation wrote a differently
ordered array on every sync of unchanged traits.

##### Verifying it (A/B in one master reload)

`traits/gorse_sync_live_test.go` pushes four fixture users in a single batch:
`fx_old_a`/`fx_old_b` carry the pre-fix label shape, `fx_new_a`/`fx_new_b` carry
what the fixed function actually emits. Both pairs share the same preferences and
differ only in score, so the old pair's strings are all singletons and the new
pair's are not. One reload, one `NumUserLabels` reading, and the delta is
attributable to label shape alone — which sequential before/after runs cannot
claim, since the master's task cycle and dataset snapshot move in between.

```bash
GORSE_LIVE_TEST=1 go test ./traits/ -run TestLivePushFixtures -v
docker restart fashion-gorse-master        # forces a dataset reload
curl -s localhost:8088/api/dashboard/stats | grep NumUserLabels
GORSE_LIVE_TEST=1 GORSE_CLEANUP=1 go test ./traits/ -run TestLiveCleanupFixtures
```

The control arm deliberately omits `price:`. That label carries no score, so both
old users share it and it *would* be indexed — including it would let the control
arm contribute to the delta and destroy the attribution. The prefix rename is
covered by `TestPriceRangeNamespace` instead.

> **Why not verify with the real 24,850 warm users** (aggregating train-split item
> labels into user labels)? Because that population is the middle arm of the Day 5
> ablation. Pushing those labels into the live Gorse now means the "no profile"
> arm has to delete them again, and the simulated cold-start design also needs
> those users' *feedback* withheld — which Day 1 already loaded. That is a
> deliberate Gorse-state decision for Day 5, not a side effect of a Day 2 fix.

#### The config is full of expr expressions, and three of them never ran

`fashion-recommend/config/config.toml` is mostly expr expressions, not literals.
Three separate ones were broken, and **only one of them failed at compile time** —
the other two compiled cleanly and failed on every evaluation, where the symptom
is one error line per item while the dashboard still reports the task `Complete`.

| expression | failure | recommender affected |
|---|---|---|
| `column = "Labels"` | compile — unresolvable identifier | tags item-to-item, tags user-to-user |
| `duration('7d')` / `duration('30d')` | **runtime** — `time.ParseDuration` has no `d` unit (only `ns/us/ms/s/m/h`) | `trending`, `new_arrivals` |
| `float(now() - item.Timestamp)` | **runtime** — `now() - item.Timestamp` is a `time.Duration` and expr's `float()` rejects it | `new_arrivals` (so it was broken on score *and* filter) |

`config/fashion_config_test.go` (in the gorse module, not fashion-recommend)
compiles **and evaluates** every expression in that file. Evaluation is the whole
point: compile-only checking catches one of these three. The test lives on the
gorse side because the semantics — which variables each env binds, what
`duration()` accepts — are defined by `logics/`, and asserting them from a module
that does not depend on gorse would just be restating them from memory. The third
bug above was found by this test, not by the logs.

```bash
go test ./config/ -run TestFashion -v   # from the repo root, not fashion-recommend
```

##### `column` is an expr expression, not a column name

Both tags-based similarity arms in `fashion-recommend/config/config.toml` were set
to `column = "Labels"`, which Gorse compiles as an expr expression whose only bound
variable is `item` (`logics/item_to_item.go:199`) or `user`
(`logics/user_to_user.go:112,167`). A bare `Labels` is an unresolvable identifier,
so both tasks failed on every entity:

```
failed to update item-to-item recommendation
error: unknown name Labels (1:1)
```

The Day 2 fix was `item.Labels` for `[[recommend.item-to-item]] style_similarity`
and **`user.Labels`** for `[[recommend.user-to-user]] style_match` — the two envs
bind different variables, so copying the item-to-item value across does not
compile either.

> The item side has since moved on to `item.Labels.f` (see the item label schema
> section). The user side is still `user.Labels`, because user labels are still a
> flat array.

Flat `[]string` labels do work here: `dataset.processLabels`
(`dataset/dataset.go:366`) converts a string array to `[]ID`, which is one of the
three types `logics`' `flatten` accepts. No re-seeding was needed for the Day 2
fix itself — the later `.f` change is what required one.

**The failure mode changes with the depth of the expression, and this is the
part worth remembering.** A bare `Labels` fails at *compile* time, so it is
caught by any check that merely compiles. `item.Labels.f` against flat labels
compiles cleanly — `data.Item.Labels` is `any`, so expr defers the member access
to runtime — and then errors on every single item, which is the quiet variant:
one log line per item, task reported `Complete`.
`config/fashion_config_test.go` therefore *evaluates* rather than compiles, and
`TestFashionColumnRejectsFlatLabels` asserts that specific runtime failure so an
incomplete re-seed is loud.

> **The dashboard reported these failing tasks as `Complete 383,460/383,460` in
> `0.0s`.** Task status is not evidence here; the error existed only in
> `docker logs fashion-gorse-master`.

**These bugs filled the Docker VM's disk.** The master logs one error line per
failed item, so each task cycle wrote several hundred thousand lines, hourly, for
days. Symptoms, in the order they appear and none of which name the cause:

- `docker logs fashion-gorse-master` hangs indefinitely (even with `--tail`)
- the master exits and `restart: unless-stopped` does not bring it back
  (exit code 2 is Gorse's generic fatal exit — it is *not* diagnostic of disk;
  the same code appears when Redis is still loading its RDB at startup)
- `docker compose up` finally says it plainly:
  `mkdir /var/lib/docker/overlay2/…: no space left on device`
- `docker system df` accounts for only ~7.5 GB of a 60 GB `Docker.raw`, because
  container log files are not counted in its `SIZE` column
- once the filesystem is full, `docker rmi` itself wedges — clean up *before*
  the disk fills, not after

The host Mac having hundreds of GB free is irrelevant; the limit is Docker
Desktop's disk image size. `docker-compose.yml` now caps `json-file` logging on
every service (50m × 3), but **that only applies to containers created after the
change** — `docker compose up -d` must actually recreate them.

##### The other resource wall: Redis gets OOM-killed and the master keeps "working"

The Docker VM has ~7.75 GiB. `Load Dataset` on this corpus spikes the master well
above its ~2.5 GiB steady state, and the whole rag-service stack (milvus, kafka,
minio, etcd) is running alongside. Redis, holding a ~500 MB RDB, is what gets
killed:

```
fashion-redis :: Exited (137)          # 137 = 128 + SIGKILL
```

**The failure mode is worse than a crash, because nothing crashes.** Docker's
embedded DNS drops records for stopped containers, so the master starts logging

```
failed to save user neighbors to cache
error: dial tcp: lookup redis on 127.0.0.11:53: no such host
```

for every single user — while sitting at ~98% CPU with its task still marked
`Running`. Since task progress is itself stored in Redis, the counter stays
pinned at `0/1484868`, which reads as "slow" rather than "every write is
failing". It will sit there indefinitely.

**Check `docker ps -a` for exit code 137 before concluding a task is merely
slow.** Recovery is `docker start fashion-redis` then `docker restart
fashion-gorse-master` — the master does not recover on its own, because it
resolves the hostname per operation and never re-establishes the connection
pool. Before a long run, free memory by stopping the rag-service stack
(`docker compose stop milvus kafka minio etcd`); their state lives in volumes,
so nothing is lost.

> This is the third variant of the same lesson in this file: **Gorse looks busy
> and reports progress while doing nothing useful.** Task status said `Complete`
> for the failing expr tasks, and says `Running` here. Neither is evidence.

### Frontend (fashion-recommend/frontend/)

```bash
cd fashion-recommend/frontend

npm install
# Note: vite.config.ts shoproxy was manually updated to include /images (not in original template):
#   '/api'    → http://localhost:5001
#   '/images' → http://localhost:5001
npm run dev       # dev server
npm run build     # build to dist/ (served by Go backend)
npm run preview
```

### Admin Dashboard (admin-dashboard/)

```bash
cd admin-dashboard

# Install dependencies
npm install

# Build frontend
npm run build     # outputs to dist/

# Start server (runs on http://localhost:3001)
node server.js

# Or use PM2 for persistent process management (recommended)
npm install -g pm2
pm2 start server.js --name admin-dashboard
pm2 status
pm2 logs admin-dashboard
pm2 stop admin-dashboard
pm2 restart admin-dashboard
```

### python-agent (python-agent/)

A LangGraph-based ReAct agent — **the only ReAct implementation in this repo**. The Go
side has no agent: `fashion-recommend/ai/service.go` is single-shot LLM calls only, and
the Go API simply proxies `/api/ai/agent-chat` and `/api/ai/agent-resume` through to this
service (`proxyToPythonAgent` in `api/server.go`). Exposes a FastAPI server with
PostgreSQL-backed multi-turn memory via LangGraph checkpointing and HITL (Human-in-the-Loop)
trait approval.

```bash
cd python-agent

# ---- First-time setup ----
pip install -r requirements.txt

# Copy and fill in secrets
cp .env.example .env   # set GOOGLE_API_KEY, TAVILY_API_KEY, DATABASE_URL, GORSE_URL

# ---- Run the API server ----
uvicorn main:app --reload --port 8001

# ---- Connectivity smoke-test (checks all 4 external services) ----
python3 test_connections.py

# ---- Unit + integration tests (no external services needed) ----
pip install pytest pytest-asyncio   # one-time
pytest tests/ -v

# Run a single test file
pytest tests/test_hitl_flow.py -v

# Run a single test by name
pytest tests/test_merge_traits.py::test_price_sensitivity_override -v
```

#### Environment Variables (python-agent)

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google AI Studio key; used by both Gemini router and Gemma finalizer |
| `TAVILY_API_KEY` | — | Tavily search API key for `search_fashion_trends` tool |
| `DATABASE_URL` | `postgresql://gorse:gorse_pass@localhost:5432/gorse` | asyncpg DSN for LangGraph checkpointer + trait storage |
| `GORSE_URL` | `http://localhost:8088` | Gorse master HTTP endpoint |
| `AGENT_ROUTER_MODEL` | `gemini-2.5-flash` | Function-calling model for ReAct tool decisions |
| `AGENT_FINAL_MODEL` | `gemma-4-31b-it` | Text-generation model for the polished final answer |
| `AGENT_MAX_ITERATIONS` | `8` | Hard cap on ReAct loop iterations |
| `AGENT_TOKEN_BUDGET` | `20000` | Cumulative token cap per turn (exits loop early if exceeded) |
| `AGENT_METRICS_PATH` | `metrics/turns.jsonl` | Per-turn latency/token/cost JSONL. Set to `""` to disable |

> Note: Google AI Studio retires model ids. As of 2026-08-23 `gemma-3-27b-it` and
> `gemma-3-12b-it` return **404 NOT_FOUND**; the working pair is `gemini-2.5-flash`
> (router) and `gemma-4-31b-it` (finalizer), which are now both the `.env` values and
> the code defaults. `test_connections.py` probes both tiers — checking only the router
> is how the dead finalizer default stayed hidden. See `python-agent/MEMORY.md`.

#### Instrumentation (latency / tokens / cost)

`agent/metrics.py` records one `NodeMetric` per graph-node execution — wall-clock
latency, plus input/output tokens and USD cost for the two LLM nodes. `AgentGraph.chat()`
assembles them into a `TurnMetrics` record and appends one JSON line per turn to
`AGENT_METRICS_PATH`. `/api/ai/agent-chat` returns `latency_ms` and `cost_usd` on every
response, and the full `node_metrics` array when `include_trace: true`.

```bash
cd python-agent

# Aggregate a run: latency percentiles, per-node breakdown, token split, cost
python eval/aggregate_metrics.py

# A specific file (e.g. one arm of an A/B), or machine-readable output
python eval/aggregate_metrics.py --input metrics/arm_guard_on.jsonl
python eval/aggregate_metrics.py --json

# Chat turns only — /agent-resume turns are pure I/O and skew the distribution
python eval/aggregate_metrics.py --turn-type chat
```

Two rules the code enforces, because breaking either produces plausible-looking wrong numbers:

- **Missing usage ≠ zero usage.** `usage_metadata` is absent on mocked models and
  some provider paths. `NodeMetric.usage_available` records the difference, and the
  aggregator prints a coverage warning rather than silently under-reporting.
- **Unpriced model ≠ free model.** A model absent from `agent/pricing.json` yields
  `cost_usd = None`; one unpriced call collapses the whole turn's cost to `None`.

`agent/pricing.json` is an **operator-supplied assumption**, not a measurement —
`verified_by_operator` is `false` until someone checks it against the provider's
current pricing page. The finalizer (Gemma, free tier) is priced at `0.0`, which makes
any dollar "saving" from model tiering partly tautological; the aggregator therefore
prints the **token share** alongside it, and that share is the number to quote.

`counterfactual_model` in the price table reprices every token — router *and*
finalizer — at the router model, answering "what would this cost as a single-model agent?"

#### HITL (Human-in-the-Loop) flow

When the agent detects an explicit preference statement (e.g. "I like minimalist style"), it:

1. Calls `update_user_traits` tool → stages updates in `pending_trait_updates` state (no DB write yet)
2. Completes the answer normally via `finalizer` node
3. Graph **pauses** at `interrupt_before=["write_traits"]` — LangGraph serialises state to Postgres
4. `/api/ai/agent-chat` response includes `pending_approval: true` and `pending_trait_updates: [...]`
5. Frontend shows an approval card; user clicks **Confirm** or **Cancel**
6. Frontend calls `POST /api/ai/agent-resume` with `{"approved": true/false}`
7. `approved=true` → graph resumes, `write_traits_node` merges & writes to DB + syncs Gorse
8. `approved=false` → updates discarded, graph advances to END

#### Test suite layout

```
python-agent/
├── pytest.ini                              # asyncio_mode = auto
├── eval/
│   └── aggregate_metrics.py                # percentiles + token/cost roll-up over turns.jsonl
└── tests/
    ├── conftest.py                         # shared fixtures (mock_db, agent_graph)
    ├── test_merge_traits.py                # pure unit — _merge_trait_updates()
    ├── test_update_user_traits_tool.py     # tool unit — validation, staging logic
    ├── test_graph_routing.py               # routing conditions — should_write_traits
    ├── test_hitl_flow.py                   # integration — full chat→approve/reject cycle
    └── test_metrics.py                     # instrumentation units + graph-wiring integration
```

All 59 tests run in ~0.4 s with no external service dependencies (MemorySaver replaces Postgres; LLM calls are AsyncMocks).

### rag-service (rag-service/)

FastAPI microservice for semantic product search. Runs on port 8002. Requires Postgres, Milvus, Redis, and OpenAI API key.

```bash
cd rag-service

# First-time setup
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY at minimum

# Run the API server
uvicorn main:app --reload --port 8002

# Run tests (no external services needed — all mocked)
pytest tests/ -v

# Eval harness: run against golden query set
python eval/run_eval.py --golden-set eval/golden_queries.json

# Check regression against locked baseline (fails if any metric drops >5%)
python eval/check_regression.py --threshold 0.05

# Load test (headless, 3 min per level)
locust -f eval/locustfile.py --host http://localhost:8002 --users 10 --spawn-rate 10 --run-time 3m --headless --csv eval/load_test_u10
```

#### Eval Baseline (locked, do not overwrite without re-running adjudication)

| Metric | Value | Target | |
|---|---|---|---|
| NDCG@10 | 0.8468 | ≥ 0.50 | pass |
| Recall@10 | 0.6993 | ≥ 0.70 | miss by 0.0007 |
| Faithfulness | 0.9580 | ≥ 0.85 | pass |

Locked 2026-08-11 in `rag-service/eval/baseline_metrics.json`. Relevance judgments (0/1/2) live inline in `rag-service/eval/golden_queries.json` under each query's `relevance` key — **1,481 (query, ASIN) judgments, 14.8 per query, across 3 adjudication rounds**.

**Recall@10 is structurally capped at 0.7615.** The golden set now averages 13.4 relevant products per query and 81 of 100 queries have more than 10 relevant products, so ten slots cannot hold them all: `max = mean(min(10,|rel|)/|rel|) = 0.7615`. The measured 0.6993 is 91.8% of that ceiling. The ≥0.70 target was set when labels averaged 5.8 per query and the ceiling was near 1.0; denser labelling lowered the ceiling without changing retrieval. Target deliberately NOT lowered to match the result. Full analysis in `rag-service/README.md` § Evaluation.

**Faithfulness is not Ragas.** `ragas` is commented out in `requirements.txt` on purpose; `eval/faithfulness_judge.py` (renamed from `ragas_judge.py`) runs a two-step GPT-4o-mini claim-extraction-and-verification judge. Installing ragas silently switches judges and invalidates comparison against this baseline. Known bias: returns 1.0 when claim extraction yields nothing, so short answers skew high.

**Labelling caveat:** round 3 judged 89% of pooled candidates relevant, above the 10–30% typical of TREC pools. Strict re-scoring (grade 2 only) gives NDCG 0.8033 / Recall 0.7785 — the conclusion holds.

#### CRAG thresholds (env-configurable)

| Variable | Value | Meaning |
|---|---|---|
| `CRAG_HIGH_THRESHOLD` | `0.45` | Above this → synthesize directly |
| `CRAG_LOW_THRESHOLD` | `0.43` | Below this → return best_effort candidates (not trending fallback) |
| `CRAG_MAX_RETRIES` | `2` | Max query-rewrite + retry attempts |
| `CRAG_TIME_BUDGET_S` | `3.5` | Hard wall-clock cap on CRAG loop |

**Threshold provenance:** selected empirically on 2026-08-06 via a 50-combo offline grid with
bootstrap validation — full write-up in `rag-service/README.md` § CRAG Threshold Calibration.
`_grade()` averages cosine over all 20 candidates including the weak tail, capping the achievable
score at **0.7091** (100-query golden set; p50 = 0.5125, p0 = 0.3205) — so a HIGH threshold of
0.75 is unreachable and would route every query to a rewrite. HIGH stayed at 0.45: no combo beat
it outside the bootstrap CI. LOW was raised `0.10 → 0.43`, trading −0.0044 NDCG (inside the noise
band) for −258 ms expected latency and a retry rate of 30% → 5%. This leaves a deliberately narrow
retry band, because the query rewrite was measured to *degrade* the retrieval grade on 75 of 100
queries. Path distribution: 70 synthesize / 2 retry / 28 best_effort.

**Important:** `score < CRAG_LOW_THRESHOLD` returns `best_effort` (real candidates, weak signal) — NOT trending products. Trending fallback is only used when retrieval returns zero candidates. This was a deliberate architectural fix in Session 4.

### Gorse Core (root module)

```bash
# Build all Gorse binaries
go build ./cmd/...

# Run tests
go test ./...

# Run a specific package test
go test -v ./logics/...
```

## Environment Variables (fashion-recommend)

| Variable | Default | Description |
|---|---|---|
| `GORSE_ENDPOINT` | `http://localhost:8088` | Gorse master HTTP endpoint |
| `GORSE_API_KEY` | `` | Gorse API key |
| `PORT` | `5000` | API server port. **The default is 5000, not 5001** — but `frontend/vite.config.ts` proxies `/api` to **5001**, and on macOS 5000 is usually taken by AirPlay Receiver. Start the API as `PORT=5001 go run main.go` or the frontend cannot reach it. |
| `DATABASE_URL` | `host=localhost port=5432 user=gorse password=gorse_pass dbname=gorse sslmode=disable` | PostgreSQL connection string |
| `AI_API_KEY` | (Aliyun DashScope key) | LLM API key |
| `AI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible LLM endpoint |
| `AI_MODEL` | `qwen-plus` | LLM model name |

> The Go service has no agent of its own — no ReAct loop, no tools, no model tiering.
> `AGENT_*` belongs to `python-agent/` only (see its table above), and web search there
> is Tavily via `TAVILY_API_KEY` — there is no `WEB_SEARCH_URL` anywhere in this repo.

## Architecture

### Gorse Distributed Nodes

The Gorse core runs as three separate processes that communicate via gRPC (port 8086):

- **Master** (`master/`, port 8086 gRPC / 8088 HTTP) — orchestrates the system, trains CF and CTR models, serves the admin dashboard
- **Server** (`server/`, port 8087 HTTP) — serves REST recommendation APIs to clients
- **Worker** (`worker/`, port 8089 HTTP) — executes background recommendation jobs

The Master trains models and pushes them to Server/Worker nodes. All inter-node communication uses Protocol Buffers defined in `protocol/`.

### Recommendation Algorithm Pipeline (`logics/`)

Recommendations are built by chaining multiple algorithms with fallbacks:

1. **Collaborative Filtering** (`cf.go`) — user-item interaction matrix factorization
2. **Item-to-Item** (`item_to_item.go`) — content similarity
3. **User-to-User** (`user_to_user.go`) — social similarity
4. **Non-personalized** (`non_personalized.go`) — trending, popular, new arrivals
5. **LLM Re-ranking** (`chat.go`) — LLM-based reranking and explanation generation

### fashion-recommend Service (`fashion-recommend/`)

A Gin-based HTTP API that sits in front of Gorse. Key packages:

- `api/` — route handlers organized by domain (auth, AI, comments, likes, items, users, recommendations)
- `ai/` — OpenAI-compatible client for single-shot LLM calls (`service.go`, default: Aliyun DashScope qwen-plus). No agent lives here; agentic behaviour is `python-agent/`, reached via the proxy routes in `api/server.go`
- `client/` — HTTP client for Gorse master/server APIs
- `traits/` — LLM-powered user style preference extraction; syncs traits back to Gorse as user labels
- `database/` — PostgreSQL models for conversations, messages, user traits, and social interactions (comments, likes)
- `auth/` — session-based auth service
- `models/` — shared data models (User, Item, Feedback)

The frontend (`fashion-recommend/frontend/`) is a React/TypeScript/Vite/Tailwind SPA. Its production build goes to `frontend/dist/` and is served as static files by the Go backend via Gin.

### Storage Layer

Gorse supports pluggable backends configured in `config/config.toml`:

- **Data store**: MySQL, PostgreSQL, MongoDB, ClickHouse, or SQLite
- **Cache store**: Redis or MySQL
- **Vector store** (`storage/vectors/`): Milvus, Qdrant, Weaviate, or SQLite — used for ANN (approximate nearest-neighbor) search powering item-to-item and user-to-user similarity
- **fashion-recommend** uses PostgreSQL directly (via `database/` package) for social features not managed by Gorse

#### Vector Store (`storage/vectors/`)

All vector backends implement a common `Database` interface (`database.go`) with these operations: `AddCollection`, `DeleteCollection`, `AddVectors`, `DeleteVectors`, `QueryVectors`. The backend is selected by URI prefix (`milvus://`, `qdrant://`, `weaviate://`, `sqlite://`).

The **Milvus** backend (`milvus.go`) is the most full-featured: it uses the official `milvus-sdk-go/v2` and manages schema creation (id, vector, categories array, timestamp fields), HNSW index creation with configurable distance metrics (Cosine/L2/IP), and filtered ANN search using Milvus expression syntax (e.g., `array_contains(categories, 'X')`). The `proxy.go` file wraps any backend with caching/proxy logic.

### LLM Integration Points

- `logics/chat.go` — LLM re-ranking within Gorse core (uses Ollama/qwen2.5 by default per `config/config.toml`)
- `fashion-recommend/ai/service.go` — single-shot chat, recommendation explanation, style advice (uses Aliyun DashScope qwen-plus)
- `python-agent/agent/graph.py` — the stateful ReAct agent serving `POST /api/ai/agent-chat` (the Go route of that name is a proxy). Router model handles tool-call iterations, final model synthesizes the answer; returns an optional per-iteration trace and per-node metrics (`include_trace: true`). Five tools, all in `python-agent/agent/tools.py`:
  - `get_recommendations` — personalized item recommendations via Gorse
  - `get_user_preferences` — stored trait data (style, colour, price, brands, occasions) from PostgreSQL
  - `get_item_details` — single-item lookup via Gorse
  - `search_fashion_trends` — external trend lookup via Tavily (`TAVILY_API_KEY`)
  - `update_user_traits` — stages trait updates for HITL approval; returns a `Command`, which is why the tools node has to handle both return shapes
- `fashion-recommend/traits/extractor.go` — extracts structured style/color/occasion preferences from user text; maps Chinese keywords to English Gorse labels

### Data Flow

```
User actions (comments, likes) → fashion-recommend API
    → AI trait extraction → Gorse (user labels)
    → Gorse recommendation algorithms (CF, item-to-item, etc.)
    → Cached recommendations → fashion-recommend API → Frontend
```

## Database Setup (Manual — without Docker)

### Install PostgreSQL

macOS:
```bash
brew install postgresql@15
brew services start postgresql@15
# Stop: brew services stop postgresql@15
```

Ubuntu/Debian:
```bash
sudo apt update && sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Create user and database
```sql
-- run: psql postgres
CREATE USER gorse WITH PASSWORD 'gorse_pass';
CREATE DATABASE gorse OWNER gorse;
GRANT ALL PRIVILEGES ON DATABASE gorse TO gorse;
\q
```

### Verify connection
```bash
psql -h localhost -U gorse -d gorse -W
# password: gorse_pass
```

### Migrations are NOT applied automatically

`fashion-recommend/database/migrations/*.sql` exists but nothing runs it — not
the API at startup, not `make init-data`. `003_create_product_likes.sql` had
never been applied, so `product_likes` and `product_like_stats` did not exist,
and every like returned 500 from `GetProductLikeCount` before reaching the
Gorse feedback call. The tables the app needs are a superset of the ones listed
below.

```bash
for f in fashion-recommend/database/migrations/*.sql; do
  PGPASSWORD=gorse_pass psql -h localhost -U gorse -d gorse -f "$f"
done
```

Same family as the bugs in the label schema section: written, shipped, never
executed. Check `\dt` against what `database/models.go` actually queries before
concluding a feature is broken in code.

### Initialize tables (optional — auto-created on first run)
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE items (
  id SERIAL PRIMARY KEY,
  item_id VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  category VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE likes (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  item_id VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, item_id)
);
CREATE TABLE comments (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  item_id VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Key Configuration Files

- `config/config.toml` — Gorse core configuration (DB, ports, algorithm parameters, LLM)
- `fashion-recommend/config/config.toml` — fashion subsystem configuration
- `fashion-recommend/docker-compose.yml` — spins up Postgres, Redis, and all three Gorse nodes

---

## Stage 4: Frontend HITL Integration — Design Decisions

> Branch: `feat/HITL`. These decisions were made but not yet implemented.
> Pick up from here in the next session.

### Decision 1: Replace Go chat with Python agent on `/ai-chat`

The current `/ai-chat` page calls the Go backend's `/api/ai/chat` endpoint, which is a
stateless single-shot LLM call with no connection to Gorse or the real product catalogue
(it hallucinates recommendations from training data).

**Chosen approach:** Wire `/ai-chat` to the Python agent (`POST /api/ai/agent-chat` on
`:8001`) instead. The Python agent is strictly better: it queries Gorse for real products
via `get_recommendations` tool, has multi-turn PostgreSQL-backed memory, and supports HITL.
No new page or tab needed — same URL, better backend.

### Decision 2: HITL approval card is inline in the chat thread

When `pending_approval: true` is returned by the agent, an approval card appears as the
**next message bubble** in the conversation — not a modal, not a sidebar, not a separate page.

Rationale: the approval is a direct response to something the user just said. Routing them
away breaks the conversational moment.

**Card behaviour:**
- Appears immediately after the AI's response message
- Shows the staged trait fields with scores (e.g. `minimalist 0.8`, `price: low`)
- Input box is **disabled** while the card is visible (no new messages mid-approval)
- **Confirm** button → calls `POST /api/ai/agent-resume` with `{ approved: true }`; card
  replaced by a small success chip ("✓ Preferences saved"); input re-enabled
- **Cancel** button → calls `POST /api/ai/agent-resume` with `{ approved: false }`; card
  dismisses quietly; input re-enabled
- Uses the existing orange/amber colour palette from `ProductCard.tsx`

### Decision 3: Proxy routing

Add two new proxy entries to `fashion-recommend/frontend/vite.config.ts`:
```
/api/ai/agent-chat  →  http://localhost:8001
/api/ai/agent-resume → http://localhost:8001
```
The existing `/api` proxy (→ `:5001`) still handles all Go backend routes.

### Files to change for Stage 4

| File | Change |
|---|---|
| `fashion-recommend/frontend/vite.config.ts` | Add proxy entries for `:8001` agent endpoints |
| `fashion-recommend/frontend/src/services/api.ts` | Add `agentChat()` and `agentResume()` functions |
| `fashion-recommend/frontend/src/pages/AIChat.tsx` | Switch endpoint; add approval card UI; disable input during pending |

---

## Known Gaps / Future Work

### Feedback loop (wired — pipeline only, not a training signal source)

Frontend interactions now reach Gorse. Treat this as **plumbing, not data**: the
models train on `reco_interactions`' train split, and a handful of demo clicks
changes nothing about eval numbers.

| Interaction | Gorse type | Where |
|---|---|---|
| Like ❤️ | `favorite` | `api/like_handlers.go` → `sendFeedback`, best-effort (a Gorse failure must not 500 a like that already persisted) |
| Add to Cart | `add_to_cart` | `ProductCard.tsx` → `handleAddToCart` |
| View / impression | `view` | `ProductCard.tsx` → `IntersectionObserver` |

**Verified end to end 2026-08-31** against a running stack (`PORT=5001 go run
main.go` + `npm run dev`, anonymous `guest` identity), reading back from Gorse
rather than trusting the API response:

| type | count | how |
|---|---|---|
| `favorite` | 1 | heart button |
| `add_to_cart` | 1 | Add button |
| `view` | 6 | scrolling, by hand |

The `view` timestamps are the useful part: 6 impressions arrived in 3 pairs
across 25 seconds on a page holding 20 cards. Without the gate a single scroll
would have stamped all 20 at once, so the batching is evidence the throttle
itself works, not merely that the request fires.

Two things blocked this and neither was in the feedback code. `product_likes`
did not exist, so every like 500'd before reaching `InsertFeedback` (see the
migrations section). And `view` cannot be verified from a headless or hidden
browser at all: with `document.visibilityState === "hidden"` Chrome stops
delivering IntersectionObserver callbacks, and emulating a viewport size does
not help because the page still is not compositing. A hand-rolled observer on
the same element stays silent too, which is how that was separated from a bug
in `ProductCard`. Verifying this one needs a real visible window.

Three things worth keeping:

- **The impression gate is 50% visible for 1 continuous second.** Without an area
  threshold, scrolling past a screen edge counts as a view; without a dwell time,
  one flick to the bottom of the feed stamps a view on every card on the page.
  `view` lands in `read_feedback_types`, so that noise directly dilutes the
  relative weight of positive feedback.
- **Unlike has no counterpart.** `GorseClient` has no delete-feedback method, so
  the `favorite` stays. Deliberate — see the "pipeline only" framing above.
- **The wire format is PascalCase.** `GorseFeedback` in `services/api.ts` matches
  Gorse's field names because the Go route just forwards the body. See the user
  label schema section for what the snake_case version was silently doing.

### Discover feed user id
`HomePage.tsx` reads `localStorage.getItem('username')`; the anonymous fallback is
now `guest`, matching `ProductCard.tsx` and `like_handlers.go`. It used to fall
back to `user_001` — a real user — which served one person's personalised feed to
every logged-out visitor.

> `npx tsc --noEmit` reports two pre-existing `TS6133` unused-variable errors in
> `HomePage.tsx` (`scrollY`, `navigate`). `npm run build` is plain `vite build`
> and does not typecheck, so the build is unaffected.

### Product images missing
`public/images/` contains generic stock photos, not `product_001.jpg` etc. `ProductCard`
falls back to showing the product ID as text. Add real product images or generate
placeholder images named `product_001.jpg` through `product_010.jpg`.

### Edge case hardening in Python agent (discussed, not coded)
- **409 guard on `chat()`** — reject new messages while a HITL interrupt is pending
- **409 guard on `resume()`** — reject resume calls when no interrupt is active
- **DB failure safety in `write_traits_node`** — `try/except` to prevent silent graph crash
- **Score clamping in `_merge_trait_updates`** — clamp style/colour scores to `[0.0, 1.0]`