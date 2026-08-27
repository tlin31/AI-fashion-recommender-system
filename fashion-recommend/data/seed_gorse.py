# Reads from rag_products (Amazon data), downloads product images, and seeds Gorse
# with N_PRODUCTS items + synthetic demo feedback for CF training signal.
#
# Run: python fashion-recommend/data/seed_gorse.py
#
# ── Two catalogue sizes, do not conflate them ────────────────────────────────
#
# There are two different numbers in play, and mixing them up silently corrupts
# any coverage metric computed later:
#
#   rag_products rows  — the full normalised Amazon catalogue in Postgres (5,000).
#                        This is what the join statistics in the eval plan are
#                        measured against.
#   N_PRODUCTS         — how many of those this script actually posts to Gorse,
#                        taken as the top-N by rating_count (default 1,000).
#                        Gorse only ever knows about these, so catalog-coverage
#                        denominators must use N_PRODUCTS, not the table size.
#
# Override with the SEED_N_PRODUCTS env var. run() prints both numbers so the
# distinction is visible in the log of every seed run.

from __future__ import annotations

import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras

# ── Tunable parameters ────────────────────────────────────────────────────────

N_PRODUCTS   = int(os.environ.get("SEED_N_PRODUCTS", "1000"))
N_VIEWS      = 8    # view events per demo user
N_LIKES      = 3    # favorite events per demo user (subset of views)
GORSE_URL    = os.environ.get("GORSE_URL", "http://localhost:8088")
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://gorse:gorse_pass@localhost:5432/gorse"
)

IMAGES_DEST = Path(__file__).parent.parent / "public" / "images"
DEMO_USERS  = ["user_001", "user_002", "user_003", "user_004", "user_005"]

# ── Category normalisation ────────────────────────────────────────────────────
# Collapses the 60+ granular keywords from normalize.py into ~12 Gorse-level
# buckets wide enough for CF item-to-item similarity to find patterns.

_CATEGORY_MAP: dict[str, set[str]] = {
    "dresses":    {"dress", "maxi", "midi", "mini", "wrap", "romper", "jumpsuit",
                   "bodysuit", "kimono", "caftan"},
    "tops":       {"top", "t-shirt", "crop top", "tank", "polo", "tee", "blouse",
                   "tunic", "shirt"},
    "outerwear":  {"jacket", "coat", "blazer", "hoodie", "sweater", "sweatshirt",
                   "cardigan", "pullover", "vest", "parka", "puffer", "windbreaker",
                   "overcoat", "raincoat", "shrug", "turtleneck", "fleece", "thermal"},
    "bottoms":    {"jeans", "pants", "trousers", "leggings", "shorts", "joggers",
                   "cargos", "skirt", "palazzo", "churidar", "salwar"},
    "footwear":   {"sneaker", "boot", "sandal", "loafer", "slipper", "moccasin",
                   "oxford", "heel", "pump", "flat", "shoe"},
    "accessories":{"handbag", "purse", "clutch", "tote", "scarf", "belt", "hat",
                   "cap", "beanie", "glove", "mittens", "headband", "sock",
                   "stocking", "tights"},
    "jewellery":  {"necklace", "bracelet", "earring", "anklet"},
    "swimwear":   {"swimsuit", "swimwear", "bikini", "tankini", "one-piece",
                   "rash guard", "board short", "swim trunk"},
    "intimates":  {"bra", "underwear", "panty", "panties", "lingerie", "pajama",
                   "pyjama", "nightgown", "sleepwear", "robe"},
    "ethnic":     {"kurta set", "co-ord", "lehenga", "kurti", "kurta", "saree", "dupatta"},
    "activewear": {"suit", "tracksuit", "track suit", "activewear", "sportswear",
                   "athletic", "compression"},
    "maternity":  {"maternity"},
}

def _normalise_category(raw: str) -> str:
    raw = (raw or "").lower().strip()
    for gorse_cat, keywords in _CATEGORY_MAP.items():
        if raw in keywords:
            return gorse_cat
    return "apparel"


# ── Item-side style tagging ───────────────────────────────────────────────────
#
# CONTENT-TAGGING ASSUMPTION — this is a heuristic, not ground truth.
#
# rag_products has no style column, so item-side `style:` labels are produced by
# keyword-matching the product name + description. Two things to know:
#
#   1. The six style classes deliberately mirror `traits/extractor.go`'s
#      styleKeywords, so the user side and the item side speak the same
#      vocabulary. That file's keywords are Chinese and the catalogue is English
#      Amazon text, so it cannot be reused directly — matching Chinese keywords
#      against English titles yields exactly zero hits. This is the English twin.
#   2. Coverage is partial and is reported on every run (see run()). At the
#      default N_PRODUCTS=1000 it is ~69%; the remaining ~31% of items get no
#      style label at all. Any claim made about style labels has to be read
#      against that number.

STYLE_KEYWORDS: dict[str, list[str]] = {
    "minimalist": ["minimalist", "minimal", "simple", "basic", "clean", "plain",
                   "solid", "essential", "understated", "sleek"],
    "casual":     ["casual", "everyday", "relaxed", "comfy", "comfortable",
                   "lounge", "loungewear", "weekend", "laid-back"],
    "formal":     ["formal", "business", "office", "professional", "elegant",
                   "dress shirt", "suit", "blazer", "tailored", "work wear",
                   "workwear", "cocktail", "evening"],
    "streetwear": ["streetwear", "street", "urban", "hip hop", "hiphop",
                   "oversized", "graphic", "skate", "athletic", "sporty",
                   "sport", "gym", "activewear"],
    "vintage":    ["vintage", "retro", "classic", "timeless", "throwback",
                   "old school", "heritage", "traditional"],
    "romantic":   ["romantic", "sweet", "feminine", "lace", "floral", "ruffle",
                   "bow", "chiffon", "dainty", "delicate"],
}

# Pre-compiled with word boundaries that tolerate spaces/hyphens inside a phrase
# but refuse partial-word hits — without this, "sport" matches "transportation"
# and "bow" matches "elbow".
_STYLE_PATTERNS: dict[str, list[re.Pattern]] = {
    style: [re.compile(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])") for kw in kws]
    for style, kws in STYLE_KEYWORDS.items()
}


def _style_labels(name: str | None, description: str | None) -> list[str]:
    blob = f"{name or ''} {description or ''}".lower()
    return sorted(
        style for style, patterns in _STYLE_PATTERNS.items()
        if any(p.search(blob) for p in patterns)
    )


# ── Item-side colour normalisation ────────────────────────────────────────────
#
# The rag_products.colour column is unusable as a feature straight from the DB:
# at N_PRODUCTS=1000 it is the empty string for 941 items, and the surviving 59
# values are uncontrolled free text — 'cut vines', 'smokey quartz', '3x', 'no',
# '3 bands - red + white + blue'. That produced 45 distinct `color:` labels of
# which 41 were singletons, so Gorse's frequency filter dropped 91% of them.
#
# Worse, none of it lined up with the USER side: traits/extractor.go emits a
# canonical ten-colour palette (color:black, color:blue, ...), so even a
# correctly-indexed `color:smokey quartz` could never correspond to anything a
# user trait expresses. The palette below is that same ten-colour vocabulary,
# so the two sides finally describe colour in the same terms.
#
# Same content-tagging caveat as style: this is a heuristic over product text,
# and coverage is reported on every run.

COLOUR_KEYWORDS: dict[str, list[str]] = {
    "black":  ["black", "jet black", "onyx"],
    "white":  ["white", "ivory", "cream", "off-white"],
    "gray":   ["gray", "grey", "charcoal", "slate", "silver"],
    "blue":   ["blue", "navy", "denim", "teal", "turquoise", "indigo", "cobalt"],
    "red":    ["red", "burgundy", "maroon", "wine", "crimson", "scarlet"],
    "pink":   ["pink", "rose", "blush", "fuchsia", "magenta", "coral"],
    "beige":  ["beige", "khaki", "tan", "camel", "nude", "sand", "taupe"],
    "brown":  ["brown", "chocolate", "coffee", "espresso", "mocha", "bronze"],
    "green":  ["green", "olive", "sage", "emerald", "mint", "forest"],
    "yellow": ["yellow", "mustard", "gold", "golden", "amber"],
}

_COLOUR_PATTERNS: dict[str, list[re.Pattern]] = {
    colour: [re.compile(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])") for kw in kws]
    for colour, kws in COLOUR_KEYWORDS.items()
}


def _colour_labels(colour: str | None, name: str | None,
                   description: str | None) -> list[str]:
    """Map the colour column AND the product text onto the canonical palette.

    The column is searched too (not just trusted), because values like
    'black/white' or 'cool black' carry a real colour that plain string use
    would fragment into its own singleton label.
    """
    blob = f"{colour or ''} {name or ''} {description or ''}".lower()
    return sorted(
        c for c, patterns in _COLOUR_PATTERNS.items()
        if any(p.search(blob) for p in patterns)
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def count_catalogue() -> int:
    """Total rows in rag_products — the denominator that is NOT the seeded one."""
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM rag_products")
            return cur.fetchone()[0]
    finally:
        conn.close()


def load_products() -> list[dict]:
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT product_id, name, brand, category, colour, occasion,
                       material, price, price_range, avg_rating, rating_count,
                       description, image_url
                FROM rag_products
                ORDER BY rating_count DESC
                LIMIT %s
            """, (N_PRODUCTS,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [_build_product(row) for row in rows]


def build_item_labels(row: dict) -> tuple[list[str], list[str], list[str]]:
    """Build the Gorse item labels for one rag_products row.

    Pure — no DB, no network — so the label schema can be unit-tested directly
    (see test_seed_labels.py). Returns (labels, style_labels, colour_labels).

    Invariant enforced by the tests: NO label embeds a float. A score or measure
    baked into the label string turns one feature into many near-unique ones,
    which Gorse then drops wholesale. That is exactly the failure the user-side
    trait sync suffers from (traits/gorse_sync.go emits "style:minimalist:0.8"),
    and this side must not reintroduce it.
    """
    labels = []
    if row["brand"]:
        labels.append(f"brand:{row['brand']}")
    colour_labels = _colour_labels(row["colour"], row["name"], row["description"])
    labels.extend(f"color:{c}" for c in colour_labels)
    if row["occasion"]:
        labels.append(f"occasion:{row['occasion']}")
    if row["material"]:
        labels.append(f"material:{row['material']}")
    # price_range: 'mid' is a null sentinel upstream, not a price band.
    #
    # rag-service/data/normalize.py:340 assigns "mid" to every product whose
    # price is NULL:  bucket_price(...) if pd.notna(p) else "mid".
    # 68% of rag_products have no price, so at N_PRODUCTS=1000 the raw column
    # reads 803 'mid' — but only 131 of those 803 actually have a price. The
    # other 672 mean "unknown". Emitting that as `price_range:mid` hands the
    # CTR model a feature whose dominant value is a missing-data marker, and
    # tells the frontend a made-up price band.
    #
    # normalize.py belongs to rag-service, whose eval baseline is locked, so
    # this is corrected here at seed time rather than upstream.
    if row["price"] is None:
        labels.append("price_range:unknown")
    else:
        labels.append(f"price_range:{row['price_range'] or 'mid'}")

    # Carrier labels, not model features.
    #
    # `item_name:` is unique by construction and `price:` is near-unique, so
    # Gorse's frequency filter (master/tasks.go:307-330 indexes a label only
    # on its SECOND occurrence) drops them from the model. That is fine —
    # they exist so api/server.go:246 getRecommend can enrich responses for
    # the frontend. They are deliberately left in place: removing `price:`
    # would silently zero out item.Price in the API response.
    #
    # Measured, so the "high cardinality" worry is not taken on faith: at
    # N_PRODUCTS=1000, int-truncated price yields 54 distinct labels of which
    # only 13 are singletons. A coarser price bucket was considered and
    # rejected as redundant — price_range already IS that bucket, once the
    # null conflation above is fixed.
    if row["name"]:
        labels.append(f"item_name:{row['name']}")
    if row["price"] is not None:
        labels.append(f"price:{int(row['price'])}")
    # 18 distinct values at N=1000, 2 of them singletons — dense enough to
    # survive the frequency filter and serve as a real feature.
    #
    # Rounded to one decimal deliberately: this is the one label that legitimately
    # carries a number, so its cardinality is bounded here (~41 possible values)
    # rather than left to whatever precision the upstream column happens to hold.
    # An unrounded 4.333333 would be a singleton, i.e. a silently dropped feature.
    if row["avg_rating"] is not None:
        labels.append(f"avg_rating:{round(float(row['avg_rating']), 1)}")

    # Heuristic content tags — see STYLE_KEYWORDS above.
    style_labels = _style_labels(row["name"], row["description"])
    labels.extend(f"style:{s}" for s in style_labels)

    return labels, style_labels, colour_labels


def _build_product(row: dict) -> dict:
    labels, style_labels, colour_labels = build_item_labels(row)
    return {
        "product_id": row["product_id"],
        "image_url":  row["image_url"] or "",
        "gorse_item": {
            "ItemId":     row["product_id"],
            "Categories": [_normalise_category(row["category"])],
            "Labels":     labels,
            "Comment":    (row["description"] or "")[:500],
            "Timestamp":  datetime.now(timezone.utc).isoformat() + "Z",
        },
        "_name":         row["name"],
        "_category":     row["category"],
        "_gorse_cat":    _normalise_category(row["category"]),
        "_price":        row["price"],
        "_rating_count": row["rating_count"],
        "_styles":       style_labels,
        "_colours":      colour_labels,
        "_raw_colour":   row["colour"],
    }


# ── Image downloading ─────────────────────────────────────────────────────────

def download_images(products: list[dict]) -> tuple[int, int]:
    IMAGES_DEST.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0

    for p in products:
        dest = IMAGES_DEST / f"{p['product_id']}.jpg"
        if dest.exists():
            skipped += 1
            continue
        if not p["image_url"]:
            continue
        try:
            resp = httpx.get(p["image_url"], timeout=10, follow_redirects=True)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                downloaded += 1
            else:
                print(f"  [image] {p['product_id']}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [image] {p['product_id']}: {e}")
        time.sleep(0.05)   # be polite to Amazon CDN

    return downloaded, skipped


# ── Gorse API calls ───────────────────────────────────────────────────────────

BATCH_SIZE = 100   # Gorse recommends batching, not single large POST

def post_items(products: list[dict]) -> None:
    items = [p["gorse_item"] for p in products]
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        for attempt in range(1, 5):
            try:
                resp = httpx.post(f"{GORSE_URL}/api/items", json=batch, timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = attempt * 2
                print(f"  Batch {i+1}–{i+len(batch)}: error ({e}), retrying in {wait}s...")
                time.sleep(wait)
        print(f"  Posted items {i+1}–{i+len(batch)}/{len(items)}")


def post_feedback(products: list[dict]) -> None:
    item_ids = [p["product_id"] for p in products]
    now = datetime.now(timezone.utc)
    feedback = []

    for user in DEMO_USERS:
        viewed = random.sample(item_ids, min(N_VIEWS, len(item_ids)))
        liked  = random.sample(viewed,   min(N_LIKES, len(viewed)))

        for i, item_id in enumerate(viewed):
            feedback.append({
                "FeedbackType": "view",
                "UserId":       user,
                "ItemId":       item_id,
                "Timestamp":    (now - timedelta(days=i + 1)).isoformat() + "Z",
            })
        for item_id in liked:
            feedback.append({
                "FeedbackType": "favorite",
                "UserId":       user,
                "ItemId":       item_id,
                "Timestamp":    (now - timedelta(days=1)).isoformat() + "Z",
            })

    resp = httpx.post(f"{GORSE_URL}/api/feedback", json=feedback, timeout=30)
    resp.raise_for_status()
    print(f"  Inserted {len(feedback)} feedback events across {len(DEMO_USERS)} users.")


# ── Label audit ───────────────────────────────────────────────────────────────

# Prefixes whose singleton rate is expected and NOT a defect, so a high drop
# count here does not trip the WARN heuristic below.
_PREFIX_NOTES = {
    "item_name": "carrier only — dropped by design",
    "price":     "carrier only — dropped by design",
    # Amazon Fashion is a long tail of small sellers: ~650 brands over 1,000
    # items, most appearing once. Dropping a brand seen a single time is the
    # correct behaviour, not a bug — one occurrence carries no CF signal. Left
    # as-is deliberately; the numbers are printed so the tail stays visible.
    "brand":     "long tail — singleton drop is expected",
}

def report_labels(products: list[dict]) -> None:
    """Print what the item labels will actually look like to Gorse.

    Gorse indexes a label only once it has been seen a SECOND time
    (master/tasks.go:307-330); singletons are dropped and never reach the CTR
    model. Printing the drop rate per prefix here means a label schema change
    that silently guts a feature shows up in the seed log instead of being
    discovered later as an unexplained flat metric.
    """
    n = len(products)
    counts: Counter[str] = Counter()
    for p in products:
        for label in p["gorse_item"]["Labels"]:
            counts[label] += 1

    by_prefix: dict[str, list[int]] = {}
    for label, c in counts.items():
        by_prefix.setdefault(label.split(":", 1)[0], []).append(c)

    print(f"{'─'*80}")
    print("Item label audit (what Gorse will index)")
    print(f"{'─'*80}")
    print(f"{'prefix':<14} {'distinct':>9} {'singleton':>10} {'indexed':>8}  note")
    for prefix in sorted(by_prefix):
        cs = by_prefix[prefix]
        singles = sum(1 for c in cs if c < 2)
        indexed = len(cs) - singles
        note = _PREFIX_NOTES.get(prefix, "")
        if not note and cs and singles / len(cs) > 0.5:
            note = "WARN: majority singleton — check for free-text values"
        print(f"{prefix:<14} {len(cs):>9} {singles:>10} {indexed:>8}  {note}")

    # Style coverage — the content-tagging assumption's headline number.
    labelled = sum(1 for p in products if p["_styles"])
    per_item = Counter(len(p["_styles"]) for p in products)
    style_hits = Counter(s for p in products for s in p["_styles"])
    print()
    print(f"style: coverage  : {labelled}/{n} = {labelled/n:.1%} of items have >=1 style label")
    print(f"style: per item  : {dict(sorted(per_item.items()))}")
    print(f"style: per class : {dict(style_hits.most_common())}")

    c_labelled = sum(1 for p in products if p["_colours"])
    colour_hits = Counter(c for p in products for c in p["_colours"])
    raw_colour = sum(1 for p in products if p["_raw_colour"])
    print()
    print(f"color: coverage  : {c_labelled}/{n} = {c_labelled/n:.1%} of items have >=1 colour label "
          f"(raw colour column was populated for only {raw_colour})")
    print(f"color: per class : {dict(colour_hits.most_common())}")

    # price_range honesty check.
    priced = sum(1 for p in products if p["_price"] is not None)
    unknown = sum(1 for p in products
                  if "price_range:unknown" in p["gorse_item"]["Labels"])
    print()
    print(f"price            : {priced}/{n} = {priced/n:.1%} of items have a real price")
    print(f"price_range      : {unknown} items labelled 'unknown' "
          f"(upstream normalize.py would have called these 'mid')")
    print(f"{'─'*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    catalogue_total = count_catalogue()
    print(f"Loading top {N_PRODUCTS} products from rag_products (by rating_count)...")
    products = load_products()
    print(f"  rag_products total : {catalogue_total}   <- full normalised catalogue")
    print(f"  seeded into Gorse  : {len(products)}   <- coverage denominators use THIS")
    if len(products) < catalogue_total:
        print(f"  NOTE: Gorse sees {len(products)/catalogue_total:.1%} of the catalogue. "
              f"Raise with SEED_N_PRODUCTS=<n>.")
    print()

    # Preview table
    print(f"{'─'*80}")
    print(f"{'ID':<14} {'Raw cat':<14} {'Gorse cat':<12} {'PriceRange':<10} {'Ratings':>8}  Name")
    print(f"{'─'*80}")
    for p in products[:20]:
        print(
            f"{p['product_id']:<14} {p['_category']:<14} {p['_gorse_cat']:<12} "
            f"{(p['gorse_item']['Labels'] and next((l.split(':')[1] for l in p['gorse_item']['Labels'] if l.startswith('price_range:')), '—')):<10} "
            f"{p['_rating_count']:>8}  {p['_name'][:40]}"
        )
    if len(products) > 20:
        print(f"  ... and {len(products) - 20} more")
    print(f"{'─'*80}\n")

    report_labels(products)

    if dry_run:
        print("--dry-run: stopping before images / Gorse writes.")
        return

    print(f"Downloading images → {IMAGES_DEST}")
    downloaded, skipped = download_images(products)
    print(f"  Downloaded {downloaded}, skipped {skipped} (already exist)\n")

    print(f"Posting {len(products)} items to Gorse ({GORSE_URL}) in batches of {BATCH_SIZE}...")
    post_items(products)
    print("  Done.\n")

    print("Seeding feedback for demo users...")
    post_feedback(products)
    print()

    print("✓ Seed complete.")
    print(f"  Gorse dashboard : {GORSE_URL}")
    print(f"  Frontend        : http://localhost:5173")


if __name__ == "__main__":
    import sys
    # --dry-run: build and audit the labels, write nothing. Lets the label schema
    # be checked without a running Gorse (and without re-downloading images).
    run(dry_run="--dry-run" in sys.argv)
