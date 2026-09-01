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

import json
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


_TYPE_SLUG = re.compile(r"[^a-z0-9]+")


def _product_type(raw: str) -> str:
    """The raw category column, slugified — `type:` labels come from here.

    This is the single most informative feature the catalogue has, and it was
    being thrown away. Measured over all 95,335 eval products:

        raw category   106 distinct   100% coverage   2 singletons
        _normalise_category rollup     12 distinct    100% coverage

    The rollup is what Gorse's Categories field uses (for filtering), and it is
    kept as a coarse `cat:` label as well. Keeping BOTH levels is the point: it
    is the only way this label schema can express "somewhat alike". Jaccard has
    exactly two states per label, shared or not, so a flat vocabulary can only
    say "same" or "different". With a two-level taxonomy a pair lands on one of
    three rungs instead:

        t-shirt vs t-shirt   shares type:t-shirt AND cat:tops
        t-shirt vs blouse    shares cat:tops only
        t-shirt vs necklace  shares neither

    Singletons are left in rather than filtered. Gorse's frequency filter
    (master/tasks.go:369-377) already drops them from the CTR index, and on the
    similarity path a singleton simply never matches anything — it costs one
    dictionary entry and cannot distort a score.
    """
    return _TYPE_SLUG.sub("-", (raw or "").lower().strip()).strip("-")


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


# ── Garment attributes ────────────────────────────────────────────────────────
#
# WHY THESE EXIST, since they were not in the original plan for this change.
#
# The restructure was meant to stop at "carriers off the similarity path,
# type/cat on it". Replaying Gorse's own tags distance offline
# (logics/item_to_item.go:337, score = 1/(1+distance) at :127) over the real
# 95,335-product catalogue showed that stopping there swaps one failure for
# another:
#
#   flat schema    every score in 0.5007-0.5107, nothing distinguishable
#   type+cat+style+color   MEDIAN top-10 has ONE distinct score
#
# The second is exact ties. type/cat/style/color put the median item in an
# equivalence class of 66 items with byte-identical feature sets, and 80.5% of
# the catalogue sits in a class of 10 or more — so a top-10 is typically filled
# by one class, all at distance 0, ordered arbitrarily by the ANN index.
#
# Six more dimensions, all read out of the product title, cut the median class
# from 66 items to 3 and take a top-10 from 1 distinct score to 7.
#
# ── What was measured and rejected ───────────────────────────────────────────
#
#   brand, floored at >=10 items   55.3% coverage, 1,733 values. Cuts the median
#       class 3 -> 2 and adds one distinct score, but its IDF is ~7.5 against
#       ~1.9 for `cat`, so it dominates a norm the real attributes barely move.
#       That is the carrier dynamic in a milder form, bought for almost nothing.
#   price deciles   11.9% of the eval catalogue has a price at all. Not viable.
#   avg_rating      100% coverage, and still excluded: two items rated 4.5 are
#       not similar. Coverage is not relevance.
#
# Same content-tagging caveat as style and colour: these are keyword heuristics
# over Amazon title text, not ground truth, and coverage is printed on every run.
# Coverage is uneven by design (audience 82%, length 10%) -- an attribute that is
# absent simply adds no resolution for that item, and type/cat still guarantee
# nobody ends up with an empty feature set.

ATTRIBUTE_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "audience": {
        # "men" must not fire inside "women" -- the lookbehind below is what
        # prevents it, and it is the reason these are regexes and not `in`.
        "women": ["women", "womens", "woman", "ladies", "female"],
        "men":   ["men", "mens", "man", "male"],
        "girls": ["girl", "girls"],
        "boys":  ["boy", "boys"],
        "kids":  ["kids", "children", "toddler", "baby", "infant", "junior",
                  "juniors", "youth"],
        "unisex": ["unisex"],
    },
    "pattern": {
        "striped":    ["striped", "stripe", "stripes", "pinstripe"],
        "floral":     ["floral", "flower print", "flowers"],
        "plaid":      ["plaid", "checkered", "check print", "gingham", "tartan"],
        "polka-dot":  ["polka dot", "polka dots", "dotted"],
        "animal":     ["leopard", "animal print", "cheetah", "zebra", "snakeskin"],
        "camo":       ["camo", "camouflage"],
        "tie-dye":    ["tie dye", "tie-dye"],
        "solid":      ["solid", "solid color", "solid colour"],
        "graphic":    ["graphic", "graphic print", "printed", "print"],
        "embellished": ["embroidered", "embroidery", "sequin", "sequins",
                        "beaded", "rhinestone"],
    },
    "sleeve": {
        "long":       ["long sleeve", "long-sleeve", "long sleeved"],
        "short":      ["short sleeve", "short-sleeve", "short sleeved"],
        "three-quarter": ["3/4 sleeve", "three quarter sleeve"],
        "cap":        ["cap sleeve"],
        "puff":       ["puff sleeve", "balloon sleeve", "bishop sleeve"],
        "sleeveless": ["sleeveless", "tank top", "camisole"],
        "strapless":  ["strapless", "tube top"],
        "halter":     ["halter"],
        "spaghetti":  ["spaghetti strap", "spaghetti straps"],
    },
    "fit": {
        "slim":       ["slim fit", "slim-fit", "fitted", "skinny"],
        "regular":    ["regular fit", "classic fit", "straight fit"],
        "loose":      ["loose", "loose fit", "relaxed fit", "baggy"],
        "oversized":  ["oversized", "oversize"],
        "bodycon":    ["bodycon", "body con"],
        "a-line":     ["a-line", "a line"],
        "wide-leg":   ["wide leg", "wide-leg", "palazzo", "flare", "flared"],
        "high-waist": ["high waist", "high waisted", "high-waisted", "high rise"],
        "plus-size":  ["plus size", "plus-size"],
        "petite":     ["petite"],
    },
    "neck": {
        "v-neck":     ["v neck", "v-neck", "vneck"],
        "crew":       ["crew neck", "crewneck", "round neck"],
        "scoop":      ["scoop neck"],
        "turtleneck": ["turtleneck", "turtle neck", "mock neck"],
        "off-shoulder": ["off shoulder", "off-shoulder", "cold shoulder"],
        "cowl":       ["cowl neck"],
        "square":     ["square neck"],
        "collared":   ["collared", "button down", "button-down"],
    },
    "length": {
        "maxi":    ["maxi", "floor length", "ankle length"],
        "midi":    ["midi", "knee length"],
        "mini":    ["mini", "short dress"],
        "cropped": ["crop", "cropped", "crop top"],
        "longline": ["longline", "tunic length"],
    },
}

_ATTRIBUTE_PATTERNS: dict[str, dict[str, list[re.Pattern]]] = {
    dim: {
        value: [re.compile(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])")
                for kw in kws]
        for value, kws in values.items()
    }
    for dim, values in ATTRIBUTE_KEYWORDS.items()
}


def _attribute_labels(name: str | None, description: str | None) -> list[str]:
    """Garment attributes read out of the product title. See the note above."""
    blob = f"{name or ''} {description or ''}".lower()
    out = []
    for dim in ATTRIBUTE_KEYWORDS:
        for value, patterns in _ATTRIBUTE_PATTERNS[dim].items():
            if any(p.search(blob) for p in patterns):
                out.append(f"{dim}:{value}")
    return sorted(out)


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


def build_item_labels(row: dict) -> tuple[dict, list[str], list[str]]:
    """Build the Gorse item labels for one rag_products row.

    Pure — no DB, no network — so the label schema can be unit-tested directly
    (see test_seed_labels.py). Returns (labels, style_labels, colour_labels).

    ── The map form, and why the labels are no longer one flat list ───────────

    Gorse accepts `Labels` as either a flat `[]string` or a nested map, and the
    two reach different consumers:

      * tags item-to-item reads ONE branch, named by an expr expression
        (`column = "item.Labels.f"`). `logics`' flatten (item_to_item.go:379)
        accepts `dataset.ID` / `[]dataset.ID` / `map[string]any`, so a string
        array under `f` arrives intact and nothing outside `f` is visible to it.
      * the CTR model reads the WHOLE map (`ctr.ConvertLabels`, model/ctr/data.go:39),
        flattening `{"brand": "Nike"}` to the feature name `brand.Nike`.

    That split is the entire fix. Measured on the flat schema, item-to-item
    scores spanned 0.500689–0.510655 with a within-list spread of 0.000339 —
    every score inside 1% of [0,1]. The cause was not the similarity function:

        the labels that discriminate never match,
        and the labels that match don't discriminate.

    `item_name` and `brand` were near-unique (2,186 and 1,492 distinct in a
    3,000-item sample), so they never matched. `price_range` had 4 values with
    ~80% on "mid" and `avg_rating` 25 coarse buckets, so they matched almost
    always. Two arbitrary items therefore shared "price_range:mid" plus
    "avg_rating:4.5" and nothing else — and two items rated 4.5 are not similar
    in any sense a shopper would recognise.

    So `f` now carries only labels that are BOTH shared often enough to match
    and specific enough to mean something:

        type:   106 distinct, 100% coverage   (raw category column)
        cat:     12 distinct, 100% coverage   (coarse rollup, see _product_type)
        style:    6 distinct,  48.7% coverage (keyword heuristic)
        color:   10 distinct,  62.1% coverage (keyword heuristic)
        audience/pattern/sleeve/fit/neck/length -- see ATTRIBUTE_KEYWORDS,
                  added because type+cat+style+color alone left the median item
                  in an equivalence class of 66 with a top-10 of pure ties
        occasion / material                   (rag_products only; the eval
                                               catalogue has neither column)

    Roughly 6,360 equivalence classes against the flat schema's ~336, and —
    because type/cat cover everything — zero items with an empty feature set,
    down from 21.0%.

    `brand`, `price_range` and `avg_rating` are NOT dropped. They are real CTR
    features (an FM can learn a brand embedding); they were only ever wrong as
    *similarity* signals. Moving them out of `f` keeps them in the CTR path and
    takes them off the similarity path, which is a different statement from
    deleting them.

    Carriers (`item_name`, `price`) leave Labels altogether — see
    build_item_comment.

    Invariant enforced by the tests: NO label embeds a float. A score or measure
    baked into the label string turns one feature into many near-unique ones,
    which Gorse then drops wholesale. That is exactly the failure the user-side
    trait sync suffered from (traits/gorse_sync.go emitted "style:minimalist:0.8"),
    and this side must not reintroduce it.
    """
    colour_labels = _colour_labels(row.get("colour"), row["name"], row["description"])
    style_labels  = _style_labels(row["name"], row["description"])

    features = []
    product_type = _product_type(row["category"])
    if product_type:
        features.append(f"type:{product_type}")
    features.append(f"cat:{_normalise_category(row['category'])}")
    features.extend(f"style:{s}" for s in style_labels)
    features.extend(f"color:{c}" for c in colour_labels)
    features.extend(_attribute_labels(row["name"], row["description"]))
    if row.get("occasion"):
        features.append(f"occasion:{row['occasion']}")
    if row.get("material"):
        features.append(f"material:{row['material']}")

    labels: dict = {"f": features}

    if row["brand"]:
        labels["brand"] = str(row["brand"])

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
        labels["price_range"] = "unknown"
    else:
        labels["price_range"] = str(row["price_range"] or "mid")

    # Rounded to one decimal deliberately, so its cardinality is bounded here
    # (~41 possible values) rather than left to whatever precision the upstream
    # column happens to hold. An unrounded 4.333333 would be a singleton, i.e. a
    # silently dropped feature. Kept as a STRING: a JSON number under a map key
    # reaches neither model — ctr.convertLabels has no float64 branch (gorm's
    # json serializer decodes into float64, not json.Number) and flatten ignores
    # it — so a numeric leaf here would look like a feature and be inert.
    if row["avg_rating"] is not None:
        labels["avg_rating"] = f"{round(float(row['avg_rating']), 1)}"

    return labels, style_labels, colour_labels


def build_item_comment(row: dict) -> str:
    """The carrier payload: data the frontend needs and no model should see.

    `item_name` and `price` were labels, which was the wrong field for them.
    They are not features — they exist so api/server.go's enrichment loop can
    fill in name and price for the frontend — but sitting in Labels they were
    charged as features anyway. On the tags path they were the worst kind: both
    near-unique, so their IDF is about log(95335) ≈ 11.5 each against ≈ 1.5 for
    a genuinely shared style tag, and the distance function (item_to_item.go:337)
    divides by sqrt of the sum over ALL of an item's tags. Two carriers added
    ~23 to a norm the real features barely moved.

    Comment is the right field because it is provably inert: grepping
    logics/ master/ model/ dataset/ for `.Comment` returns nothing. It is
    storage that no recommender reads, which is exactly what a carrier wants.

    The product description rides along in the same JSON — it used to be
    Comment's whole content, and it is still worth keeping for the dashboard
    and for a future embedding arm.
    """
    return json.dumps({
        "name":  row["name"] or "",
        "price": float(row["price"]) if row["price"] is not None else None,
        "desc":  (row["description"] or "")[:400],
    }, ensure_ascii=False)


def _build_product(row: dict) -> dict:
    labels, style_labels, colour_labels = build_item_labels(row)
    return {
        "product_id": row["product_id"],
        "image_url":  row["image_url"] or "",
        "gorse_item": {
            "ItemId":     row["product_id"],
            "Categories": [_normalise_category(row["category"])],
            "Labels":     labels,
            "Comment":    build_item_comment(row),
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
    "type":  "raw category column — the strongest feature available",
    "cat":   "coarse rollup — gives Jaccard a middle rung",
    # Amazon Fashion is a long tail of small sellers: ~650 brands over 1,000
    # items, most appearing once. Dropping a brand seen a single time is the
    # correct behaviour, not a bug — one occurrence carries no CF signal. Left
    # as-is deliberately; the numbers are printed so the tail stays visible.
    "brand": "long tail — singleton drop is expected; CTR only",
    # These two were the compression. They match almost every pair, so they
    # added no ranking information while dominating the tag norm. Kept as CTR
    # features, taken off the similarity path.
    "price_range": "CTR only — ~80% 'mid', useless as similarity",
    "avg_rating":  "CTR only — two items rated 4.5 are not similar",
    # Present only when auditing the OLD flat schema.
    "item_name": "carrier — now in Comment, not Labels",
    "price":     "carrier — now in Comment, not Labels",
}

def flatten_labels(labels) -> list[tuple[str, str, bool]]:
    """Yield (prefix, "prefix:value", on_similarity_path) for one item's labels.

    Accepts both shapes so the audit can be run against the old flat schema too:
    a flat list is reported as entirely on the similarity path, which is exactly
    what it was, and what the compression measurement was about.
    """
    if isinstance(labels, list):
        return [(l.split(":", 1)[0], l, True) for l in labels]
    out = []
    for key, value in labels.items():
        if key == "f":
            out.extend((l.split(":", 1)[0], l, True) for l in value)
        else:
            out.append((key, f"{key}:{value}", False))
    return out


def report_labels(products: list[dict]) -> None:
    """Print what the item labels will actually look like to Gorse.

    Gorse indexes a label only once it has been seen a SECOND time
    (master/tasks.go:369-377); singletons are dropped and never reach the CTR
    model. Printing the drop rate per prefix here means a label schema change
    that silently guts a feature shows up in the seed log instead of being
    discovered later as an unexplained flat metric.

    The `sim` column is the one to read after the map-form restructure: it says
    whether a prefix reaches tags item-to-item at all. A prefix with `sim = no`
    is still a CTR feature — it has been taken off the similarity path, which is
    not the same as being deleted.
    """
    n = len(products)
    counts: Counter[str] = Counter()
    on_sim: dict[str, bool] = {}
    for p in products:
        for prefix, label, sim in flatten_labels(p["gorse_item"]["Labels"]):
            counts[label] += 1
            on_sim[prefix] = sim

    by_prefix: dict[str, list[int]] = {}
    for label, c in counts.items():
        by_prefix.setdefault(label.split(":", 1)[0], []).append(c)

    print(f"{'─'*80}")
    print("Item label audit (what Gorse will index)")
    print(f"{'─'*80}")
    print(f"{'prefix':<14} {'sim':>4} {'distinct':>9} {'singleton':>10} {'indexed':>8}  note")
    for prefix in sorted(by_prefix, key=lambda k: (not on_sim.get(k), k)):
        cs = by_prefix[prefix]
        singles = sum(1 for c in cs if c < 2)
        indexed = len(cs) - singles
        note = _PREFIX_NOTES.get(prefix, "")
        if not note and cs and singles / len(cs) > 0.5:
            note = "WARN: majority singleton — check for free-text values"
        sim = "yes" if on_sim.get(prefix) else "no"
        print(f"{prefix:<14} {sim:>4} {len(cs):>9} {singles:>10} {indexed:>8}  {note}")

    # ── The similarity path's actual resolving power ──────────────────────────
    #
    # This block is the reason the schema changed, so it is reported on every
    # run rather than measured once. Two numbers:
    #
    #   empty feature sets  items tags item-to-item literally cannot place. On
    #                       the old schema this was 21.0% of the catalogue.
    #   equivalence classes distinct feature-label SETS. Items inside one class
    #                       are mutually indistinguishable to Jaccard, so this
    #                       is the ceiling on how finely the arm can ever rank.
    feature_sets = [frozenset(l for _, l, sim in flatten_labels(p["gorse_item"]["Labels"]) if sim)
                    for p in products]
    empty = sum(1 for fs in feature_sets if not fs)
    classes = len(set(feature_sets))
    sizes = Counter(feature_sets)
    print()
    print("similarity path (labels under `f`, what column = \"item.Labels.f\" sees)")
    print(f"  empty feature sets : {empty}/{n} = {empty/n:.1%}  <- item-to-item cannot place these")
    print(f"  equivalence classes: {classes}  ({n/max(classes,1):.1f} items per class on average)")
    print(f"  largest class      : {sizes.most_common(1)[0][1]} items")
    print(f"  labels per item    : {sum(len(fs) for fs in feature_sets)/n:.2f} mean")

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
                  if p["gorse_item"]["Labels"].get("price_range") == "unknown")
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
