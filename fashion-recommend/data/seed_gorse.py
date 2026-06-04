# Reads from rag_products (Amazon data), downloads product images, and seeds Gorse
# with 1,000 items + synthetic demo feedback for CF training signal.
#
# Run: python fashion-recommend/data/seed_gorse.py

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg2
import psycopg2.extras

# ── Tunable parameters ────────────────────────────────────────────────────────

N_PRODUCTS   = 1000
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


# ── Data loading ──────────────────────────────────────────────────────────────

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

    products = []
    for row in rows:
        labels = []
        if row["brand"]:
            labels.append(f"brand:{row['brand']}")
        if row["colour"]:
            labels.append(f"color:{row['colour']}")
        if row["occasion"]:
            labels.append(f"occasion:{row['occasion']}")
        if row["material"]:
            labels.append(f"material:{row['material']}")
        labels.append(f"price_range:{row['price_range'] or 'mid'}")
        labels.append(f"item_name:{row['name']}")
        if row["price"] is not None:
            labels.append(f"price:{int(row['price'])}")
        if row["avg_rating"] is not None:
            labels.append(f"avg_rating:{row['avg_rating']}")

        products.append({
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
        })

    return products


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


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print(f"Loading top {N_PRODUCTS} products from rag_products...")
    products = load_products()
    print(f"  Loaded {len(products)} products\n")

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
    run()
