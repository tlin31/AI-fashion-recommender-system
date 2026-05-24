# One-time script to seed Gorse with real Myntra products and copy product images.
# Also inserts synthetic feedback for demo users so the CF algorithm has signal.
# Run manually: python fashion-recommend/data/seed_gorse.py

from __future__ import annotations

import ast
import random
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

# ── Tunable parameters ────────────────────────────────────────────────────────

N_PRODUCTS = 10          # ← CHANGE THIS to seed more products (e.g. 1000, 5000)
N_VIEWS = 5              # ← CHANGE THIS: view events per demo user
N_LIKES = 2              # ← CHANGE THIS: favorite events per demo user (subset of views)
GORSE_URL = "http://localhost:8088"   # ← CHANGE THIS if Gorse runs on a different port

REPO_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = REPO_ROOT / "Myntra Fashion Product Dataset" / "Fashion Dataset.csv"
IMAGES_SRC = REPO_ROOT / "Myntra Fashion Product Dataset" / "Images" / "Images"
IMAGES_DEST = Path(__file__).parent.parent / "public" / "images"

DEMO_USERS = ["user_001", "user_002", "user_003", "user_004", "user_005"]

# ── Helpers (inlined — not imported from rag-service to keep services decoupled) ──

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_attrs(raw: str) -> dict:
    # p_attributes is a Python dict literal with single quotes, not valid JSON.
    try:
        result = ast.literal_eval(raw)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


# Keywords to scan the product name when Top Type is missing from p_attributes.
_CATEGORY_KEYWORDS = [
    "kurta", "saree", "dress", "top", "jeans", "trousers", "leggings",
    "skirt", "jacket", "blazer", "sweater", "sweatshirt", "hoodie",
    "shorts", "co-ord", "palazzo", "kurti", "suit",
]

def _derive_category(attrs: dict, name: str) -> str:
    # Prefer explicit Top Type; fall back to first keyword match in product name.
    top_type = attrs.get("Top Type", "").lower().strip()
    if top_type:
        return top_type
    name_lower = name.lower()
    for kw in _CATEGORY_KEYWORDS:
        if kw in name_lower:
            return kw
    return "apparel"


def bucket_price(price: float, p25: float, p75: float, p90: float) -> str:
    # Buckets match Gorse user price_preference labels for CF correlation.
    if price < p25:
        return "budget"
    if price < p75:
        return "mid-range"
    if price < p90:
        return "high-end"
    return "luxury"


# ── Data preparation ──────────────────────────────────────────────────────────

def load_products() -> list[dict]:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df.dropna(subset=["p_id", "name", "description", "ratingCount"])
    df["ratingCount"] = pd.to_numeric(df["ratingCount"], errors="coerce")
    df = df.dropna(subset=["ratingCount"])

    # Keep the N_PRODUCTS most-reviewed products — highest signal quality.
    df = df.nlargest(N_PRODUCTS, "ratingCount").reset_index()

    price_vals = pd.to_numeric(df["price"], errors="coerce").dropna()
    p25, p75, p90 = price_vals.quantile([0.25, 0.75, 0.90])

    products = []
    for row_index, row in df.iterrows():
        attrs = parse_attrs(str(row.get("p_attributes", "")))
        price = float(row["price"]) if pd.notna(row.get("price")) else 0.0
        product_id = str(int(float(row["p_id"])))

        labels = []
        if row.get("brand"):
            labels.append(f"brand:{str(row['brand']).lower().strip()}")
        if row.get("colour"):
            labels.append(f"color:{str(row['colour']).lower().strip()}")
        if attrs.get("Occasion"):
            labels.append(f"occasion:{attrs['Occasion'].lower().strip()}")
        if attrs.get("Top Fabric"):
            labels.append(f"material:{attrs['Top Fabric'].lower().strip()}")
        labels.append(f"price_range:{bucket_price(price, p25, p75, p90)}")
        labels.append("season:all_season")
        # Store name and price so the Go handler can enrich recommendation responses.
        # Use a pipe separator for name to avoid collision with the colon split in Go.
        labels.append(f"item_name:{str(row['name'])}")
        labels.append(f"price:{int(price)}")

        products.append({
            "product_id": product_id,
            "row_index": row_index,    # needed for image filename mapping
            "gorse_item": {
                "ItemId": product_id,
                "Categories": [_derive_category(attrs, str(row["name"]))],
                "Labels": labels,
                "Comment": strip_html(str(row["description"]))[:500],
                "Timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            },
            # for display / verification
            "_name": row["name"],
            "_price": price,
            "_rating_count": int(row["ratingCount"]),
        })

    return products


# ── Image copying ─────────────────────────────────────────────────────────────

def copy_images(products: list[dict]) -> int:
    IMAGES_DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in products:
        src = IMAGES_SRC / f"{p['row_index']}.jpg"
        dest = IMAGES_DEST / f"{p['product_id']}.jpg"
        if src.exists():
            shutil.copy(src, dest)
            copied += 1
        else:
            print(f"  [image] WARNING: {src.name} not found for product {p['product_id']}")
    return copied


# ── Gorse API calls ───────────────────────────────────────────────────────────

def post_items(products: list[dict]) -> None:
    items = [p["gorse_item"] for p in products]
    resp = httpx.post(f"{GORSE_URL}/api/items", json=items, timeout=30)
    resp.raise_for_status()


def post_feedback(products: list[dict]) -> None:
    item_ids = [p["product_id"] for p in products]
    now = datetime.now(timezone.utc)
    feedback = []

    for user in DEMO_USERS:
        # Each user views N_VIEWS random products and likes a subset of N_LIKES.
        viewed = random.sample(item_ids, min(N_VIEWS, len(item_ids)))
        liked = random.sample(viewed, min(N_LIKES, len(viewed)))

        for i, item_id in enumerate(viewed):
            feedback.append({
                "FeedbackType": "view",
                "UserId": user,
                "ItemId": item_id,
                "Timestamp": (now - timedelta(days=i + 1)).isoformat() + "Z",
            })
        for item_id in liked:
            feedback.append({
                "FeedbackType": "favorite",
                "UserId": user,
                "ItemId": item_id,
                "Timestamp": (now - timedelta(days=1)).isoformat() + "Z",
            })

    resp = httpx.post(f"{GORSE_URL}/api/feedback", json=feedback, timeout=30)
    resp.raise_for_status()
    print(f"  Inserted {len(feedback)} feedback events across {len(DEMO_USERS)} users.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print(f"Loading top {N_PRODUCTS} products from Myntra dataset...\n")
    products = load_products()

    print(f"{'─'*70}")
    print(f"{'ID':<12} {'Category':<15} {'Occasion':<15} {'PriceRange':<12} {'Ratings':>8}  Name")
    print(f"{'─'*70}")
    for p in products:
        item = p["gorse_item"]
        occasion = next((l.split(":")[1] for l in item["Labels"] if l.startswith("occasion:")), "—")
        price_range = next((l.split(":")[1] for l in item["Labels"] if l.startswith("price_range:")), "—")
        category = item["Categories"][0] if item["Categories"] else "—"
        print(f"{item['ItemId']:<12} {category:<15} {occasion:<15} {price_range:<12} {p['_rating_count']:>8}  {p['_name'][:45]}")
    print(f"{'─'*70}\n")

    print("Copying images...")
    copied = copy_images(products)
    print(f"  Copied {copied}/{len(products)} images → {IMAGES_DEST}\n")

    print(f"Posting {len(products)} items to Gorse ({GORSE_URL})...")
    post_items(products)
    print("  Done.\n")

    print("Seeding feedback for demo users...")
    post_feedback(products)
    print()

    print("✓ Seed complete.")
    print(f"  Gorse dashboard: {GORSE_URL}")
    print(f"  Frontend:        http://localhost:5173")


if __name__ == "__main__":
    run()
