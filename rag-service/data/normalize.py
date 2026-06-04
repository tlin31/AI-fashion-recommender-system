# Loads the McAuley 2023 Amazon Fashion metadata JSONL, cleans and normalizes it,
# and writes the result to the rag_products table in PostgreSQL.
#
# Source: McAuley-Lab/Amazon-Reviews-2023, raw/meta_categories/meta_Amazon_Fashion.jsonl
# Download: huggingface_hub.hf_hub_download (see migration plan Phase 2)
#
# Run: python data/normalize.py [--src <path_to_jsonl>]

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DEFAULT_JSONL = (
    Path(__file__).parent.parent
    / "amazon_data"
    / "raw"
    / "meta_categories"
    / "meta_Amazon_Fashion.jsonl"
)
SUBSAMPLE_SIZE = 5000
CHUNK_SIZE     = 50_000   # read JSONL in chunks to avoid loading 1.4 GB at once

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_products (
    product_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    brand        TEXT,
    category     TEXT,
    colour       TEXT,
    occasion     TEXT,
    material     TEXT,
    price        FLOAT,
    price_range  TEXT,
    avg_rating   FLOAT,
    rating_count INT,
    description  TEXT NOT NULL,
    image_url    TEXT
);
"""

UPSERT_SQL = """
INSERT INTO rag_products
    (product_id, name, brand, category, colour, occasion, material,
     price, price_range, avg_rating, rating_count, description, image_url)
VALUES
    (%(product_id)s, %(name)s, %(brand)s, %(category)s, %(colour)s, %(occasion)s,
     %(material)s, %(price)s, %(price_range)s, %(avg_rating)s, %(rating_count)s,
     %(description)s, %(image_url)s)
ON CONFLICT (product_id) DO UPDATE SET
    name         = EXCLUDED.name,
    description  = EXCLUDED.description,
    occasion     = EXCLUDED.occasion,
    price_range  = EXCLUDED.price_range,
    image_url    = EXCLUDED.image_url;
"""


# ── Category keyword list ─────────────────────────────────────────────────────
# Ordered most-specific first to prevent substring collisions
# (e.g. "sweatshirt" before "shirt", "kurta set" before "kurta").
# Covers both Indian fashion (legacy) and Western/global Amazon fashion.

_CATEGORY_KEYWORDS = [
    # Indian fashion (most specific first)
    "kurta set", "co-ord", "lehenga", "palazzo", "churidar",
    "dupatta", "kurti", "kurta", "saree", "salwar",

    # Multi-word Western (before single words they contain)
    "sweatshirt", "jumpsuit", "bodysuit", "turtleneck", "cardigan",
    "pullover", "leggings", "trousers", "joggers", "cargos",
    "board short", "swim trunk", "crop top", "t-shirt",
    "track suit", "tracksuit", "windbreaker", "raincoat", "overcoat",

    # Tops / outerwear
    "blazer", "sweater", "hoodie", "jacket", "coat", "vest",
    "blouse", "romper", "tunic", "shrug", "parka", "puffer",
    "tank", "polo", "tee",

    # Bottoms / full-length
    "dress", "skirt", "jeans", "pants", "shorts", "shirt", "top", "suit",

    # Activewear / sportswear
    "compression", "activewear", "sportswear", "athletic", "fleece",
    "thermal", "rash guard",

    # Swimwear / intimates
    "swimsuit", "swimwear", "bikini", "tankini", "one-piece",
    "bra", "underwear", "panty", "panties", "lingerie",
    "pajama", "pyjama", "nightgown", "sleepwear", "robe",

    # Footwear
    "sneaker", "boot", "sandal", "loafer", "slipper",
    "moccasin", "oxford", "heel", "pump", "flat", "shoe",

    # Accessories
    "scarf", "belt", "glove", "mittens", "beanie", "hat", "cap",
    "headband", "sock", "stocking", "tights",
    "necklace", "bracelet", "earring", "anklet",
    "handbag", "purse", "clutch", "tote",

    # Dress silhouettes (catch maxi/midi/mini before "dress")
    "maxi", "midi", "mini", "wrap", "kimono", "caftan",

    # Maternity / plus
    "maternity",
]


def _derive_category(title: str) -> str:
    """Keyword scan of product title — most specific match wins.

    Returns None if the title contains no fashion keyword at all.
    Callers should drop None rows to exclude non-fashion products
    (watches, pencil cases, tech accessories) that slipped into the
    Amazon Fashion category.
    """
    title_lower = title.lower()
    for kw in _CATEGORY_KEYWORDS:
        if kw in title_lower:
            return kw
    return None   # not a fashion item — caller will drop this row


# ── Field extraction helpers ──────────────────────────────────────────────────

def _parse_details(raw) -> dict:
    """details arrives as a dict (already parsed from JSON) or occasionally a string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            result = ast.literal_eval(raw)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_price(raw) -> float | None:
    """price can be None, 'None', '$29.99', or a float."""
    if raw is None:
        return None
    s = str(raw).strip().lstrip("$").replace(",", "")
    if s.lower() in ("none", ""):
        return None
    try:
        val = float(s)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


# Matches trailing "(Color, Size)" or "(Color)" variant suffixes Amazon appends to
# child-ASIN titles (e.g. "Women's Dress(Navy Blue, XL)").  We strip these before
# embedding so the vector represents the product, not one specific variant.
_VARIANT_SUFFIX = re.compile(
    r"\s*\([^)]*\b(?:XS|S|M|L|XL|XXL|2XL|3XL|4XL|5XL"
    r"|black|white|red|blue|green|pink|gr[ae]y|navy|beige|brown"
    r"|purple|yellow|orange|ivory|cream|burgundy|olive|teal|coral|rose"
    r"|\d+W|\d+L)\b[^)]*\)\s*$",
    re.IGNORECASE,
)


def _clean_title(title: str) -> str:
    """Strip trailing variant suffix from title (color/size combos)."""
    return _VARIANT_SUFFIX.sub("", title).strip()


def _build_description(row: pd.Series) -> str:
    """
    Concatenate title + features + description + key details fields.
    Amazon listings often have empty features/description — the title + details
    is the fallback. Order: title first (most reliable), then features (bullet
    points), then description paragraphs, then structured detail fields.
    """
    parts: list[str] = []

    title = _clean_title(str(row.get("title") or "").strip())
    if title:
        parts.append(title)

    features = row.get("features") or []
    if isinstance(features, list):
        parts.extend(f.strip() for f in features if isinstance(f, str) and f.strip())

    description = row.get("description") or []
    if isinstance(description, list):
        parts.extend(d.strip() for d in description if isinstance(d, str) and d.strip())

    # Pull useful structured fields from details into the description text
    details = _parse_details(row.get("details"))
    for key in ["Style", "Fit Type", "Closure", "Neckline", "Sleeve Length",
                "Pattern", "Occasion", "Material", "Fabric Type"]:
        val = details.get(key)
        if val and str(val).strip():
            parts.append(f"{key}: {str(val).strip()}")

    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _extract_occasion(details: dict) -> str:
    for key in ["Occasion", "Special Occasion", "Department"]:
        val = details.get(key)
        if val:
            return str(val).lower().strip()
    return ""


def _extract_material(details: dict) -> str:
    for key in ["Material", "Fabric Type", "Fabric", "Outer Material", "Shell Material"]:
        val = details.get(key)
        if val:
            return str(val).lower().strip()[:64]   # cap — some values are very long
    return ""


def _extract_colour(details: dict) -> str:
    for key in ["Color", "Colour", "Color Name"]:
        val = details.get(key)
        if val:
            return str(val).lower().strip()
    return ""


def _extract_image_url(images) -> str:
    """Return the best available image URL from the images list."""
    if not images or not isinstance(images, list):
        return ""
    first = images[0] if isinstance(images[0], dict) else {}
    return first.get("large") or first.get("hi_res") or first.get("thumb") or ""


def bucket_price(price: float, p33: float, p67: float) -> str:
    if price < p33:
        return "budget"
    if price < p67:
        return "mid"
    return "premium"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _apply_with_progress(df: pd.DataFrame, func, label: str, batch: int = 50_000) -> pd.Series:
    """Run df.apply(func) in batches, printing progress every `batch` rows."""
    total = len(df)
    parts = []
    t0 = time.time()
    for start in range(0, total, batch):
        end = min(start + batch, total)
        parts.append(df.iloc[start:end].apply(func, axis=1))
        elapsed = time.time() - t0
        rate = (end) / elapsed if elapsed > 0 else 0
        eta = (total - end) / rate if rate > 0 else 0
        print(f"  [{_ts()}] {label}: {end:,}/{total:,} rows  ({rate:,.0f} rows/s  ETA {eta:.0f}s)")
    return pd.concat(parts)


def clean(src: Path) -> pd.DataFrame:
    print(f"[{_ts()}] Reading {src.name}  (chunked, {CHUNK_SIZE:,} rows/chunk)...")

    chunks: list[pd.DataFrame] = []
    total_read = 0

    for chunk in pd.read_json(src, lines=True, chunksize=CHUNK_SIZE):
        total_read += len(chunk)

        # Early filter: need at minimum a parent_asin and a non-trivial title
        chunk = chunk.dropna(subset=["parent_asin", "title"])
        chunk = chunk[chunk["title"].astype(str).str.len() > 3]

        # Keep only products with at least one rating (signal quality filter)
        chunk["rating_number"] = pd.to_numeric(
            chunk.get("rating_number", pd.Series(dtype=float)), errors="coerce"
        )
        chunk = chunk[chunk["rating_number"].fillna(0) > 0]
        chunks.append(chunk)

        kept = sum(len(c) for c in chunks)
        print(f"  [{_ts()}] Read {total_read:,} rows → kept {kept:,} after early filter...")

    df = pd.concat(chunks, ignore_index=True)
    print(f"[{_ts()}] Total after early filter: {len(df):,} products")

    # Build derived fields — slow apply, prints progress every 50k rows
    print(f"[{_ts()}] Building descriptions ({len(df):,} rows) ...")
    df["description"] = _apply_with_progress(df, _build_description, "description")
    df = df[df["description"].str.len() > 30]
    print(f"[{_ts()}] Descriptions done — {len(df):,} rows with len>30")

    print(f"[{_ts()}] Extracting details metadata ({len(df):,} rows) ...")
    details_col       = _apply_with_progress(df, lambda r: _parse_details(r.get("details")), "details")
    df["occasion"]    = details_col.map(_extract_occasion)
    df["material"]    = details_col.map(_extract_material)
    df["colour"]      = details_col.map(_extract_colour)
    df["category"]    = df["title"].astype(str).map(_derive_category)
    print(f"[{_ts()}] Metadata done")

    # Drop non-fashion items (watches, pencil cases, tech accessories, etc.)
    # that slipped through Amazon's loose "AMAZON FASHION" category.
    before = len(df)
    df = df[df["category"].notna()]
    print(f"[{_ts()}] Dropped {before - len(df):,} non-fashion items (no keyword match) → {len(df):,} remain")

    df["image_url"]   = df.get("images", pd.Series([[] for _ in range(len(df))])).map(_extract_image_url)

    # Normalise scalar fields
    print(f"[{_ts()}] Normalising scalar fields ...")
    df["product_id"]   = df["parent_asin"].astype(str).str.strip()
    df["name"]         = df["title"].astype(str).map(_clean_title)
    df["brand"]        = df.get("store", pd.Series([""] * len(df))).fillna("").astype(str).str.lower().str.strip()
    df["price"]        = df.get("price", pd.Series([None] * len(df))).map(_parse_price)
    df["avg_rating"]   = pd.to_numeric(df.get("average_rating"), errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_number"], errors="coerce").astype("Int64")

    # Price buckets — relative to this catalog's own distribution
    valid_prices = df["price"].dropna()
    if len(valid_prices) > 0:
        p33 = valid_prices.quantile(0.33)
        p67 = valid_prices.quantile(0.67)
        df["price_range"] = df["price"].map(
            lambda p: bucket_price(p, p33, p67) if pd.notna(p) else "mid"
        )
    else:
        df["price_range"] = "mid"
    print(f"[{_ts()}] Scalar fields done")

    # ── Quality filters ──────────────────────────────────────────────────────────

    # 1. Richness: title must be descriptive AND have real feature/description text.
    #    Products that only have a title produce near-useless embeddings for RAG.
    print(f"[{_ts()}] Computing richness signals ({len(df):,} rows) ...")
    raw_desc_len = _apply_with_progress(
        df, lambda r: len(" ".join(r.get("description") or [])), "raw_desc_len"
    )
    feature_count = _apply_with_progress(
        df,
        lambda r: len(r.get("features") or []) if isinstance(r.get("features"), list) else 0,
        "feature_count",
    )
    before = len(df)
    df = df[
        (df["title"].astype(str).str.len() >= 60) &
        ((feature_count >= 2) | (raw_desc_len >= 100))
    ]
    print(f"[{_ts()}] Richness filter (title≥60 + feat≥2 or desc≥100): dropped {before - len(df):,} → {len(df):,} remain")

    # 2. Minimum review signal: at least 5 ratings to avoid long-tail noise.
    before = len(df)
    df = df[df["rating_count"] >= 5]
    print(f"[{_ts()}] Min reviews filter (≥5): dropped {before - len(df):,} → {len(df):,} remain")

    # 3. Price sanity: drop obvious non-fashion outliers (>$500).
    #    NULL price is fine — majority of records have no price.
    before = len(df)
    df = df[df["price"].isna() | (df["price"] <= 500)]
    print(f"[{_ts()}] Price filter (≤$500 or null): dropped {before - len(df):,} → {len(df):,} remain")

    # 4. Duplicate product dedup: two strategies combined —
    #
    #    a) parent_asin dedup: child ASINs sharing the same parent_asin (true variants).
    #       Keep the highest-rated row per parent_asin.
    #
    #    b) title+brand dedup: separate parent ASINs that are the same physical product
    #       listed multiple times (e.g. MOSHENGQI leggings with 12 near-identical entries,
    #       each with a different parent_asin but identical cleaned title and brand).
    #       Keep the highest-rated row per (cleaned_name, brand) pair.
    #
    #    Sort descending first so drop_duplicates keeps the first (= highest-rated) row.
    before = len(df)
    df = df.sort_values("rating_count", ascending=False)
    df = df.drop_duplicates(subset=["product_id"])           # (a) true child variants
    df = df.drop_duplicates(subset=["name", "brand"])        # (b) same title+brand duplicates
    print(f"[{_ts()}] Dedup (parent ASIN + title+brand): dropped {before - len(df):,} duplicates → {len(df):,} unique products remain")

    # Subsample: top SUBSAMPLE_SIZE by rating_count (highest signal quality)
    print(f"[{_ts()}] Selecting top {SUBSAMPLE_SIZE:,} by rating_count ...")
    df = df.dropna(subset=["rating_count"])
    df = df.nlargest(SUBSAMPLE_SIZE, "rating_count")

    print(f"\n[{_ts()}] === Subsample stats ({len(df):,} products) ===")
    print(f"  Top categories : {df['category'].value_counts().head(8).to_dict()}")
    print(f"  Price buckets  : {df['price_range'].value_counts().to_dict()}")
    print(f"  With image_url : {(df['image_url'] != '').sum()}")
    print(f"  With occasion  : {(df['occasion'] != '').sum()}")
    print(f"  With material  : {(df['material'] != '').sum()}")
    print(f"  With colour    : {(df['colour'] != '').sum()}")

    return df[["product_id", "name", "brand", "category", "colour", "occasion",
               "material", "price", "price_range", "avg_rating", "rating_count",
               "description", "image_url"]]


def write_to_postgres(df: pd.DataFrame) -> int:
    url = os.environ["POSTGRES_URL"]
    new_ids = list(df["product_id"].astype(str))
    conn = psycopg2.connect(url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)

                # Upsert new/updated rows
                records = df.to_dict("records")
                # Convert pandas NA/NaT to None for psycopg2
                for rec in records:
                    for k, v in rec.items():
                        try:
                            if pd.isna(v):
                                rec[k] = None
                        except (TypeError, ValueError):
                            pass
                cur.executemany(UPSERT_SQL, records)

                # Delete stale rows — products that were in the table from a
                # previous run but are no longer in the current top-N selection
                # (e.g. deduped-out child variants like MOSHENGQI duplicates).
                cur.execute(
                    "DELETE FROM rag_products WHERE product_id != ALL(%s)",
                    (new_ids,)
                )
                deleted = cur.rowcount

        ts = time.strftime('%H:%M:%S')
        print(f"\n[{ts}] Upserted {len(records):,} products into rag_products.")
        if deleted:
            print(f"[{ts}] Deleted {deleted:,} stale rows no longer in top-{len(records):,} selection.")
        return len(records)
    finally:
        conn.close()


def run(src: Path) -> None:
    df = clean(src)
    write_to_postgres(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=Path, default=DEFAULT_JSONL,
        help="Path to meta_Amazon_Fashion.jsonl"
    )
    args = parser.parse_args()
    run(args.src)
