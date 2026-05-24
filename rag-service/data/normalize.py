# Loads the raw Myntra CSV, cleans and normalizes it, and writes the result to
# the rag_products table in PostgreSQL. This is the single source of truth that
# feeds both the BM25 index (at service startup) and the Milvus ingestion pipeline.
# Run manually: python data/normalize.py [--src <path_to_csv>]

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CSV = Path(__file__).parent.parent.parent / "Myntra Fashion Product Dataset" / "Fashion Dataset.csv"
SUBSAMPLE_SIZE = 5000

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_products (
    product_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    brand       TEXT,
    category    TEXT,
    colour      TEXT,
    occasion    TEXT,
    material    TEXT,
    price       FLOAT,
    price_range TEXT,
    avg_rating  FLOAT,
    rating_count INT,
    description TEXT NOT NULL
);
"""

UPSERT_SQL = """
INSERT INTO rag_products
    (product_id, name, brand, category, colour, occasion, material,
     price, price_range, avg_rating, rating_count, description)
VALUES
    (%(product_id)s, %(name)s, %(brand)s, %(category)s, %(colour)s, %(occasion)s,
     %(material)s, %(price)s, %(price_range)s, %(avg_rating)s, %(rating_count)s,
     %(description)s)
ON CONFLICT (product_id) DO UPDATE SET
    name         = EXCLUDED.name,
    description  = EXCLUDED.description,
    occasion     = EXCLUDED.occasion,
    price_range  = EXCLUDED.price_range;
"""


# ── Cleaning helpers ──────────────────────────────────────────────────────────

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


def bucket_price(price: float, p33: float, p67: float) -> str:
    if price < p33:
        return "budget"
    if price < p67:
        return "mid"
    return "premium"


# Ordered from most specific to least so longer terms match before their substrings
# (e.g. "kurta set" before "kurta", "sweatshirt" before "shirt")
_CATEGORY_KEYWORDS = [
    "kurta set", "co-ord", "jumpsuit", "sweatshirt", "lehenga",
    "palazzo", "leggings", "trousers", "cardigan", "pullover",
    "blazer", "sweater", "hoodie", "jacket", "churidar", "joggers",
    "cargos", "shorts", "dupatta", "tunic", "shrug", "kurta", "kurti",
    "saree", "dress", "skirt", "jeans", "top", "shirt", "suit",
]


def _derive_category(attrs: dict, name: str) -> str:
    """Use Top Type from p_attributes; fall back to keyword scan of product name."""
    top_type = attrs.get("Top Type", "").lower().strip()
    if top_type:
        return top_type
    name_lower = name.lower()
    for kw in _CATEGORY_KEYWORDS:
        if kw in name_lower:
            return kw
    return "apparel"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop rows missing the fields we can't recover
    df = df.dropna(subset=["p_id", "name", "description"])

    # 2. Strip HTML from descriptions — the raw field has <ul>, <li>, <br>, etc.
    df["description"] = df["description"].astype(str).map(strip_html)

    # 3. Drop descriptions that became empty after stripping
    df = df[df["description"].str.len() > 30]

    # 4. Parse p_attributes JSON and extract the fields we need as metadata
    attrs = df["p_attributes"].astype(str).map(parse_attrs)
    df["occasion"] = attrs.map(lambda a: a.get("Occasion", "")).str.lower().str.strip()
    df["material"] = attrs.map(lambda a: a.get("Top Fabric", "")).str.lower().str.strip()
    df["category"] = [
        _derive_category(a, n)
        for a, n in zip(attrs, df["name"].astype(str))
    ]

    # 5. Normalise scalar fields
    df["brand"] = df["brand"].astype(str).str.lower().str.strip()
    df["colour"] = df["colour"].astype(str).str.lower().str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["ratingCount"], errors="coerce").astype("Int64")

    # 6. Bucket price using the dataset's own distribution so thresholds are
    #    relative to this catalog, not hardcoded USD/INR amounts.
    p33 = df["price"].quantile(0.33)
    p67 = df["price"].quantile(0.67)
    df["price_range"] = df["price"].map(lambda p: bucket_price(p, p33, p67) if pd.notna(p) else "mid")

    # 7. Subsample: keep the top SUBSAMPLE_SIZE by rating_count so we retain
    #    the most-reviewed (highest signal quality) products.
    df = df.dropna(subset=["rating_count"])
    df = df.nlargest(SUBSAMPLE_SIZE, "rating_count")

    df["product_id"] = df["p_id"].astype(int).astype(str)

    return df


def write_to_postgres(df: pd.DataFrame) -> int:
    url = os.environ["POSTGRES_URL"]
    conn = psycopg2.connect(url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                records = df[["product_id", "name", "brand", "category", "colour",
                              "occasion", "material", "price", "price_range",
                              "avg_rating", "rating_count", "description"]].to_dict("records")
                cur.executemany(UPSERT_SQL, records)
        print(f"Upserted {len(records)} products into rag_products.")
        return len(records)
    finally:
        conn.close()


def run(src: Path) -> None:
    print(f"Loading {src}...")
    df = pd.read_csv(src, low_memory=False)
    print(f"  Raw rows: {len(df)}")

    df = clean(df)
    print(f"  After cleaning and subsample: {len(df)} products")
    print(f"  Price buckets: {df['price_range'].value_counts().to_dict()}")
    print(f"  Top occasions: {df['occasion'].value_counts().head(5).to_dict()}")

    write_to_postgres(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_CSV, help="Path to raw CSV")
    args = parser.parse_args()
    run(args.src)
