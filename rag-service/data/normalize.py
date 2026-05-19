# Cleans and normalizes raw product and review records before ingestion.
# Strips HTML, normalizes prices and categorical fields, and deduplicates reviews
# by token overlap within the same product.

from __future__ import annotations


def normalize_product(raw: dict) -> dict:
    # Strip HTML from descriptions, normalize price to float,
    # lowercase and strip category/brand/occasion fields.
    raise NotImplementedError


def normalize_review(raw: dict) -> dict | None:
    # Strip HTML, drop if duplicate (>30% token overlap with existing review
    # for same product_id). Returns None for dropped reviews.
    raise NotImplementedError
