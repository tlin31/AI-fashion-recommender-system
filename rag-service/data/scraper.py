# Lightweight review scraper to augment the Kaggle catalog with ~500-1000 real reviews.
# Not load-bearing — the pipeline works on product descriptions alone if scraping fails.

from __future__ import annotations

# Lightweight review scraper targeting ~500-1000 reviews to augment Kaggle catalog.
# Not load-bearing — product description RAG works without reviews.
# Run manually: python data/scraper.py


def scrape_reviews(product_ids: list[str]) -> list[dict]:
    raise NotImplementedError


if __name__ == "__main__":
    scrape_reviews([])
