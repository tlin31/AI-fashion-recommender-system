# Hybrid retrieval combining BM25 sparse search and Milvus HNSW dense search.
# Results from both are fused with Reciprocal Rank Fusion (RRF, k=60) into a
# single ranked candidate list. Metadata pre-filters are applied inside Milvus
# before ANN search, not as a post-filter.

from __future__ import annotations


async def build_bm25_index():
    # Called once at startup. Loads all product titles + descriptions from
    # Postgres and builds an in-memory BM25 index. ~5K products = negligible RAM.
    raise NotImplementedError


async def hybrid_search(query: str, filters: dict | None, top_k: int = 20) -> list[dict]:
    # 1. Embed query with text-embedding-3-small
    # 2. Run BM25 over titles/descriptions
    # 3. Run Milvus HNSW with metadata pre-filter (applied before ANN, not post)
    # 4. Fuse both ranked lists with RRF (k=60)
    # Returns top_k candidates with chunk text and product metadata.
    raise NotImplementedError
