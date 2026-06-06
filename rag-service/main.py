# FastAPI application entry point for the RAG service.
# Manages startup lifecycle: loads the cross-encoder reranker, builds the BM25
# index from PostgreSQL, and connects to the Milvus collection — all before
# the first request is served.
#
# Startup dependency order (matches CLAUDE.md):
#   1. PostgreSQL  — BM25 index build; hard failure (tenacity retries in retrieval.py)
#   2. Milvus      — vector search; logs CRITICAL if unavailable, starts degraded
#   3. Redis       — embedding cache; optional, silently disabled if unreachable
#   4. OpenAI API  — needed per-request, not at startup
#
# app.state attributes set here and consumed by api/routes.py:
#   .bm25_index        (BM25Okapi, list[str]) | (empty index, [])
#   .milvus_client     MilvusClient | None
#   .milvus_collection str
#   .reranker          Reranker

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from pymilvus import MilvusClient

from api.routes import router
from pipeline.reranker import Reranker
from pipeline.retrieval import build_bm25_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def _connect_milvus() -> MilvusClient | None:
    """Create a MilvusClient, load the collection, and return it.

    Uses the new MilvusClient API (pymilvus ≥ 2.4) instead of the deprecated
    ORM-style connections.connect() + Collection() pattern.

    Returns None — instead of raising — so the service can start in a degraded
    state. The /health endpoint surfaces the problem; hybrid_search() already
    handles milvus_client=None gracefully by returning [].
    """
    host            = os.environ.get("MILVUS_HOST", "localhost")
    port            = int(os.environ.get("MILVUS_PORT", 19530))
    collection_name = os.environ.get("MILVUS_COLLECTION", "fashion_rag")
    uri             = f"http://{host}:{port}"

    try:
        client = MilvusClient(uri=uri)
        # load_collection() moves the collection's HNSW index into QueryNode
        # memory. It is idempotent — safe to call even if already loaded.
        client.load_collection(collection_name)
        logger.info(
            "Milvus collection '%s' loaded from %s", collection_name, uri
        )
        return client

    except Exception as exc:
        logger.critical(
            "Cannot connect to Milvus at %s — vector search will be unavailable. "
            "Restart the service once Milvus is healthy. Error: %s",
            uri, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before serving; teardown cleanly on shutdown."""

    # ── 1. Cross-encoder reranker ──────────────────────────────────────────
    # Loads ms-marco-MiniLM-L-6-v2 weights from disk (~90MB, ~2s).
    # Must happen once at startup — not thread-safe per-request.
    logger.info("Loading cross-encoder reranker...")
    app.state.reranker = Reranker()
    logger.info("Cross-encoder reranker ready")

    # ── 2. BM25 index (PostgreSQL) ─────────────────────────────────────────
    # _build_bm25_sync() retries up to 3 times with exponential backoff.
    # If Postgres is genuinely unreachable after retries, psycopg2.OperationalError
    # is reraised and uvicorn aborts startup — correct, nothing works without BM25.
    logger.info("Building BM25 index from PostgreSQL (rag_products)...")
    app.state.bm25_index = await build_bm25_index()
    _, corpus_ids = app.state.bm25_index
    logger.info("BM25 index ready — %d products indexed", len(corpus_ids))

    # ── 3. Milvus collection ───────────────────────────────────────────────
    # Returns None on failure — service starts in degraded mode.
    logger.info("Connecting to Milvus...")
    app.state.milvus_collection = os.environ.get("MILVUS_COLLECTION", "fashion_rag")
    app.state.milvus_client = _connect_milvus()
    if app.state.milvus_client is None:
        logger.critical(
            "rag-service starting DEGRADED — Milvus unavailable. "
            "All /query requests will return empty results until Milvus is restored."
        )

    logger.info("rag-service ready on port 8002")
    yield  # ← serve requests

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("rag-service shutting down...")
    if app.state.milvus_client is not None:
        try:
            app.state.milvus_client.close()
            logger.info("Milvus client closed")
        except Exception as exc:
            logger.warning("Error closing Milvus client: %s", exc)
    logger.info("rag-service shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fashion RAG Service",
    description="Natural language product search with grounded recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
