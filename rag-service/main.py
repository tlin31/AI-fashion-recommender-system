# FastAPI application entry point for the RAG service.
# Manages startup lifecycle: loads the cross-encoder reranker and builds the BM25 index
# from PostgreSQL before the first request is served.

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from api.routes import router
from pipeline.reranker import Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load cross-encoder once at startup — expensive, must not happen per-request
    logger.info("Loading cross-encoder reranker...")
    app.state.reranker = Reranker()

    # BM25 index is built inside retrieval.py at import time from Postgres;
    # any startup errors surface here before the first request.
    from pipeline.retrieval import build_bm25_index
    logger.info("Building BM25 index from PostgreSQL...")
    app.state.bm25_index = await build_bm25_index()

    logger.info("rag-service ready on port 8002")
    yield

    logger.info("rag-service shutting down")


app = FastAPI(
    title="Fashion RAG Service",
    description="Natural language product search with grounded recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
