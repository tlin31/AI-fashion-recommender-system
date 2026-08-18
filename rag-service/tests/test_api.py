# End-to-end tests for api/routes.py.
# All pipeline functions and external services are mocked — no live dependencies.
#
# Cases:
#   POST /query off-topic      → 400
#   POST /query happy path     → 200 with correct response shape
#   GET  /health all healthy   → status "ok", all True
#   GET  /health Milvus down   → status "degraded", milvus False
#   POST /ingest with IDs      → 202, queued == len(ids)
#   POST /ingest no Kafka      → 503

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rank_bm25 import BM25Okapi

from api.routes import router


# ---------------------------------------------------------------------------
# App fixture — router wired to a minimal FastAPI app with mock state
# ---------------------------------------------------------------------------

def _make_app(*, milvus_client=MagicMock(), kafka_producer=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    app.state.bm25_index       = (BM25Okapi([[""]]), [])
    app.state.milvus_client    = milvus_client
    app.state.milvus_collection = "fashion_rag"
    app.state.kafka_producer   = kafka_producer

    # Reranker returns input chunks unchanged (identity mock).
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, candidates, top_k=5: candidates[:top_k]
    app.state.reranker = mock_reranker

    return app


def _make_chunk(product_id: str) -> dict:
    return {
        "chunk_id":    f"chunk_{product_id}",
        "product_id":  product_id,
        "text":        f"Description of {product_id}",
        "score":       0.8,
        "milvus_score": 0.7,
        "metadata":    {"chunk_type": "description", "category": "tops",
                        "price_range": "mid", "occasion": "casual", "brand": "zara"},
    }


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def test_query_off_topic_returns_400():
    """Off-topic queries are still rejected — but only after retrieval and rerank.

    The guardrail now runs concurrently and is awaited just before generation, so
    an off-topic query pays for the cheap deterministic stages before being turned
    away. That is the deliberate trade: no query pays a serial LLM hop, and the
    rejection still happens before the expensive one.
    """
    chunks = [_make_chunk("p001")]
    app = _make_app()
    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query", new_callable=AsyncMock, return_value=False), \
             patch("api.routes.hybrid_search",    new_callable=AsyncMock, return_value=chunks), \
             patch("api.routes.embed",            new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",         new_callable=AsyncMock,
                   return_value=(chunks, "synthesize")), \
             patch("api.routes.generate",         new_callable=AsyncMock) as mock_gen:
            resp = client.post("/query", json={"query": "best JavaScript framework"})

    assert resp.status_code == 400
    assert "fashion" in resp.json()["detail"].lower()
    # The point of gating before generation: no LLM call is made for an off-topic query.
    mock_gen.assert_not_called()


def test_query_happy_path_returns_200_with_correct_shape():
    chunks = [_make_chunk("p001"), _make_chunk("p002")]
    app = _make_app()

    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query",  new_callable=AsyncMock, return_value=True), \
             patch("api.routes.hybrid_search",     new_callable=AsyncMock, return_value=chunks), \
             patch("api.routes.embed",             new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",          new_callable=AsyncMock, return_value=(chunks, "synthesize")), \
             patch("api.routes.generate",          new_callable=AsyncMock,
                   return_value=("Great choice [1]!", ["p001"])), \
             patch("api.routes._fetch_product_details", new_callable=AsyncMock,
                   return_value={"p001": {"name": "Blue Dress", "price": 49.99},
                                 "p002": {"name": "Red Top",   "price": 29.99}}):

            resp = client.post("/query", json={"query": "blue casual dress"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Great choice [1]!"
    assert body["cited_sources"] == ["p001"]
    assert body["retrieval_path"] == "synthesize"
    assert "guardrail_ms" in body["latency_ms"]
    assert len(body["products"]) == 2
    assert body["products"][0]["product_id"] == "p001"
    assert body["products"][0]["name"] == "Blue Dress"


# ---------------------------------------------------------------------------
# POST /query — degradation instead of failure
# ---------------------------------------------------------------------------

def test_generation_failure_returns_products_not_5xx():
    """An LLM outage must not discard retrieval that already succeeded."""
    chunks = [_make_chunk("p001"), _make_chunk("p002")]
    app = _make_app()

    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query",  new_callable=AsyncMock, return_value=True), \
             patch("api.routes.hybrid_search",     new_callable=AsyncMock, return_value=chunks), \
             patch("api.routes.embed",             new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",          new_callable=AsyncMock, return_value=(chunks, "synthesize")), \
             patch("api.routes.generate",          new_callable=AsyncMock,
                   side_effect=Exception("OpenAI 503")), \
             patch("api.routes._fetch_product_details", new_callable=AsyncMock,
                   return_value={"p001": {"name": "Blue Dress", "price": 49.99},
                                 "p002": {"name": "Red Top",   "price": 29.99}}):

            resp = client.post("/query", json={"query": "blue casual dress"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] == ["generation_failed"]
    # Products survive, and every one is offered as a source.
    assert [p["product_id"] for p in body["products"]] == ["p001", "p002"]
    assert body["cited_sources"] == ["p001", "p002"]
    assert body["answer"]        # non-empty, deterministic fallback text


def test_retrieval_degradations_surface_in_response():
    """Degradations recorded inside hybrid_search reach the caller."""
    chunks = [_make_chunk("p001")]
    app = _make_app()

    async def _degrading_search(*args, **kwargs):
        kwargs["degradations"].extend(["milvus_unavailable", "bm25_only"])
        return chunks

    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query",  new_callable=AsyncMock, return_value=True), \
             patch("api.routes.hybrid_search",     side_effect=_degrading_search), \
             patch("api.routes.embed",             new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",          new_callable=AsyncMock, return_value=(chunks, "best_effort")), \
             patch("api.routes.generate",          new_callable=AsyncMock, return_value=("ok", ["p001"])), \
             patch("api.routes._fetch_product_details", new_callable=AsyncMock,
                   return_value={"p001": {"name": "Blue Dress", "price": 49.99}}):

            resp = client.post("/query", json={"query": "blue casual dress"})

    assert resp.status_code == 200
    assert resp.json()["degraded"] == ["milvus_unavailable", "bm25_only"]


def test_healthy_path_reports_no_degradation():
    chunks = [_make_chunk("p001")]
    app = _make_app()

    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query",  new_callable=AsyncMock, return_value=True), \
             patch("api.routes.hybrid_search",     new_callable=AsyncMock, return_value=chunks), \
             patch("api.routes.embed",             new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",          new_callable=AsyncMock, return_value=(chunks, "synthesize")), \
             patch("api.routes.generate",          new_callable=AsyncMock, return_value=("ok [1]", ["p001"])), \
             patch("api.routes._fetch_product_details", new_callable=AsyncMock,
                   return_value={"p001": {"name": "Blue Dress", "price": 49.99}}):

            resp = client.post("/query", json={"query": "blue casual dress"})

    assert resp.json()["degraded"] == []


def test_guardrail_and_retrieval_run_concurrently():
    """Both stages must be in flight together, not serialised."""
    import asyncio

    order: list[str] = []

    async def _slow_guard(query):
        order.append("guard_start")
        await asyncio.sleep(0.05)
        order.append("guard_end")
        return True

    async def _slow_search(*args, **kwargs):
        order.append("search_start")
        await asyncio.sleep(0.05)
        order.append("search_end")
        return [_make_chunk("p001")]

    app = _make_app()
    with TestClient(app) as client:
        with patch("api.routes.is_fashion_query", side_effect=_slow_guard), \
             patch("api.routes.hybrid_search",    side_effect=_slow_search), \
             patch("api.routes.embed",            new_callable=AsyncMock, return_value=[0.0] * 1536), \
             patch("api.routes.run_crag",         new_callable=AsyncMock,
                   return_value=([_make_chunk("p001")], "synthesize")), \
             patch("api.routes.generate",         new_callable=AsyncMock, return_value=("ok", ["p001"])), \
             patch("api.routes._fetch_product_details", new_callable=AsyncMock,
                   return_value={"p001": {"name": "Blue Dress", "price": 49.99}}):

            resp = client.post("/query", json={"query": "blue casual dress"})

    assert resp.status_code == 200
    # Serial execution would give guard_start, guard_end, search_start, search_end.
    # Concurrency means both start before either finishes.
    assert order.index("search_start") < order.index("guard_end")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_all_ok():
    app = _make_app(milvus_client=MagicMock())

    with TestClient(app) as client:
        with patch("api.routes._check_postgres_sync", return_value=True), \
             patch("api.routes._check_redis",         new_callable=AsyncMock, return_value=True):
            resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"]   == "ok"
    assert body["milvus"]   is True
    assert body["postgres"] is True
    assert body["redis"]    is True


def test_health_degraded_when_milvus_is_none():
    app = _make_app(milvus_client=None)

    with TestClient(app) as client:
        with patch("api.routes._check_postgres_sync", return_value=True), \
             patch("api.routes._check_redis",         new_callable=AsyncMock, return_value=True):
            resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"]  == "degraded"
    assert body["milvus"]  is False


def test_health_degraded_when_postgres_down():
    app = _make_app()

    with TestClient(app) as client:
        with patch("api.routes._check_postgres_sync", return_value=False), \
             patch("api.routes._check_redis",         new_callable=AsyncMock, return_value=True):
            resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"]   == "degraded"
    assert resp.json()["postgres"] is False


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

def test_ingest_queues_one_message_per_product_id():
    mock_producer = MagicMock()
    mock_producer.send = AsyncMock()
    app = _make_app(kafka_producer=mock_producer)

    with TestClient(app) as client:
        resp = client.post("/ingest", json={"product_ids": ["p001", "p002", "p003"]})

    assert resp.status_code == 202
    body = resp.json()
    assert body["queued"] == 3
    assert "job_id" in body
    assert mock_producer.send.call_count == 3


def test_ingest_no_kafka_returns_503():
    app = _make_app(kafka_producer=None)

    with TestClient(app) as client:
        resp = client.post("/ingest", json={"product_ids": ["p001"]})

    assert resp.status_code == 503
