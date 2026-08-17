# Defines the three HTTP endpoints: POST /query (RAG pipeline), POST /ingest (async ingestion),
# and GET /health (readiness probe). Each handler delegates immediately to pipeline or ingestion
# modules — no business logic lives here.

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import psycopg2
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyBreakdown,
    ProductResult,
    QueryRequest,
    QueryResponse,
)
from pipeline.crag import run_crag
from pipeline.generator import generate
from pipeline.guardrail import is_fashion_query
from pipeline.retrieval import hybrid_search
from pipeline.utils import Timer, _get_redis, embed

logger = logging.getLogger(__name__)

# Returned in place of a generated answer when the LLM is unavailable. Deterministic
# and honest: it does not pretend to be a synthesised recommendation, and it tells the
# caller the products below are still ranked results rather than a fallback list.
_GENERATION_FALLBACK_ANSWER = (
    "I couldn't generate a written summary just now, but these products are the "
    "closest matches for your search."
)

router = APIRouter()

_POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://gorse:gorse_pass@localhost:5432/gorse",
)

_KAFKA_INGEST_TOPIC = os.environ.get("KAFKA_INGEST_TOPIC", "rag-ingest")


# ---------------------------------------------------------------------------
# Postgres helper — product detail lookup for ProductResult construction
# ---------------------------------------------------------------------------

def _fetch_product_details_sync(product_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch name and price for a list of product IDs.

    Called via run_in_executor — psycopg2 is synchronous and must not run
    on the async event loop directly.
    Returns a dict keyed by product_id; missing products get safe defaults.
    """
    conn = psycopg2.connect(_POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT product_id, name, price FROM rag_products WHERE product_id = ANY(%s)",
                (product_ids,),
            )
            return {
                row[0]: {"name": row[1] or "", "price": float(row[2] or 0.0)}
                for row in cur.fetchall()
            }
    finally:
        conn.close()


async def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch_product_details_sync, product_ids)
    except Exception as exc:
        logger.error("Product detail lookup failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# POST /query — full RAG pipeline
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    total_start = Timer()
    total_start.__enter__()
    degraded: list[str] = []

    # ── 1+2. Guardrail and retrieval, concurrently ────────────────────────
    # The guardrail is a GPT-4o-mini round-trip and was the second-largest stage
    # in the latency breakdown, sitting in front of retrieval purely because it
    # can reject. But its verdict is independent of retrieval, so the two overlap
    # and the guardrail's wall-clock cost disappears behind the retrieval it used
    # to block. An off-topic query now does the retrieval work before being
    # rejected — cheap locally, and it costs one embedding call, which is the
    # deliberate trade for removing a serial LLM hop from every good query.
    #
    # Timers now measure overlapping windows, so guardrail_ms + retrieval_ms
    # exceeds the wall-clock time they jointly consume. total_ms remains truthful.
    with Timer() as t_parallel:
        guard_task = asyncio.create_task(is_fashion_query(body.query))
        retrieval_task = asyncio.create_task(
            hybrid_search(
                query=body.query,
                filters=body.filters,
                top_k=20,
                bm25_index=request.app.state.bm25_index,
                milvus_client=request.app.state.milvus_client,
                collection_name=request.app.state.milvus_collection,
                degradations=degraded,
            )
        )
        is_fashion, candidates = await asyncio.gather(guard_task, retrieval_task)

    if not is_fashion:
        raise HTTPException(status_code=400, detail="Query is not fashion-related.")

    with Timer() as t_embed:
        # Cached by the hybrid_search call above — a Redis hit, not an API call.
        query_embedding = await embed(body.query)

    # Both stages ran inside the same window, so each is reported as that window
    # rather than as a slice of it. Summing the per-stage fields therefore
    # overstates total_ms; total_ms is the number to trust.
    guardrail_ms = t_parallel.ms
    retrieval_ms = t_parallel.ms + t_embed.ms

    # ── 3. CRAG loop ──────────────────────────────────────────────────────
    with Timer() as t_crag:
        crag_chunks, retrieval_path = await run_crag(
            query=body.query,
            candidates=candidates,
            query_embedding=query_embedding,
            bm25_index=request.app.state.bm25_index,
            milvus_client=request.app.state.milvus_client,
            collection_name=request.app.state.milvus_collection,
        )

    # ── 4. Cross-encoder reranker ─────────────────────────────────────────
    with Timer() as t_rerank:
        reranked = request.app.state.reranker.rerank(
            query=body.query,
            candidates=crag_chunks,
            top_k=body.top_k,
        )

    # ── 5. Generator ──────────────────────────────────────────────────────
    # Generation is the only stage whose failure used to fail the whole request,
    # which inverted the value: retrieval had already succeeded, so the ranked
    # products were sitting right there. A product list without prose is strictly
    # more useful than a 5xx, so an LLM outage degrades to exactly that.
    with Timer() as t_gen:
        try:
            answer, cited_sources = await generate(body.query, reranked)
        except Exception as exc:
            logger.error("Generation failed (%s); returning products without prose", exc)
            degraded.append("generation_failed")
            answer = _GENERATION_FALLBACK_ANSWER
            # Every returned product is a "source" here — nothing was cited because
            # nothing was written, but the caller still needs the IDs to render cards.
            cited_sources = list(dict.fromkeys(c["product_id"] for c in reranked))

    # ── 6. Fetch product details for response construction ─────────────────
    product_ids = [c["product_id"] for c in reranked]
    details = await _fetch_product_details(product_ids)

    products = [
        ProductResult(
            product_id=chunk["product_id"],
            name=details.get(chunk["product_id"], {}).get("name", ""),
            category=chunk["metadata"].get("category", ""),
            price=details.get(chunk["product_id"], {}).get("price", 0.0),
            score=round(chunk["score"], 4),
            chunk_text=chunk["text"],
        )
        for chunk in reranked
    ]

    total_start.__exit__(None, None, None)

    return QueryResponse(
        answer=answer,
        products=products,
        cited_sources=cited_sources,
        retrieval_path=retrieval_path,
        degraded=degraded,
        latency_ms=LatencyBreakdown(
            guardrail_ms=round(guardrail_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            crag_ms=round(t_crag.ms, 1),
            rerank_ms=round(t_rerank.ms, 1),
            generation_ms=round(t_gen.ms, 1),
            total_ms=round(total_start.ms, 1),
        ),
    )


# ---------------------------------------------------------------------------
# POST /ingest — async ingestion trigger
# ---------------------------------------------------------------------------

@router.post("/ingest", status_code=202, response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    producer = request.app.state.kafka_producer
    if producer is None:
        raise HTTPException(status_code=503, detail="Kafka unavailable.")

    product_ids = body.product_ids or []
    job_id = str(uuid.uuid4())

    if product_ids:
        # Queue one message per product ID.
        for pid in product_ids:
            await producer.send(
                _KAFKA_INGEST_TOPIC,
                value=json.dumps({"job_id": job_id, "product_id": pid}).encode(),
            )
        queued = len(product_ids)
    else:
        # No IDs specified — trigger a full catalog re-ingest.
        await producer.send(
            _KAFKA_INGEST_TOPIC,
            value=json.dumps({"job_id": job_id, "action": "ingest_all"}).encode(),
        )
        queued = 1

    logger.info("Ingest job %s: %d message(s) queued to %s", job_id, queued, _KAFKA_INGEST_TOPIC)
    return IngestResponse(job_id=job_id, queued=queued)


# ---------------------------------------------------------------------------
# GET /health — dependency readiness probe
# ---------------------------------------------------------------------------

def _check_postgres_sync() -> bool:
    try:
        conn = psycopg2.connect(_POSTGRES_URL)
        conn.close()
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    r = _get_redis()
    if r is None:
        return False
    try:
        return bool(await r.ping())
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    loop = asyncio.get_event_loop()

    # Run Postgres check in thread pool (psycopg2 is synchronous).
    postgres_ok, redis_ok = await asyncio.gather(
        loop.run_in_executor(None, _check_postgres_sync),
        _check_redis(),
    )

    milvus_ok = request.app.state.milvus_client is not None
    all_ok = postgres_ok and milvus_ok and redis_ok

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        milvus=milvus_ok,
        postgres=postgres_ok,
        redis=redis_ok,
    )
