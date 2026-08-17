# Pydantic request and response models for all API endpoints.
# QueryResponse includes per-component latency_ms so callers can instrument
# retrieval, reranking, and generation stages independently.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # A required string field. The ... (called an Ellipsis) tells Python:
    # "This field is mandatory and has no default value."
    query: str = Field(..., description="Natural language product query")

    # A dictionary (map) that is optional. The dict | None syntax means
    # it can accept either a dictionary or a None value. It defaults to None.
    filters: dict | None = Field(None, description="Structured pre-filters, e.g. {'price_max': 80, 'category': 'trousers'}")

    # An integer field that defaults to 5 if omitted. It enforces strict mathematical limits.
    # 1<= k <=20
    top_k: int = Field(5, ge=1, le=20, description="Number of products to return")


class ProductResult(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    score: float = Field(..., description="Final reranker score")
    chunk_text: str = Field(..., description="The retrieved chunk that grounded this result")


class LatencyBreakdown(BaseModel):
    guardrail_ms: float
    retrieval_ms: float
    crag_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded natural language answer")
    products: list[ProductResult]
    cited_sources: list[str] = Field(..., description="Product IDs cited in the answer")
    retrieval_path: Literal["synthesize", "retry", "fallback", "best_effort"]
    latency_ms: LatencyBreakdown
    degraded: list[str] = Field(
        default_factory=list,
        description=(
            "Components that failed during this request. Empty on the healthy path. "
            "The response is still usable — degradations are surfaced rather than "
            "raised so a partial outage returns partial results instead of a 5xx. "
            "Values: milvus_unavailable, milvus_search_failed, embedding_failed, "
            "bm25_only, generation_failed."
        ),
    )


class IngestRequest(BaseModel):
    product_ids: list[str] | None = Field(None, description="Specific product IDs to ingest; omit to ingest all")


class IngestResponse(BaseModel):
    job_id: str
    queued: int = Field(..., description="Number of products queued to the Kafka topic")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    milvus: bool
    postgres: bool
    redis: bool
