# Pydantic request and response models for all API endpoints.
# QueryResponse includes per-component latency_ms so callers can instrument
# retrieval, reranking, and generation stages independently.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language product query")
    filters: dict | None = Field(None, description="Structured pre-filters, e.g. {'price_max': 80, 'category': 'trousers'}")
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
    retrieval_path: Literal["synthesize", "retry", "fallback"]
    latency_ms: LatencyBreakdown


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
