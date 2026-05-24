# Defines the three HTTP endpoints: POST /query (RAG pipeline), POST /ingest (async ingestion),
# and GET /health (readiness probe). Each handler delegates immediately to pipeline or ingestion
# modules — no business logic lives here.

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    raise NotImplementedError


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    raise NotImplementedError


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    raise NotImplementedError
