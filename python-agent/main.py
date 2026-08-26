"""FastAPI application entry point for the Python agent microservice.

Uses the same 9 environment variable names as the Go server so no .env
changes are needed when running both services side-by-side:

  GOOGLE_API_KEY      – Google AI Studio API key (required)
  GOOGLE_MODEL        – base model for trait extractor (default: agent_final_model)
  AGENT_ROUTER_MODEL  – cheap routing model (default: gemini-2.5-flash)
  AGENT_FINAL_MODEL   – strong answer model (default: gemma-4-31b-it)
  AGENT_MAX_ITERATIONS – ReAct iteration cap (default: 8)
  TAVILY_API_KEY      – Tavily web search API key, backs search_fashion_trends
  DATABASE_URL        – asyncpg connection string
  GORSE_URL           – Gorse HTTP endpoint (default: http://localhost:8088)
  GORSE_API_KEY       – optional Gorse API key

Port: 5002  (Go API: 5001, Gorse: 8088)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import uvicorn
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.graph import AgentConfig, AgentGraph
from agent.metrics import MetricsSink
from api.agent_handler import router as agent_router
from db.client import DBClient
from db.gorse_client import GorseClient
from traits.extractor import TraitExtractor
from traits.gorse_sync import GorseSync


# ---------------------------------------------------------------------------
# Settings — loaded from environment (or .env file if present)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    google_model: str = ""  # falls back to agent_final_model if not set in .env
    # Must be declared here, not just on AgentConfig: extra="ignore" drops
    # unknown env vars instead of storing them, so reading settings.agent_final_model
    # without this field raises AttributeError at startup whenever GOOGLE_MODEL is
    # unset (Docker/CI). Default mirrors AgentConfig.final_model.
    agent_final_model: str = "gemma-4-31b-it"

    mock_ai: bool = False

    # Tavily API key — read automatically by TavilySearchResults via TAVILY_API_KEY env var.
    # Setting it here makes it visible in Settings and allows .env loading.
    tavily_api_key: str = ""

    database_url: str = "postgresql://gorse:gorse_pass@localhost:5432/gorse"
    gorse_url: str = "http://localhost:8088"
    gorse_api_key: str = ""

    # Per-turn latency/token/cost records, one JSON line each. Set to "" to
    # disable. eval/aggregate_metrics.py consumes this file.
    agent_metrics_path: str = "metrics/turns.jsonl"


# ---------------------------------------------------------------------------
# Lifespan: initialise all shared singletons, teardown on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = Settings()

    # ---- PostgreSQL connection pool ----
    pool = await asyncpg.create_pool(settings.database_url)
    db = DBClient(pool)

    # ---- External HTTP clients ----
    gorse = GorseClient(settings.gorse_url, settings.gorse_api_key)

    # ---- LangGraph agent ----
    # TavilySearch reads TAVILY_API_KEY directly from os.environ at construction time.
    # pydantic-settings loads the value but doesn't back-propagate it into os.environ,
    # so we set it explicitly here before AgentGraph (and make_tools) is instantiated.
    if settings.tavily_api_key:
        os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

    # ---- LangGraph checkpoint saver ----
    # from_conn_string() is an async context manager that owns its own psycopg
    # connection pool (separate from the asyncpg pool above).
    # setup() is idempotent — creates langgraph_checkpoints* tables on first run.
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
        await checkpointer.setup()

        agent_cfg = AgentConfig()
        metrics_sink = MetricsSink(settings.agent_metrics_path or None)
        graph = AgentGraph(
            agent_cfg, db, gorse,
            checkpointer=checkpointer,
            metrics_sink=metrics_sink,
        )

        # ---- Mock mode (MOCK_AI=true) — bypasses all real LLM calls ----
        if settings.mock_ai:
            from agent.mocks import install_mock_models
            install_mock_models(graph)

        # ---- Trait extraction LLM (same Google credentials, base model) ----
        if settings.mock_ai:
            from unittest.mock import AsyncMock
            llm = AsyncMock()
            llm.ainvoke = AsyncMock(return_value=type("R", (), {"content": ""})())
        else:
            llm = ChatGoogleGenerativeAI(
                model=settings.google_model or settings.agent_final_model,
                google_api_key=settings.google_api_key,
            )
        extractor = TraitExtractor(llm, db)
        gorse_sync = GorseSync(db, gorse)

        # Store on app.state so handlers can access without global state.
        app.state.db = db
        app.state.graph = graph
        app.state.extractor = extractor
        app.state.gorse_sync = gorse_sync

        yield  # app is running — checkpointer context stays open for the lifetime

    # ---- Teardown (outside async with — checkpointer already closed) ----
    await pool.close()
    await gorse.aclose()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fashion Agent (Python / LangGraph)",
    description="ReAct agent microservice — POST /api/ai/agent-chat",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(agent_router, prefix="/api/ai")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)
