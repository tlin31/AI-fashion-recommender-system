"""Shared fixtures for all HITL tests.

Fixtures here are automatically available to every test file in this package
without an explicit import, courtesy of pytest's conftest discovery.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import AgentConfig, AgentGraph
from db.client import DBClient
from db.gorse_client import GorseClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
# ---------------------------------------------------------------------------
# Environment — set fake keys before any code that reads env vars runs
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Ensure API-key reads never hit real services during tests."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-fakekey-00000000")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key-for-tests")


# ---------------------------------------------------------------------------
# DB / Gorse mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    db = MagicMock(spec=DBClient)
    # Default: user has no existing traits (new user path in write_traits_node).
    db.get_user_traits = AsyncMock(return_value=None)
    db.save_user_traits = AsyncMock(return_value=None)
    return db


@pytest.fixture
def mock_gorse():
    """GorseClient mock — individual methods are configured per test if needed."""
    return MagicMock(spec=GorseClient)


# ---------------------------------------------------------------------------
# AgentGraph with MemorySaver (no Postgres, no real LLMs)
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_graph(mock_db, mock_gorse):
    """Real AgentGraph wired to an in-memory checkpointer.

    ChatGoogleGenerativeAI is patched so __init__ never touches the network.
    After construction, _router_model and _final_model are replaced with
    AsyncMocks whose return values are configured per test.

    GorseSync.sync_user_traits is also replaced with an AsyncMock so
    write_traits_node does not attempt a real Gorse HTTP call.
    """
    with patch("agent.graph.ChatGoogleGenerativeAI"):
        g = AgentGraph(
            config=AgentConfig(api_key="fake-key"),
            db=mock_db,
            gorse=mock_gorse,
            checkpointer=MemorySaver(),
        )

    # Replace model handles — router_node / finalizer_node access them via
    # `self` at call time, so swapping the attributes is enough; no rebuild.
    g._router_model = AsyncMock()
    g._final_model = AsyncMock()

    # Prevent write_traits_node from making a real Gorse HTTP call.
    g._gorse_sync.sync_user_traits = AsyncMock()

    return g
