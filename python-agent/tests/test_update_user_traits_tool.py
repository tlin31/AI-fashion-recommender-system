"""Unit tests for the update_user_traits tool.

Strategy: call make_tools(), find the tool by name, then invoke its
underlying coroutine directly (bypassing LangChain's schema validation)
so we can pass InjectedState / InjectedToolCallId params explicitly.

This keeps the tests self-contained — no LangGraph graph, no DB calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent.tools import make_tools
from db.client import DBClient
from db.gorse_client import GorseClient


# ---------------------------------------------------------------------------
# Fixture: extract the update_user_traits coroutine
# ---------------------------------------------------------------------------

@pytest.fixture
def traits_tool_coroutine():
    """Return the raw async coroutine of update_user_traits."""
    mock_db = MagicMock(spec=DBClient)
    mock_gorse = MagicMock(spec=GorseClient)
    tools = make_tools(mock_db, mock_gorse)
    tool = next(t for t in tools if t.name == "update_user_traits")
    # StructuredTool stores the async function as .coroutine
    return tool.coroutine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _invoke(coroutine, *, tool_call_id="test-id-001", pending=None, **kwargs):
    """Call the coroutine with sensible defaults for injected params."""
    return await coroutine(
        style_preferences=kwargs.get("style_preferences"),
        color_preferences=kwargs.get("color_preferences"),
        price_sensitivity=kwargs.get("price_sensitivity"),
        brand_preferences=kwargs.get("brand_preferences"),
        occasions=kwargs.get("occasions"),
        keywords=kwargs.get("keywords"),
        interests=kwargs.get("interests"),
        tool_call_id=tool_call_id,
        pending_trait_updates=pending if pending is not None else [],
    )


def _tool_message(cmd: Command) -> ToolMessage:
    """Extract the ToolMessage from a Command's update dict."""
    msgs = cmd.update.get("messages", [])
    assert msgs, "Command contained no messages"
    return msgs[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_invalid_price_sensitivity_returns_error(traits_tool_coroutine):
    cmd = await _invoke(traits_tool_coroutine, price_sensitivity="medium-high")
    msg = _tool_message(cmd)
    assert isinstance(msg, ToolMessage)
    assert "无效" in msg.content or "low/medium/high" in msg.content
    # Pending list must NOT have grown
    assert "pending_trait_updates" not in cmd.update or cmd.update["pending_trait_updates"] == []


async def test_all_none_fields_returns_no_op(traits_tool_coroutine):
    """Calling the tool with no actual data should produce a no-op message."""
    cmd = await _invoke(traits_tool_coroutine)   # all params default to None
    msg = _tool_message(cmd)
    assert isinstance(msg, ToolMessage)
    assert "没有" in msg.content or "no" in msg.content.lower()
    assert "pending_trait_updates" not in cmd.update or cmd.update["pending_trait_updates"] == []


async def test_valid_partial_update_staged(traits_tool_coroutine):
    cmd = await _invoke(
        traits_tool_coroutine,
        style_preferences={"minimalist": 0.9},
        price_sensitivity="low",
    )
    staged = cmd.update.get("pending_trait_updates", [])
    assert len(staged) == 1
    entry = staged[0]
    assert entry["style_preferences"] == {"minimalist": 0.9}
    assert entry["price_sensitivity"] == "low"
    # Fields that were None must NOT appear in the staged dict
    assert "color_preferences" not in entry
    assert "brand_preferences" not in entry


async def test_appends_to_existing_pending_list(traits_tool_coroutine):
    """If there are already staged updates, the new one is appended."""
    prior = [{"style_preferences": {"casual": 0.5}}]
    cmd = await _invoke(
        traits_tool_coroutine,
        price_sensitivity="medium",
        pending=prior,
    )
    staged = cmd.update.get("pending_trait_updates", [])
    assert len(staged) == 2
    assert staged[0]["style_preferences"]["casual"] == 0.5
    assert staged[1]["price_sensitivity"] == "medium"


async def test_all_seven_fields_staged(traits_tool_coroutine):
    """All 7 TraitsData fields should be present in the staged dict when passed."""
    cmd = await _invoke(
        traits_tool_coroutine,
        style_preferences={"formal": 0.8},
        color_preferences={"black": 0.9},
        price_sensitivity="high",
        brand_preferences=["Gucci"],
        occasions=["party"],
        keywords=["奢华"],
        interests=["艺术"],
    )
    staged = cmd.update.get("pending_trait_updates", [])
    assert len(staged) == 1
    entry = staged[0]
    assert "style_preferences" in entry
    assert "color_preferences" in entry
    assert "price_sensitivity" in entry
    assert "brand_preferences" in entry
    assert "occasions" in entry
    assert "keywords" in entry
    assert "interests" in entry
