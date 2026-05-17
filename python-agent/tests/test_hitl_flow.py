"""Integration tests for the full HITL chat() → approve/reject cycle.

Uses a real AgentGraph wired to MemorySaver (no Postgres).
LLM calls are mocked via AsyncMock on _router_model and _final_model.
DB and Gorse calls are mocked at the fixture level (see conftest.py).

Session IDs are unique per test so each test gets a fresh checkpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Local message factories
# ---------------------------------------------------------------------------

def make_tool_call_msg(
    style_preferences: dict | None = None,
    price_sensitivity: str | None = None,
    call_id: str = "call_test_001",
) -> AIMessage:
    args: dict = {}
    if style_preferences is not None:
        args["style_preferences"] = style_preferences
    if price_sensitivity is not None:
        args["price_sensitivity"] = price_sensitivity
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "update_user_traits",
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def make_plain_msg(text: str = "好的，我了解了。") -> AIMessage:
    return AIMessage(content=text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_normal_turn(g):
    """Router answers directly (no tool call); finalizer writes the answer."""
    g._router_model.ainvoke = AsyncMock(return_value=make_plain_msg("推荐您试试简约风。"))
    g._final_model.ainvoke = AsyncMock(return_value=AIMessage(content="这是我的推荐！"))


def _setup_hitl_turn(g, style=None, price=None):
    """Router calls update_user_traits once, then answers; finalizer polishes."""
    style = style or {"minimalist": 0.9}
    price = price or "low"
    tool_msg = make_tool_call_msg(style_preferences=style, price_sensitivity=price)
    plain_msg = make_plain_msg("好的，我已记录您的偏好。")
    g._router_model.ainvoke = AsyncMock(side_effect=[tool_msg, plain_msg])
    g._final_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="您的简约偏好已暂存，请确认是否保存。")
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_normal_turn_no_interrupt(agent_graph):
    """When update_user_traits is never called, pending_approval must be False."""
    _setup_normal_turn(agent_graph)

    result = await agent_graph.chat("给我推荐一些商品", "user_A", "sess_normal_01")

    assert result.pending_approval is False
    assert result.pending_trait_updates == []
    assert result.answer  # non-empty answer


async def test_hitl_interrupt_detected(agent_graph):
    """When update_user_traits stages updates, chat() must return pending_approval=True."""
    _setup_hitl_turn(agent_graph)

    result = await agent_graph.chat("我喜欢简约风，价格实惠", "user_B", "sess_hitl_01")

    assert result.pending_approval is True
    assert len(result.pending_trait_updates) == 1
    staged = result.pending_trait_updates[0]
    assert staged.get("style_preferences") == {"minimalist": 0.9}
    assert staged.get("price_sensitivity") == "low"


async def test_resume_approved_writes_db(agent_graph, mock_db):
    """Approving the interrupt should call db.save_user_traits exactly once."""
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("我喜欢简约风，价格实惠", "user_C", "sess_approve_01")

    # Graph is now paused at interrupt_before=["write_traits"].
    await agent_graph.resume("sess_approve_01", approved=True)

    mock_db.save_user_traits.assert_called_once()
    saved_user_id, saved_traits, saved_confidence = mock_db.save_user_traits.call_args[0]
    assert saved_user_id == "user_C"
    assert saved_traits["style_preferences"]["minimalist"] == pytest.approx(0.9)
    assert saved_traits["price_sensitivity"] == "low"
    assert saved_confidence > 0.0


async def test_resume_rejected_skips_db(agent_graph, mock_db):
    """Rejecting the interrupt must NOT write to the DB."""
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("我喜欢简约风，价格实惠", "user_D", "sess_reject_01")

    await agent_graph.resume("sess_reject_01", approved=False)

    mock_db.save_user_traits.assert_not_called()


async def test_resume_approved_clears_pending(agent_graph):
    """After approval, the checkpoint's pending_trait_updates must be empty."""
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("我喜欢简约风，价格实惠", "user_E", "sess_clear_01")

    await agent_graph.resume("sess_clear_01", approved=True)

    config = {"configurable": {"thread_id": "sess_clear_01"}}
    snap = await agent_graph._compiled.aget_state(config)
    assert snap.values.get("pending_trait_updates") == []
