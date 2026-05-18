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

from agent.graph import HITLNotPendingError, HITLPendingError


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


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

async def test_chat_raises_409_while_interrupt_pending(agent_graph):
    """Sending a new message while a HITL interrupt is pending must raise
    HITLPendingError (translated to HTTP 409 by the handler)."""
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("I love minimalist style", "user_F", "sess_409_chat")
    # Graph is now paused — second message must be rejected.
    _setup_normal_turn(agent_graph)
    with pytest.raises(HITLPendingError):
        await agent_graph.chat("show me more", "user_F", "sess_409_chat")


async def test_resume_raises_409_when_not_pending(agent_graph):
    """Calling resume() when the graph is not paused must raise
    HITLNotPendingError (translated to HTTP 409 by the handler)."""
    _setup_normal_turn(agent_graph)
    await agent_graph.chat("just a normal message", "user_G", "sess_409_resume")
    # No interrupt was triggered — resume must be rejected.
    with pytest.raises(HITLNotPendingError):
        await agent_graph.resume("sess_409_resume", approved=True)


async def test_resume_raises_409_on_double_call(agent_graph):
    """Calling resume() twice on the same session must raise HITLNotPendingError
    on the second call (the first call already consumed the interrupt)."""
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("I prefer low prices", "user_H", "sess_double_resume")
    await agent_graph.resume("sess_double_resume", approved=False)
    # Second resume on the same (now-completed) session must be rejected.
    with pytest.raises(HITLNotPendingError):
        await agent_graph.resume("sess_double_resume", approved=True)


async def test_write_traits_db_failure_does_not_crash_graph(agent_graph, mock_db):
    """If the DB write in write_traits_node fails, the graph must still reach END
    and pending_trait_updates must be cleared (no stuck checkpoint)."""
    mock_db.save_user_traits.side_effect = RuntimeError("DB connection lost")
    _setup_hitl_turn(agent_graph)
    await agent_graph.chat("I like vintage style", "user_I", "sess_db_fail")

    # Should not raise — the node catches and logs the error.
    await agent_graph.resume("sess_db_fail", approved=True)

    # Graph must have advanced to END — a second resume must now raise 409.
    with pytest.raises(HITLNotPendingError):
        await agent_graph.resume("sess_db_fail", approved=True)


async def test_score_clamping_in_merge(agent_graph, mock_db):
    """Scores outside [0.0, 1.0] must be clamped after merging."""
    # Stage an update with an out-of-range score.
    _setup_hitl_turn(agent_graph, style={"minimalist": 1.8, "casual": -0.3})
    await agent_graph.chat("I really love minimalist", "user_J", "sess_clamp")
    await agent_graph.resume("sess_clamp", approved=True)

    _, saved_traits, _ = mock_db.save_user_traits.call_args[0]
    assert saved_traits["style_preferences"]["minimalist"] == pytest.approx(1.0)
    assert saved_traits["style_preferences"]["casual"] == pytest.approx(0.0)
