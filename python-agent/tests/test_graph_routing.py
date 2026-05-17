"""Unit tests for the should_write_traits routing condition.

should_write_traits is a 3-line nested function inside AgentGraph._build_graph().
Rather than extracting it (which would require changing production code), we
define a local mirror and test the three conditions it checks.

The actual compiled routing behaviour is validated end-to-end in
test_hitl_flow.py::test_hitl_interrupt_detected and
test_hitl_flow.py::test_normal_turn_no_interrupt.
"""

from __future__ import annotations

from langgraph.graph import END


# ---------------------------------------------------------------------------
# Local mirror of should_write_traits (graph.py, _build_graph scope)
# Keep this in sync with any changes to the production routing function.
# ---------------------------------------------------------------------------

def _should_write_traits(state: dict) -> str:
    """Mirror of should_write_traits nested inside AgentGraph._build_graph()."""
    if state.get("pending_trait_updates"):
        return "write_traits"
    return END


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_pending_routes_to_end():
    state = {"pending_trait_updates": []}
    assert _should_write_traits(state) == END


def test_non_empty_pending_routes_to_write_traits():
    state = {"pending_trait_updates": [{"price_sensitivity": "low"}]}
    assert _should_write_traits(state) == "write_traits"


def test_missing_key_routes_to_end():
    """State dict without the key at all (e.g. first turn) must not raise."""
    assert _should_write_traits({}) == END
