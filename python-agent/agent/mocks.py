"""Mock LLM models for MOCK_AI=true mode.

Swaps _router_model and _final_model on an already-constructed AgentGraph
with AsyncMock objects that return canned AIMessage responses — no real API
calls made. Pattern mirrors tests/conftest.py exactly.

Scenario selection is keyword-based on the last user message:
  - HITL path  : message contains preference keywords → stages update_user_traits
  - Happy path : everything else → get_user_preferences → get_recommendations → STOP
"""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

_HAPPY_PATH_RESPONSES = [
    AIMessage(
        content="",
        tool_calls=[{"name": "get_user_preferences", "args": {}, "id": "mock_call_1", "type": "tool_call"}],
    ),
    AIMessage(
        content="",
        tool_calls=[{"name": "get_recommendations", "args": {"n": 5}, "id": "mock_call_2", "type": "tool_call"}],
    ),
    AIMessage(content=""),  # STOP — no tool_calls
]

_HITL_PATH_RESPONSES = [
    AIMessage(
        content="",
        tool_calls=[{
            "name": "update_user_traits",
            "args": {"style_preferences": {"minimalist": 0.9}, "color_preferences": {"neutral": 0.8}},
            "id": "mock_call_1",
            "type": "tool_call",
        }],
    ),
    AIMessage(content=""),  # STOP
]

_HITL_KEYWORDS = {"prefer", "love", "like", "minimalist", "style", "color", "colour", "price", "brand", "casual", "formal"}

_FINALIZER_RESPONSE = AIMessage(content="""Here are **3 minimalist tops** I'd recommend for you:

**1. Ivory Ribbed Tee** *(product_002)*
- Fabric: 95% organic cotton
- Colors available: Ivory, Stone, Slate
- Perfect for: *casual days, layering under blazers*

**2. Structured Linen Shirt** *(product_005)*
- Fabric: 100% linen
- Colors available: Off-white, Sand
- Perfect for: *smart-casual, work-from-home*

**3. Oversized Poplin Blouse** *(product_008)*
- Fabric: Lightweight poplin
- Colors available: Cream, Pale blue
- Perfect for: *weekend outings, tucked into wide-leg trousers*

---

> 💡 **Styling tip:** All three pair well with neutral-toned wide-leg pants — a staple of the minimalist wardrobe right now.""")


# ---------------------------------------------------------------------------
# Router mock — stateful per-call counter using itertools.cycle
# ---------------------------------------------------------------------------

class _MockRouter:
    """Inspects the last HumanMessage to pick a scenario, then cycles through
    the scenario's canned AIMessage responses."""

    def __init__(self) -> None:
        self._iter: itertools.cycle | None = None
        self._scenario: str | None = None

    def _pick_scenario(self, messages: list) -> str:
        from langchain_core.messages import HumanMessage
        for msg in reversed(messages):
            if not isinstance(msg, HumanMessage):
                continue
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                words = set(content.lower().split())
                if words & _HITL_KEYWORDS:
                    return "hitl"
            break  # only check the most recent HumanMessage
        return "happy"

    async def ainvoke(self, messages: list, **kwargs) -> AIMessage:
        scenario = self._pick_scenario(messages)
        if scenario != self._scenario:
            responses = _HITL_PATH_RESPONSES if scenario == "hitl" else _HAPPY_PATH_RESPONSES
            self._iter = itertools.cycle(responses)
            self._scenario = scenario
        return next(self._iter)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_mock_models(graph) -> None:  # graph: AgentGraph (avoid circular import)
    """Replace the real LLMs on *graph* with mock objects. Call this after
    AgentGraph is constructed but before the first request arrives."""
    graph._router_model = _MockRouter()

    final_mock = AsyncMock()
    final_mock.ainvoke = AsyncMock(return_value=_FINALIZER_RESPONSE)
    graph._final_model = final_mock
