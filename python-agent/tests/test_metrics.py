"""Tests for the per-node latency / token / cost instrumentation.

Split in two halves:

  * Pure units for metrics.py — the zero-vs-unknown distinction and the
    "unpriced model must not read as free" rule are the two invariants that,
    if broken, silently produce plausible-looking wrong numbers.

  * Integration against a real AgentGraph (MemorySaver, mocked LLMs) for the
    graph wiring. The tools node is the load-bearing one: ToolNode returns a
    plain dict normally but a list of Commands when a tool returns Command
    (update_user_traits does), and the wrapper has to preserve both.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from agent.metrics import (
    MetricsSink,
    NodeMetric,
    Pricing,
    Stopwatch,
    TurnMetrics,
    extract_usage,
)
from eval.aggregate_metrics import aggregate, percentile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def msg_with_usage(text: str = "hi", inp: int = 100, out: int = 20) -> AIMessage:
    m = AIMessage(content=text)
    m.usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    return m


PRICED = Pricing(
    models={"router-model": {"input": 1.0, "output": 2.0}, "free-model": {"input": 0.0, "output": 0.0}},
    as_of="2026-01-01",
    counterfactual_model="router-model",
)


# ---------------------------------------------------------------------------
# extract_usage — zero vs unknown
# ---------------------------------------------------------------------------

def test_extract_usage_reads_langchain_metadata():
    assert extract_usage(msg_with_usage(inp=100, out=20)) == (100, 20, 120, True)


def test_extract_usage_missing_metadata_is_unavailable_not_zero():
    # The flag is the whole point: a mocked or provider-silent call must be
    # distinguishable from a call that genuinely consumed nothing.
    assert extract_usage(AIMessage(content="x")) == (0, 0, 0, False)


def test_extract_usage_derives_total_when_provider_omits_it():
    m = AIMessage(content="x")
    m.usage_metadata = {"input_tokens": 7, "output_tokens": 3}
    assert extract_usage(m) == (7, 3, 10, True)


def test_extract_usage_survives_garbage_metadata():
    m = AIMessage(content="x")
    m.usage_metadata = {"input_tokens": "not-a-number"}
    assert extract_usage(m) == (0, 0, 0, False)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def test_pricing_computes_per_million_rate():
    # 1_000_000 in @ $1/M + 1_000_000 out @ $2/M = $3
    assert PRICED.cost("router-model", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_unknown_model_costs_none_not_zero():
    assert PRICED.cost("some-model-nobody-priced", 1_000, 1_000) is None


def test_free_model_costs_zero_not_none():
    assert PRICED.cost("free-model", 1_000, 1_000) == 0.0


def test_pricing_normalises_models_prefix():
    assert PRICED.cost("models/router-model", 1_000_000, 0) == pytest.approx(1.0)


def test_shipped_pricing_table_covers_code_default_models():
    # Guards against renaming a default model in AgentConfig without adding it
    # to pricing.json, which would silently disable cost reporting.
    #
    # Reads the DEFAULTS off the field definitions rather than instantiating
    # AgentConfig: instantiating loads .env, so on a developer machine this
    # would assert against whatever that machine happens to be running.
    # Operator-chosen models that are missing from the table are surfaced at
    # aggregation time instead (cost.unpriced_models).
    from agent.graph import AgentConfig

    shipped = Pricing.load()
    for field in ("router_model", "final_model"):
        default = AgentConfig.model_fields[field].default
        assert shipped.cost(default, 1, 1) is not None, f"{field}={default} missing from pricing.json"


def test_missing_pricing_file_disables_cost_without_raising():
    p = Pricing.load("/nonexistent/pricing.json")
    assert p.cost("anything", 1, 1) is None


# ---------------------------------------------------------------------------
# TurnMetrics roll-ups
# ---------------------------------------------------------------------------

def _turn(nodes: list[NodeMetric]) -> TurnMetrics:
    return TurnMetrics(
        session_id="s", user_id="u", total_latency_ms=500.0,
        nodes=[n.as_dict() for n in nodes],
    )


def test_turn_totals_exclude_calls_without_usage():
    t = _turn([
        NodeMetric("router", 10.0, model="m", input_tokens=100, output_tokens=10,
                   total_tokens=110, usage_available=True, cost_usd=0.001),
        NodeMetric("finalizer", 20.0, model="m", input_tokens=999, output_tokens=999,
                   total_tokens=1998, usage_available=False),
    ])
    totals = t.token_totals()
    assert totals["total"] == 110
    # Coverage makes the omission visible rather than just producing a low total.
    assert totals["llm_calls"] == 2
    assert totals["llm_calls_with_usage"] == 1


def test_turn_cost_is_none_when_any_call_is_unpriced():
    t = _turn([
        NodeMetric("router", 10.0, model="m", usage_available=True, cost_usd=0.001),
        NodeMetric("finalizer", 20.0, model="unpriced", usage_available=True, cost_usd=None),
    ])
    assert t.total_cost() is None


def test_turn_cost_sums_when_all_priced():
    t = _turn([
        NodeMetric("router", 10.0, model="a", usage_available=True, cost_usd=0.001),
        NodeMetric("finalizer", 20.0, model="b", usage_available=True, cost_usd=0.002),
    ])
    assert t.total_cost() == pytest.approx(0.003)


def test_turn_record_is_json_serialisable():
    rec = _turn([NodeMetric("router", 1.0, model="a", usage_available=True, cost_usd=0.0)]).as_record()
    assert json.loads(json.dumps(rec))["session_id"] == "s"


# ---------------------------------------------------------------------------
# MetricsSink
# ---------------------------------------------------------------------------

def test_sink_is_disabled_by_default():
    # Library and test use must never write files; main.py opts in explicitly.
    assert MetricsSink().enabled is False
    MetricsSink().write({"a": 1})  # no-op, no exception


def test_sink_appends_one_json_line_per_write(tmp_path):
    path = tmp_path / "nested" / "turns.jsonl"
    sink = MetricsSink(path)
    sink.write({"n": 1})
    sink.write({"n": 2})
    lines = path.read_text().strip().splitlines()
    assert [json.loads(x)["n"] for x in lines] == [1, 2]


def test_sink_swallows_write_errors(tmp_path):
    # A metrics failure must never fail the user's request.
    bad = tmp_path / "afile"
    bad.write_text("x")
    MetricsSink(bad / "cannot" / "exist").write({"a": 1})


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------

def test_stopwatch_measures_elapsed():
    with Stopwatch() as sw:
        sum(range(200_000))
    assert sw.ms > 0.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,expected", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)])
def test_percentile_endpoints_and_median(q, expected):
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], q) == expected


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 0.5) == 5.0


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.95) == 0.0


def _rec(router_tokens: int, final_tokens: int, latency: float, path: str = "finalizer") -> dict:
    return TurnMetrics(
        session_id="s", user_id="u", total_latency_ms=latency, path=path, iterations=1,
        nodes=[
            NodeMetric("router", latency * 0.6, model="router-model", input_tokens=router_tokens,
                       output_tokens=0, total_tokens=router_tokens, usage_available=True,
                       cost_usd=PRICED.cost("router-model", router_tokens, 0)).as_dict(),
            NodeMetric("finalizer", latency * 0.3, model="free-model", input_tokens=final_tokens,
                       output_tokens=0, total_tokens=final_tokens, usage_available=True,
                       cost_usd=PRICED.cost("free-model", final_tokens, 0)).as_dict(),
        ],
    ).as_record()


def test_aggregate_splits_tokens_by_node_and_model():
    agg = aggregate([_rec(800, 200, 100.0), _rec(800, 200, 300.0)], PRICED)
    assert agg["turns"] == 2
    assert agg["tokens"]["by_node"]["router"]["total"] == 1600
    assert agg["tokens"]["by_node"]["router"]["share"] == pytest.approx(0.8)
    assert agg["tokens"]["by_model"]["free-model"]["share"] == pytest.approx(0.2)


def test_aggregate_reports_unattributed_framework_overhead():
    # Nodes account for 90% of each turn; the rest is LangGraph/checkpointer.
    agg = aggregate([_rec(100, 100, 1000.0)], PRICED)
    assert agg["latency"]["unattributed_share"] == pytest.approx(0.1, abs=1e-6)


def test_aggregate_counterfactual_reprices_all_tokens_at_router_model():
    agg = aggregate([_rec(800, 200, 100.0)], PRICED)
    cost = agg["cost"]
    assert cost["available"] is True
    # actual: 800 router tokens @ $1/M; finalizer is free.
    assert cost["total_usd"] == pytest.approx(800 / 1_000_000)
    # counterfactual: all 1000 tokens at router price.
    assert cost["counterfactual_total_usd"] == pytest.approx(1000 / 1_000_000)
    assert cost["saving_vs_counterfactual"] == pytest.approx(0.2)


def test_aggregate_refuses_cost_when_a_model_is_unpriced():
    rec = _rec(100, 100, 50.0)
    rec["nodes"][1]["model"] = "mystery-model"
    agg = aggregate([rec], PRICED)
    assert agg["cost"]["available"] is False
    assert agg["cost"]["unpriced_models"] == ["mystery-model"]


def test_aggregate_counts_answer_paths():
    agg = aggregate([_rec(1, 1, 10.0, "finalizer"), _rec(1, 1, 10.0, "fallback")], PRICED)
    assert agg["paths"] == {"finalizer": 1, "fallback": 1}


# ---------------------------------------------------------------------------
# Integration — graph wiring
# ---------------------------------------------------------------------------

async def test_chat_records_router_and_finalizer_separately(agent_graph):
    agent_graph._router_model.ainvoke = AsyncMock(return_value=msg_with_usage("ok", 500, 50))
    agent_graph._final_model.ainvoke = AsyncMock(return_value=msg_with_usage("answer", 200, 80))

    result = await agent_graph.chat("hello", "u1", "sess-metrics-1")

    by_node = {n["node"]: n for n in result.node_metrics}
    assert by_node["router"]["total_tokens"] == 550
    assert by_node["finalizer"]["total_tokens"] == 280
    # The old single counter is preserved for backwards compatibility.
    assert result.tokens_used == 830
    assert result.latency_ms > 0.0


async def test_chat_records_tool_node_when_tools_run(agent_graph):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_preferences", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    agent_graph._router_model.ainvoke = AsyncMock(side_effect=[tool_call, msg_with_usage("done")])
    agent_graph._final_model.ainvoke = AsyncMock(return_value=msg_with_usage("answer"))

    result = await agent_graph.chat("what do I like", "u1", "sess-metrics-2")

    nodes = [n["node"] for n in result.node_metrics]
    assert nodes.count("router") == 2
    assert "tools" in nodes


async def test_tools_node_metric_survives_command_returning_tool(agent_graph):
    """update_user_traits returns a Command, so ToolNode returns a LIST of
    updates rather than a dict. The wrapper must still attach its metric AND
    still stage the trait update."""
    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "update_user_traits",
            "args": {"style_preferences": {"minimalist": 0.9}},
            "id": "c1", "type": "tool_call",
        }],
    )
    agent_graph._router_model.ainvoke = AsyncMock(side_effect=[tool_call, AIMessage(content="ok")])
    agent_graph._final_model.ainvoke = AsyncMock(return_value=msg_with_usage("answer"))

    result = await agent_graph.chat("I love minimalist style", "u1", "sess-metrics-3")

    assert "tools" in [n["node"] for n in result.node_metrics]
    # The Command path still worked — the interrupt fired.
    assert result.pending_approval is True


async def test_fallback_path_is_tagged_and_costs_nothing(agent_graph, mock_db):
    """All tools empty → quality gate routes to fallback, which makes no LLM
    call. The path tag is what the hallucination A/B will bucket on."""
    mock_db.get_user_traits = AsyncMock(return_value=None)
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_preferences", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    agent_graph._router_model.ainvoke = AsyncMock(side_effect=[tool_call, AIMessage(content="")])
    agent_graph._final_model.ainvoke = AsyncMock(return_value=msg_with_usage("should not be used"))

    result = await agent_graph.chat("recommend me something", "u1", "sess-metrics-4")

    nodes = [n["node"] for n in result.node_metrics]
    assert "fallback" in nodes
    assert "finalizer" not in nodes
    agent_graph._final_model.ainvoke.assert_not_called()


async def test_mocked_models_report_unknown_cost_not_free(agent_graph):
    """Mocks carry no usage_metadata. Cost must come back None, never 0.0 —
    otherwise a MOCK_AI=true run would look like a free production run."""
    agent_graph._router_model.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
    agent_graph._final_model.ainvoke = AsyncMock(return_value=AIMessage(content="answer"))

    result = await agent_graph.chat("hello", "u1", "sess-metrics-5")

    assert result.cost_usd is None
    assert all(n["usage_available"] is False for n in result.node_metrics if n["model"])


async def test_chat_writes_one_line_to_the_sink(agent_graph, tmp_path):
    path = tmp_path / "turns.jsonl"
    agent_graph._metrics_sink = MetricsSink(path)
    agent_graph._router_model.ainvoke = AsyncMock(return_value=msg_with_usage("ok", 10, 5))
    agent_graph._final_model.ainvoke = AsyncMock(return_value=msg_with_usage("answer", 10, 5))

    await agent_graph.chat("hello", "u1", "sess-metrics-6")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["turn_type"] == "chat"
    assert rec["session_id"] == "sess-metrics-6"
    assert rec["tokens"]["total"] == 30
