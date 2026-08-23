"""Per-node latency / token / cost instrumentation for the agent graph.

Why this module exists
----------------------
Before this, AgentState carried a single `tokens_used` counter that summed the
router model and the finalizer model into one number, and nothing measured
time at all.  That makes the two headline claims about this agent unfalsifiable:

  * "model tiering moved synthesis off the function-calling model"  — needs the
    router/finalizer token split, not their sum.
  * "p95 latency is X"                                              — needs timing.

So each graph node now emits a NodeMetric, the turn assembles them into a
TurnMetrics record, and MetricsSink appends one JSON line per turn.
eval/aggregate_metrics.py turns that file into percentiles and cost shares.

Zero vs unknown
---------------
`usage_metadata` is absent on mocked models, on some streaming paths, and when
the provider omits it.  Treating that as 0 tokens silently deflates every
aggregate.  extract_usage() therefore returns an explicit `available` flag and
the aggregator reports coverage, so a run with missing usage is visible rather
than merely cheap-looking.

Likewise an unpriced model yields cost_usd = None, never 0.0.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PRICING_PATH = Path(__file__).parent / "pricing.json"


# ---------------------------------------------------------------------------
# Node-level record
# ---------------------------------------------------------------------------

@dataclass
class NodeMetric:
    """One execution of one graph node.

    A node can run several times per turn (router_node runs once per ReAct
    iteration), so `iteration` disambiguates them within a turn.
    """

    node: str
    latency_ms: float
    iteration: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # False means "the provider did not report usage", NOT "this call used 0
    # tokens".  Aggregations must not average the two together.
    usage_available: bool = False
    cost_usd: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------

def extract_usage(resp: Any) -> tuple[int, int, int, bool]:
    """Pull (input, output, total, available) out of a LangChain AIMessage.

    langchain-core normalises provider usage onto `usage_metadata` with keys
    input_tokens / output_tokens / total_tokens.  Some providers report the
    components but not the total, so total falls back to input + output.
    """
    usage = getattr(resp, "usage_metadata", None)
    if not usage:
        return 0, 0, 0, False
    try:
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or 0) or (inp + out)
    except (AttributeError, TypeError, ValueError):
        return 0, 0, 0, False
    return inp, out, total, True


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class Pricing:
    """USD cost lookup backed by pricing.json.

    Model ids are normalised by stripping a leading "models/" (the Google SDK
    accepts and sometimes echoes both forms).
    """

    def __init__(
        self,
        models: dict[str, dict[str, float]],
        as_of: str = "",
        source: str = "",
        counterfactual_model: str = "",
    ) -> None:
        self._models = models
        self.as_of = as_of
        self.source = source
        self.counterfactual_model = counterfactual_model

    @classmethod
    def load(cls, path: Path | str = _PRICING_PATH) -> "Pricing":
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # Missing/broken price table must not take the agent down — costs
            # simply become None for every model.
            logger.warning("Pricing table unreadable at %s (%s) — costs disabled.", path, exc)
            return cls(models={})
        return cls(
            models=raw.get("models") or {},
            as_of=raw.get("as_of", ""),
            source=raw.get("source", ""),
            counterfactual_model=raw.get("counterfactual_model", ""),
        )

    @staticmethod
    def normalise(model: str) -> str:
        return (model or "").removeprefix("models/")

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """USD for one call, or None when the model is not in the table."""
        entry = self._models.get(self.normalise(model))
        if entry is None:
            return None
        return (
            input_tokens * float(entry.get("input", 0.0))
            + output_tokens * float(entry.get("output", 0.0))
        ) / 1_000_000


# ---------------------------------------------------------------------------
# Turn-level record
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    """Everything measured for one /agent-chat (or /agent-resume) request."""

    session_id: str
    user_id: str
    total_latency_ms: float
    nodes: list[dict[str, Any]] = field(default_factory=list)
    turn_type: str = "chat"          # chat | resume
    iterations: int = 0
    # Which answer node produced the reply: "finalizer" (LLM synthesis),
    # "fallback" (deterministic no-signal reply), or "" for a resume turn.
    # The hallucination A/B reads this to bucket turns by which guard fired.
    path: str = ""
    ts: str = ""
    pricing_as_of: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # -- derived roll-ups -------------------------------------------------

    def token_totals(self) -> dict[str, Any]:
        llm_nodes = [n for n in self.nodes if n.get("model")]
        reported = [n for n in llm_nodes if n.get("usage_available")]
        return {
            "input": sum(n["input_tokens"] for n in reported),
            "output": sum(n["output_tokens"] for n in reported),
            "total": sum(n["total_tokens"] for n in reported),
            # Coverage, so a run where the provider withheld usage is obvious
            # instead of just looking cheap.
            "llm_calls": len(llm_nodes),
            "llm_calls_with_usage": len(reported),
        }

    def tokens_by_model(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for n in self.nodes:
            model = n.get("model")
            if not model or not n.get("usage_available"):
                continue
            bucket = out.setdefault(model, {"input": 0, "output": 0, "total": 0})
            bucket["input"] += n["input_tokens"]
            bucket["output"] += n["output_tokens"]
            bucket["total"] += n["total_tokens"]
        return out

    def cost_by_model(self) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for n in self.nodes:
            model = n.get("model")
            if not model:
                continue
            c = n.get("cost_usd")
            if c is None:
                # One unpriced call poisons the whole model's total — better a
                # null than a number that quietly omits part of the spend.
                out[model] = None
            elif out.get(model, 0.0) is not None:
                out[model] = (out.get(model) or 0.0) + c
        return out

    def total_cost(self) -> Optional[float]:
        per_model = self.cost_by_model()
        if any(v is None for v in per_model.values()):
            return None
        return sum(v or 0.0 for v in per_model.values())

    def as_record(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "turn_type": self.turn_type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "iterations": self.iterations,
            "path": self.path,
            "tokens": self.token_totals(),
            "tokens_by_model": self.tokens_by_model(),
            "cost_usd": self.total_cost(),
            "cost_by_model": self.cost_by_model(),
            "pricing_as_of": self.pricing_as_of,
            "nodes": self.nodes,
        }


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

class MetricsSink:
    """Appends one JSON line per turn.

    Disabled (a no-op) when no path is configured, which is the default — tests
    and library use must not litter the filesystem.  main.py opts in explicitly.

    A plain lock + append is enough here: one small write per HTTP request, and
    O_APPEND writes below PIPE_BUF are atomic on POSIX, so concurrent workers
    interleave whole lines rather than corrupting them.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_env(cls) -> "MetricsSink":
        return cls(os.environ.get("AGENT_METRICS_PATH") or None)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Instrumentation must never fail a user request.  Warn once.
            if not self._warned:
                logger.warning("MetricsSink write failed (%s) — further errors suppressed.", exc)
                self._warned = True


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Stopwatch:
    """`with Stopwatch() as sw: ...` then read `sw.ms`.

    perf_counter, not time.time — monotonic and unaffected by clock changes.
    """

    __slots__ = ("_t0", "ms")

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0
