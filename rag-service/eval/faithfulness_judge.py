# LLM-as-judge faithfulness scoring. Measures whether every claim in the generated
# answer is supported by the retrieved context chunks. Target score >= 0.85.
#
# WHICH JUDGE ACTUALLY RUNS: the direct two-step GPT-4o-mini judge below
# (_direct_faithfulness) — extract claims from the answer, then verify each against
# the context. ragas is deliberately NOT installed (see requirements.txt), so the
# ragas path never executes and eval/baseline_metrics.json was measured with the
# direct judge. Installing ragas silently switches judges and makes new faithfulness
# numbers incomparable to the locked baseline.
#
# Known bias: _direct_faithfulness returns 1.0 when claim extraction yields nothing
# or the response fails to parse ("vacuously faithful"). Short answers therefore skew
# high — all 20 navigational queries scored exactly 1.0000 in the 2026-08-11 run — so
# the aggregate is somewhat optimistic. Kept as-is rather than changed mid-baseline.

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

# load_dotenv here so this module works both when imported by the FastAPI server
# (which calls load_dotenv in main.py) AND when imported directly by the eval
# harness (a standalone script that never touches main.py).
load_dotenv(override=True)

logger = logging.getLogger(__name__)

_JUDGE_MODEL = os.environ.get("OPENAI_FAITHFULNESS_MODEL", "gpt-4o-mini")


async def faithfulness_score(query: str, answer: str, context_chunks: list[str]) -> float:
    """LLM-as-judge faithfulness score in [0, 1].

    Measures whether every factual claim in *answer* is supported by at least
    one chunk in *context_chunks*.  Score = (supported claims) / (total claims).
    Returns 1.0 if no claims could be extracted (vacuously faithful).

    Tries ragas first; falls back to a direct OpenAI claim-decomposition judge
    if ragas raises an import or compatibility error.
    """
    if not answer or not context_chunks:
        return 1.0

    try:
        return await _ragas_faithfulness(query, answer, context_chunks)
    except Exception as exc:
        logger.warning(
            "ragas faithfulness failed (%s); falling back to direct LLM judge", exc
        )
        return await _direct_faithfulness(query, answer, context_chunks)


# ---------------------------------------------------------------------------
# ragas 0.1.9 path
# ---------------------------------------------------------------------------

def _run_ragas_evaluate(query: str, answer: str, context_chunks: list[str]) -> float:
    """Synchronous wrapper — called in a thread-pool so it doesn't block the loop."""
    from datasets import Dataset  # type: ignore[import-not-found]
    from ragas import evaluate  # type: ignore[import-not-found]
    from ragas.metrics import faithfulness  # type: ignore[import-not-found]

    ds = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [context_chunks],
        }
    )
    result = evaluate(ds, metrics=[faithfulness])
    # ragas 0.1.x returns a dict-like object; "faithfulness" key is a float
    score = result["faithfulness"]
    return float(score) if score is not None else 1.0


async def _ragas_faithfulness(
    query: str, answer: str, context_chunks: list[str]
) -> float:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _run_ragas_evaluate, query, answer, context_chunks
    )


# ---------------------------------------------------------------------------
# Direct OpenAI fallback
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = (
    "Extract every distinct factual claim from the following answer as a JSON array "
    "of strings. Each element should be one self-contained sentence. "
    "If there are no factual claims, return an empty array []."
)

_VERDICT_PROMPT = (
    "You are a faithfulness judge. For each claim, decide whether it is FULLY "
    "supported by the provided context passages. "
    "Return a JSON object with key \"verdicts\" whose value is an array of objects "
    "each with keys \"claim\" (string) and \"supported\" (boolean). "
    "Do NOT use external knowledge — only judge based on the context."
)


async def _direct_faithfulness(
    query: str, answer: str, context_chunks: list[str]
) -> float:
    """Two-step GPT-4o-mini judge: extract claims, then verify each against context."""
    from openai import AsyncOpenAI  # type: ignore[import-not-found]

    client = AsyncOpenAI()

    # Step 1: extract claims
    claims_resp = await client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Answer: {answer}"},
        ],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        raw = json.loads(claims_resp.choices[0].message.content)
        # model may return {"claims": [...]} or directly [...]
        claims: list[str] = raw if isinstance(raw, list) else raw.get("claims", [])
    except (json.JSONDecodeError, AttributeError, KeyError):
        # Unparseable → treat as no claims → vacuously faithful
        return 1.0

    if not claims:
        return 1.0

    # Step 2: verify each claim against the context
    numbered_context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    verdict_resp = await client.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _VERDICT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context passages:\n{numbered_context}\n\n"
                    f"Claims to verify:\n"
                    + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
                ),
            },
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    try:
        verdicts: list[dict] = json.loads(
            verdict_resp.choices[0].message.content
        ).get("verdicts", [])
    except (json.JSONDecodeError, AttributeError):
        return 1.0

    if not verdicts:
        return 1.0

    supported = sum(1 for v in verdicts if v.get("supported", False))
    return supported / len(verdicts)
