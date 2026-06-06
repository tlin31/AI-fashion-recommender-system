# Query classifier that gates pipeline entry. Off-topic (non-fashion) queries are
# rejected here with a redirect response, before any retrieval or LLM calls are made.
#
# Design decision — LLM classifier vs keyword list:
#   Keyword lists miss natural-language paraphrases ("attire for a gala",
#   "something to wear to a job interview"). GPT-4o-mini handles these correctly
#   with zero configuration. The ~50ms overhead is acceptable because the rest
#   of the pipeline takes 1-2s.
#
# Failure policy — fail open:
#   If the OpenAI call times out (>3s) or raises any exception, is_fashion_query()
#   returns True so users are never blocked by API flakiness. A false negative
#   (letting an off-topic query through) is far less harmful than a false positive
#   (silently refusing a valid fashion query).

from __future__ import annotations

import asyncio
import logging

from pipeline.utils import get_openai_client

logger = logging.getLogger(__name__)

# System prompt is a constant — never interpolate user input into it.
_SYSTEM_PROMPT = (
    "Classify: is this query about fashion, clothing, or accessories? "
    "Reply YES or NO only."
)

# Hard wall-clock timeout for the GPT-4o-mini call.
# 3s is generous for a single-token completion; p99 latency for gpt-4o-mini is ~1s.
_TIMEOUT_SECONDS = 3.0


async def is_fashion_query(query: str) -> bool:
    """Return True if the query is about fashion, clothing, or accessories.

    Uses a single GPT-4o-mini zero-shot classification call. Fails open on
    timeout or any API error — off-topic queries are a UX concern, not a
    security boundary, so blocking users on API flakiness is worse than
    occasionally letting a non-fashion query through.

    Args:
        query: The raw user query string.

    Returns:
        True  — query is fashion-related (or the guardrail check failed).
        False — query is definitively off-topic.
    """
    # Empty queries are not fashion queries; guard here before spending an API call.
    if not query or not query.strip():
        return False

    client = get_openai_client()

    try:
        # asyncio.wait_for wraps the coroutine with a hard timeout.
        # When the timeout fires it raises asyncio.TimeoutError, which is caught
        # below and treated as "fail open" rather than propagating to the caller.
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    # User content is the raw query — never interpolated into the
                    # system prompt to prevent prompt injection.
                    {"role": "user", "content": query.strip()},
                ],
                # "YES" and "NO" are each a single token; max_tokens=5 gives a
                # small buffer for leading spaces or trailing punctuation that
                # some model versions emit before the actual answer.
                max_tokens=5,
                temperature=0,  # deterministic — classification needs no creativity
            ),
            timeout=_TIMEOUT_SECONDS,
        )

        raw = response.choices[0].message.content or ""
        # Normalise: strip whitespace, uppercase, check prefix so "YES." and
        # "YES\n" both match even if the model adds trailing punctuation.
        answer = raw.strip().upper()
        is_fashion = answer.startswith("YES")

        logger.debug(
            "Guardrail: query=%r → raw_answer=%r → is_fashion=%s",
            query[:80],
            raw,
            is_fashion,
        )
        return is_fashion

    except asyncio.TimeoutError:
        # Guardrail API call timed out — fail open.
        logger.warning(
            "Guardrail timed out after %.1fs for query=%r; defaulting to True (fail open)",
            _TIMEOUT_SECONDS,
            query[:80],
        )
        return True

    except Exception as exc:
        # Any other error (network, auth, rate-limit) — fail open.
        logger.warning(
            "Guardrail check failed (%s: %s) for query=%r; defaulting to True (fail open)",
            type(exc).__name__,
            exc,
            query[:80],
        )
        return True
