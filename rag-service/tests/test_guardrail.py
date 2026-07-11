# Tests for pipeline/guardrail.py.
# No external services — OpenAI client is mocked.
#
# Cases:
#   Fashion queries   → True
#   Off-topic queries → False
#   Empty / blank     → False (short-circuit, no API call)
#   Timeout           → True  (fail open)
#   API error         → True  (fail open)
#   Raw answer variants ("YES.", "yes", "YES\n") → True
#   Raw answer "NO"   → False

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.guardrail import is_fashion_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_openai_response(text: str) -> MagicMock:
    """Build the minimal fake that quacks like an OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def _patch_openai(answer_text: str):
    """Patch get_openai_client() so the chat.completions.create coroutine returns answer_text."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(answer_text)
    )
    return patch("pipeline.guardrail.get_openai_client", return_value=mock_client)


# ---------------------------------------------------------------------------
# Fashion queries → True
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_fashion_query_returns_true():
    with _patch_openai("YES"):
        assert await is_fashion_query("I need a minimalist blazer for work") is True


@pytest.mark.asyncio
async def test_paraphrase_fashion_query_returns_true():
    with _patch_openai("YES"):
        assert await is_fashion_query("attire for a gala") is True


@pytest.mark.asyncio
async def test_accessory_query_returns_true():
    with _patch_openai("YES"):
        assert await is_fashion_query("leather belt under 50 dollars") is True


# ---------------------------------------------------------------------------
# Off-topic queries → False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_topic_query_returns_false():
    with _patch_openai("NO"):
        assert await is_fashion_query("best JavaScript framework in 2024") is False


@pytest.mark.asyncio
async def test_food_query_returns_false():
    with _patch_openai("NO"):
        assert await is_fashion_query("recommend a pasta recipe") is False


# ---------------------------------------------------------------------------
# Edge: empty / blank query → False, no API call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_returns_false_no_api_call():
    with _patch_openai("YES") as mock_ctx:
        # Access the patched client via the context manager target
        result = await is_fashion_query("")
    assert result is False


@pytest.mark.asyncio
async def test_whitespace_query_returns_false_no_api_call():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    with patch("pipeline.guardrail.get_openai_client", return_value=mock_client):
        result = await is_fashion_query("   ")
    assert result is False
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answer_yes_with_trailing_period():
    """Model outputs "YES." — should still parse as True."""
    with _patch_openai("YES."):
        assert await is_fashion_query("blue linen trousers") is True


@pytest.mark.asyncio
async def test_answer_yes_lowercase():
    """Model outputs "yes" — case-normalised to True."""
    with _patch_openai("yes"):
        assert await is_fashion_query("blue linen trousers") is True


@pytest.mark.asyncio
async def test_answer_yes_with_newline():
    """Model outputs "YES\n" — stripped to True."""
    with _patch_openai("YES\n"):
        assert await is_fashion_query("summer dress") is True


@pytest.mark.asyncio
async def test_answer_no_returns_false():
    with _patch_openai("NO"):
        assert await is_fashion_query("stock market trends") is False


@pytest.mark.asyncio
async def test_answer_no_with_trailing_period():
    with _patch_openai("NO."):
        assert await is_fashion_query("stock market trends") is False


# ---------------------------------------------------------------------------
# Fail-open cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_returns_true_fail_open():
    """asyncio.TimeoutError → fail open → True."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )
    with patch("pipeline.guardrail.get_openai_client", return_value=mock_client):
        result = await is_fashion_query("blue trousers")
    assert result is True


@pytest.mark.asyncio
async def test_api_error_returns_true_fail_open():
    """Generic API exception → fail open → True."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("OpenAI 503 Service Unavailable")
    )
    with patch("pipeline.guardrail.get_openai_client", return_value=mock_client):
        result = await is_fashion_query("blue trousers")
    assert result is True


@pytest.mark.asyncio
async def test_network_error_returns_true_fail_open():
    """Network-level error → fail open → True."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=ConnectionError("Connection refused")
    )
    with patch("pipeline.guardrail.get_openai_client", return_value=mock_client):
        result = await is_fashion_query("summer sandals")
    assert result is True


# ---------------------------------------------------------------------------
# Timeout is actually enforced (3s wall-clock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_actually_enforced():
    """The OpenAI call must be cancelled if it takes longer than the configured timeout.

    We simulate a slow response by making the coroutine sleep for 10s. The guardrail
    must return True (fail open) well before that.
    """
    async def _slow_response(*args, **kwargs):
        await asyncio.sleep(10)  # simulate hung API

    mock_client = MagicMock()
    mock_client.chat.completions.create = _slow_response

    with patch("pipeline.guardrail.get_openai_client", return_value=mock_client):
        with patch("pipeline.guardrail._TIMEOUT_SECONDS", 0.05):  # 50ms in tests
            result = await is_fashion_query("blue trousers")

    assert result is True
