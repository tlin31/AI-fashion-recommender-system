# Tests for pipeline/generator.py.
# OpenAI client is mocked — no real API calls made.
#
# Cases:
#   Empty chunks               → returns "I don't have that information."
#   Citation parsing           → [1][2] references mapped to correct product_ids
#   No citations in answer     → falls back to all chunk product_ids
#   System prompt              → pinned constant used in every call
#   Answer text                → returned verbatim from the model response

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.generator import _SYSTEM_PROMPT, _format_context, _parse_cited_sources, generate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(product_id: str, text: str = "nice product") -> dict:
    return {
        "chunk_id":    f"chunk_{product_id}",
        "product_id":  product_id,
        "text":        text,
        "score":       0.8,
        "milvus_score": 0.7,
        "metadata":    {"chunk_type": "description"},
    }


def _mock_openai(answer_text: str):
    """Patch get_openai_client() to return answer_text from chat.completions.create."""
    choice = MagicMock()
    choice.message.content = answer_text
    response = MagicMock()
    response.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    return patch("pipeline.generator.get_openai_client", return_value=mock_client)


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

def test_format_context_numbers_chunks():
    chunks = [_make_chunk("p001", "red dress"), _make_chunk("p002", "blue top")]
    ctx = _format_context(chunks)
    assert "[1]" in ctx and "red dress" in ctx
    assert "[2]" in ctx and "blue top" in ctx
    assert "Source: p001" in ctx
    assert "Source: p002" in ctx


# ---------------------------------------------------------------------------
# _parse_cited_sources
# ---------------------------------------------------------------------------

def test_parse_extracts_cited_indices():
    chunks = [_make_chunk("p001"), _make_chunk("p002"), _make_chunk("p003")]
    answer = "I recommend [1] and [3] for casual wear."
    sources = _parse_cited_sources(answer, chunks)
    assert sources == ["p001", "p003"]


def test_parse_deduplicates_same_product():
    chunks = [_make_chunk("p001"), _make_chunk("p001")]
    answer = "See [1] and [2] for details."
    sources = _parse_cited_sources(answer, chunks)
    assert sources.count("p001") == 1


def test_parse_fallback_when_no_citations():
    """No [N] references in answer → return all chunk product_ids."""
    chunks = [_make_chunk("p001"), _make_chunk("p002")]
    sources = _parse_cited_sources("Great products available.", chunks)
    assert "p001" in sources
    assert "p002" in sources


def test_parse_ignores_out_of_range_indices():
    chunks = [_make_chunk("p001")]
    answer = "See [1] and [99] for options."
    sources = _parse_cited_sources(answer, chunks)
    assert "p001" in sources
    # [99] is out of range and should be silently ignored
    assert len(sources) == 1


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

async def test_empty_chunks_returns_no_info():
    answer, sources = await generate("blue dress", [])
    assert "don't have" in answer.lower()
    assert sources == []


async def test_answer_returned_verbatim():
    chunks = [_make_chunk("p001")]
    with _mock_openai("I recommend the red blazer [1] for your event."):
        answer, _ = await generate("formal outfit", chunks)
    assert answer == "I recommend the red blazer [1] for your event."


async def test_cited_sources_parsed_from_answer():
    chunks = [_make_chunk("p001"), _make_chunk("p002")]
    with _mock_openai("The [1] item is perfect. Also consider [2]."):
        _, sources = await generate("casual top", chunks)
    assert "p001" in sources
    assert "p002" in sources


async def test_system_prompt_is_pinned():
    """The system prompt constant must be passed to the model on every call."""
    chunks = [_make_chunk("p001")]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="ok [1]"))])
    )
    with patch("pipeline.generator.get_openai_client", return_value=mock_client):
        await generate("blue dress", chunks)

    call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system_message = call_messages[0]
    assert system_message["role"] == "system"
    assert system_message["content"] == _SYSTEM_PROMPT


async def test_temperature_zero_and_max_tokens():
    """Determinism and token budget enforced in the API call."""
    chunks = [_make_chunk("p001")]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="ok [1]"))])
    )
    with patch("pipeline.generator.get_openai_client", return_value=mock_client):
        await generate("blue dress", chunks)

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 300
