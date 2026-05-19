# Query classifier that gates pipeline entry. Off-topic (non-fashion) queries are
# rejected here with a redirect response, before any retrieval or LLM calls are made.

from __future__ import annotations


async def is_fashion_query(query: str) -> bool:
    # GPT-4o-mini zero-shot: returns True if query is fashion-related.
    # Off-topic queries short-circuit the pipeline before any retrieval happens.
    raise NotImplementedError
