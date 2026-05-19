# Grounded answer synthesis using GPT-4o-mini over the top-k reranked chunks.
# The system prompt is pinned at module level to prevent prompt injection via user input.

from __future__ import annotations


async def generate(query: str, chunks: list[dict]) -> tuple[str, list[str]]:
    # Formats chunks as a numbered list and calls GPT-4o-mini.
    # System prompt is pinned at module level — never modified at runtime
    # to prevent prompt injection via user query.
    # Returns (answer_text, cited_product_ids).
    raise NotImplementedError
