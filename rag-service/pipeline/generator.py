# Grounded answer synthesis using GPT-4o-mini over the top-k reranked chunks.
# The system prompt is pinned at module level to prevent prompt injection via user input.

from __future__ import annotations

import logging
import re

from pipeline.utils import get_openai_client

logger = logging.getLogger(__name__)

# Pinned at module level — never modified at runtime.
# "ONLY from the provided context" is the key grounding constraint.
_SYSTEM_PROMPT = (
    "You are a fashion recommendation assistant. "
    "Answer ONLY from the provided product context. "
    "If a claim cannot be supported, say 'I don't have that information.' "
    "Do not invent products or prices."
)


def _format_context(chunks: list[dict]) -> str:
    """Format chunks as a numbered list so the model can cite them by index."""
    return "\n\n".join(
        f"[{i + 1}] {chunk['text']}\nSource: {chunk['product_id']}"
        for i, chunk in enumerate(chunks)
    )


def _parse_cited_sources(answer: str, chunks: list[dict]) -> list[str]:
    """Extract product IDs for every [N] reference that appears in the answer.

    Falls back to all chunk product IDs if the model cited nothing explicitly —
    this is preferable to returning an empty cited_sources field.
    """
    # Find every [1], [2], … index mentioned in the answer.
    cited_indices = set(re.findall(r"\[(\d+)\]", answer))

    product_ids = []
    for idx in sorted(cited_indices, key=int):
        i = int(idx) - 1
        if 0 <= i < len(chunks):
            pid = chunks[i]["product_id"]
            if pid not in product_ids:
                product_ids.append(pid)

    # Fallback: include all source IDs so cited_sources is never empty.
    if not product_ids:
        product_ids = list(dict.fromkeys(c["product_id"] for c in chunks))

    return product_ids


async def generate(query: str, chunks: list[dict]) -> tuple[str, list[str]]:
    """Synthesise a grounded answer from the top-k reranked chunks.

    Returns (answer_text, cited_product_ids).

    temperature=0 keeps synthesis deterministic — variance in how sources are
    cited makes faithfulness evaluation hard to reproduce across runs.
    max_tokens=300 is tight enough to prevent rambling while allowing a complete
    multi-product recommendation.
    """
    if not chunks:
        return "I don't have that information.", []

    context = _format_context(chunks)
    user_message = f"Query: {query}\n\nProduct context:\n{context}"

    client = get_openai_client()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=300,
        temperature=0,
    )

    answer = response.choices[0].message.content or ""
    cited_sources = _parse_cited_sources(answer, chunks)

    logger.debug(
        "Generator: %d chunks → answer %d chars, %d cited sources",
        len(chunks), len(answer), len(cited_sources),
    )

    return answer, cited_sources
