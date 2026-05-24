# LLM-as-judge faithfulness scoring via Ragas. Measures whether every claim in the
# generated answer is supported by the retrieved context chunks. Target score >= 0.85.

from __future__ import annotations


async def faithfulness_score(query: str, answer: str, context_chunks: list[str]) -> float:
    # Ragas LLM-as-judge: measures whether every claim in the answer
    # is supported by the retrieved context chunks.
    # Returns a score in [0, 1]; target >= 0.85.
    raise NotImplementedError
