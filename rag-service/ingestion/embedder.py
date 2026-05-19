# Batch embedding for ingestion-time workloads using text-embedding-3-small.
# Used only during data ingestion, not on the query path (query embedding lives in utils.py).

from __future__ import annotations

from pipeline.utils import get_openai_client


async def embed_batch(texts: list[str]) -> list[list[float]]:
    # Batch embed at ingestion time — never at query time.
    # OpenAI accepts up to 2048 inputs per request; chunk into batches if needed.
    response = await get_openai_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
