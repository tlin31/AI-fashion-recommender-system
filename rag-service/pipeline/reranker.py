# Cross-encoder reranker that scores all retrieved candidates in a single batch forward
# pass, applies business rule adjustments (description boost), then collapses to one
# chunk per product before selecting the top-k results passed to the generator.

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Description chunks carry structured, authoritative product facts.
# Reviews are subjective and often tangential to the query.
# A small boost ensures descriptions rank above reviews at equal cross-encoder scores.
_DESCRIPTION_BOOST = 0.05


class Reranker:
    def __init__(self) -> None:
        # Loaded once at startup (app.state.reranker) — never per-request.
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Score all candidates in one batch forward pass, apply business rules,
        collapse to one chunk per product, and return the top_k sorted by adjusted
        score descending.

        Batch predict — one forward pass over all candidates, not one per chunk.
        At 20 candidates this takes ~120ms on CPU, well within the latency budget.
        """
        if not candidates:
            return []

        # Single batched call — CrossEncoder scores (query, chunk_text) pairs jointly,
        # capturing interaction terms that a bi-encoder would miss.
        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = self._model.predict(pairs)

        adjusted = []
        for chunk, raw in zip(candidates, raw_scores):
            score = float(raw)

            # Business rule: boost description chunks over review chunks.
            if chunk.get("metadata", {}).get("chunk_type") == "description":
                score += _DESCRIPTION_BOOST

            adjusted.append({**chunk, "score": score})

        adjusted.sort(key=lambda x: x["score"], reverse=True)

        # Retrieval allows _MAX_CHUNKS_PER_PRODUCT chunks per product so the
        # cross-encoder can pick whichever one matches the query better. Only that
        # winner survives: max-pooling chunk scores up to the product. Without this,
        # one product occupies several result slots and any downstream consumer
        # counting product_ids (the eval harness, the UI) sees it more than once.
        deduped: list[dict] = []
        seen: set[str] = set()
        for chunk in adjusted:
            pid = chunk["product_id"]
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(chunk)
            if len(deduped) == top_k:
                break

        if len(deduped) < top_k:
            logger.info(
                "Reranker: only %d distinct products available from %d candidates "
                "(requested top_k=%d)",
                len(deduped), len(candidates), top_k,
            )

        logger.debug(
            "Reranker: %d candidates → %d distinct products (scores %.3f … %.3f)",
            len(candidates), len(deduped),
            deduped[0]["score"], deduped[-1]["score"],
        )

        return deduped
