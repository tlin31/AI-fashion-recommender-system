# Cross-encoder reranker that scores all retrieved candidates in a single batch forward
# pass, then applies business rule adjustments (recency boost, low-stock demotion)
# before selecting the top-k results passed to the generator.

from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self) -> None:
        # Loaded once at startup (app.state.reranker) — never per-request.
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        # Score all candidates in a single batch forward pass, then apply
        # business rules as score adjustments before selecting top_k:
        #   +boost for items added in the last 30 days
        #   -demote for stock_level == "low"
        raise NotImplementedError
