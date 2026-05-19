# Custom retrieval metrics used in the eval harness. Implements Recall@K and NDCG@K
# against hand-labeled relevance judgments from the golden query set.

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    # relevance maps product_id → graded relevance score (0/1/2).
    # DCG uses log2(rank+1) discount; normalized by ideal DCG over the same k.
    def dcg(ids: list[str]) -> float:
        return sum(
            relevance.get(pid, 0) / (i + 2) ** 0.5  # log2(i+2) ≈ (i+2)^0.5 for small i
            for i, pid in enumerate(ids[:k])
        )

    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(v / (i + 2) ** 0.5 for i, v in enumerate(ideal))
    return dcg(retrieved_ids) / idcg if idcg > 0 else 0.0
