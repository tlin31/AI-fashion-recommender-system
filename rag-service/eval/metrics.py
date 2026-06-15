# Custom retrieval metrics used in the eval harness. Implements Recall@K and NDCG@K
# against hand-labeled relevance judgments from the golden query set.

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    # Standard NDCG with exponential gain and log2 position discount.
    # Numerator:   2^rel - 1  (exponential gain; amplifies highly-relevant items)
    # Denominator: log2(i+2)  (i is 0-based, so rank = i+1, discount = log2(rank+1))
    # Normalised by IDCG: the DCG of the ideal ranking (all relevant items first).

    # 1. Map retrieved IDs to their actual relevance scores
    gains = [relevance.get(pid, 0) for pid in retrieved_ids[:k]]

    # 2. Single DCG helper — used for both actual and ideal ranking
    def compute_dcg(scores: list[int]) -> float:
        return sum((2 ** score - 1) / math.log2(i + 2) for i, score in enumerate(scores))

    # 3. Compute DCG and IDCG using the same helper
    dcg_val      = compute_dcg(gains)
    ideal_gains  = sorted(relevance.values(), reverse=True)[:k]
    idcg_val     = compute_dcg(ideal_gains)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0
