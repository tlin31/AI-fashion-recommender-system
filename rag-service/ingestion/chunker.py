# Splits product text into indexable chunks. Descriptions use a sentence-aware sliding
# window (256 tokens, ~50-token overlap); reviews are stored as single atomic chunks
# and dropped if shorter than min_words.

from __future__ import annotations


def chunk_description(text: str, max_tokens: int = 256, overlap_tokens: int = 50) -> list[str]:
    # Sentence-aware sliding window using nltk.sent_tokenize + tiktoken for counting.
    # Carries last ~overlap_tokens into the next chunk to preserve context at boundaries.
    raise NotImplementedError


def chunk_review(text: str, min_words: int = 20) -> str | None:
    # Reviews are semantically atomic — one chunk per review.
    # Returns None if review is below min_words (dropped at ingestion).
    raise NotImplementedError
