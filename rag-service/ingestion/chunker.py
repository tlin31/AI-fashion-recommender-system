# Splits product text into indexable chunks. Descriptions use a sentence-aware sliding
# window (256 tokens, ~50-token overlap); reviews are stored as single atomic chunks
# and dropped if shorter than min_words.

from __future__ import annotations

import nltk
import tiktoken

# Download punkt tokenizer data on first import (no-op if already present)
nltk.download("punkt_tab", quiet=True)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_description(text: str, max_tokens: int = 256, overlap_tokens: int = 50) -> list[str]:
    """Sentence-aware sliding window over a product description.

    Builds chunks by accumulating sentences until the token budget is hit,
    then starts the next chunk by carrying the last ~overlap_tokens worth of
    sentences forward. This avoids splitting mid-sentence, which would break
    the semantic units the embedding model relies on.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        # Single sentence longer than the window — emit it alone rather than skipping it
        if sent_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
            chunks.append(sent)
            current = []
            current_tokens = 0
            continue

        if current_tokens + sent_tokens > max_tokens:
            chunks.append(" ".join(current))

            # Carry the tail of the current chunk forward for overlap.
            # Walk backwards through sentences until we've accumulated ~overlap_tokens.
            overlap: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = _count_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_count += t

            current = overlap
            current_tokens = overlap_count

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    # Drop any chunk that is too short to be meaningful (e.g. a stray punctuation sentence)
    return [c for c in chunks if _count_tokens(c) >= 10]


def chunk_review(text: str, min_words: int = 20) -> str | None:
    """Reviews are semantically atomic — one chunk per review.

    Returns None if the review is below min_words so the caller can skip it
    at ingestion time without a separate filtering step.
    """
    if len(text.split()) < min_words:
        return None
    return text.strip()
