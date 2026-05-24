# Unit tests for ingestion/chunker.py.
# No external services needed — pure text processing logic.

import pytest
from ingestion.chunker import chunk_description, chunk_review


# ── chunk_description ─────────────────────────────────────────────────────────

def test_short_text_returns_single_chunk():
    # Must be >= 10 tokens to pass the minimum-length guard in chunk_description
    text = "A beautiful blue kurta with floral prints, delicate embroidery, and a relaxed comfortable silhouette."
    chunks = chunk_description(text)
    assert len(chunks) == 1
    assert "blue kurta" in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    # Build a text that is definitely longer than 256 tokens
    sentence = "This is a high-quality cotton kurta with intricate embroidery and a comfortable fit. "
    text = sentence * 30
    chunks = chunk_description(text)
    assert len(chunks) > 1


def test_no_chunk_exceeds_max_tokens():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    sentence = "A stylish garment with fine stitching, premium fabric, and elegant design. "
    text = sentence * 40
    chunks = chunk_description(text, max_tokens=256)
    for chunk in chunks:
        # Allow a small buffer for a single overlong sentence
        assert len(enc.encode(chunk)) <= 300


def test_chunks_do_not_split_mid_sentence():
    """Each chunk should end at a sentence boundary, not mid-word."""
    sentences = [
        "This kurta is made from premium cotton fabric.",
        "It features intricate hand embroidery on the neckline.",
        "The flared hem adds a graceful silhouette.",
        "Perfect for festive occasions and casual wear.",
    ]
    # Repeat to force splitting
    text = " ".join(sentences * 10)
    chunks = chunk_description(text, max_tokens=64, overlap_tokens=10)
    for chunk in chunks:
        # Each chunk should end with sentence-ending punctuation
        assert chunk.rstrip()[-1] in ".!?", f"Chunk does not end at sentence boundary: ...{chunk[-30:]!r}"


def test_overlap_carries_context_forward():
    """Consecutive chunks should share some tokens (the overlap)."""
    sentence = "The garment features premium fabric with a modern cut. "
    text = sentence * 20
    chunks = chunk_description(text, max_tokens=80, overlap_tokens=20)
    if len(chunks) >= 2:
        # The end of chunk N and start of chunk N+1 should share some words
        words_end = set(chunks[0].split()[-10:])
        words_start = set(chunks[1].split()[:10])
        assert len(words_end & words_start) > 0, "No overlap found between consecutive chunks"


def test_empty_text_returns_empty_list():
    assert chunk_description("") == []
    assert chunk_description("   ") == []


def test_single_overlong_sentence_returned_as_own_chunk():
    long_sentence = ("word " * 300).strip()
    chunks = chunk_description(long_sentence)
    assert len(chunks) >= 1


# ── chunk_review ──────────────────────────────────────────────────────────────

def test_review_returned_as_single_chunk():
    # Must be >= 20 words (default min_words) to pass the length guard
    review = (
        "I absolutely love this kurta. The fabric is incredibly soft and the embroidery "
        "is absolutely stunning. The fit is perfect and the color is vibrant. Highly recommend!"
    )
    result = chunk_review(review)
    assert result == review.strip()


def test_short_review_dropped():
    result = chunk_review("Nice product.")
    assert result is None


def test_review_exactly_at_min_words():
    words = ["great"] * 20
    review = " ".join(words)
    result = chunk_review(review, min_words=20)
    assert result is not None


def test_review_one_below_min_words():
    words = ["great"] * 19
    review = " ".join(words)
    result = chunk_review(review, min_words=20)
    assert result is None


def test_review_text_is_stripped():
    # >= 20 words and has leading/trailing whitespace to verify stripping
    review = "  Great product, very comfortable and well-made. Worth every penny spent on this item. The quality is exceptional and delivery was fast.  "
    result = chunk_review(review)
    assert result == review.strip()
