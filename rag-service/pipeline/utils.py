# Shared helpers used across pipeline modules: singleton OpenAI client, single-text
# embedding with Redis cache, cosine similarity, and a Timer context manager.
#
# Embedding cache design:
#   Key : "emb:" + first 24 hex chars of SHA-256(query_text)
#   TTL : 1 hour — balances freshness with cache utility for repeated queries
#   Miss: falls through to OpenAI API; cache write failure is non-fatal
#   Hit : skips the ~300ms OpenAI round-trip entirely (~5ms Redis lookup)
#
# Redis failure policy — fail open:
#   If Redis is down, _get_redis() returns None and every call falls through to
#   the live OpenAI API. The service degrades gracefully to uncached behaviour
#   rather than refusing queries.

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import redis.asyncio as aioredis
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_openai_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


# ---------------------------------------------------------------------------
# Redis client (lazy singleton)
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None
_redis_init_attempted: bool = False   # only log the connection failure once


def _get_redis() -> aioredis.Redis | None:
    """Return the Redis client, or None if Redis is unavailable.

    from_url() is synchronous and only creates the client object — the actual
    TCP connection happens on the first command. We catch that lazily inside
    embed() so startup is never blocked by Redis being down.
    """
    global _redis, _redis_init_attempted
    if _redis is None and not _redis_init_attempted:
        _redis_init_attempted = True
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            _redis = aioredis.from_url(url, decode_responses=True)
        except Exception as exc:
            logger.warning("Redis client init failed (%s); embedding cache disabled", exc)
    return _redis


# ---------------------------------------------------------------------------
# Embedding (with Redis cache)
# ---------------------------------------------------------------------------

_EMBED_CACHE_TTL = 3600   # seconds — 1 hour
_EMBED_MODEL     = "text-embedding-3-small"


def _cache_key(text: str) -> str:
    """Deterministic Redis key for a query string.

    We use the first 24 hex chars of the SHA-256 digest (96-bit prefix).
    Collision probability is negligible for a fashion product search catalog
    and the key stays short for Redis memory efficiency.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()[:24]
    return f"emb:{digest}"


async def embed(text: str) -> list[float]:
    """Embed text with text-embedding-3-small, reading/writing a Redis cache.

    Cache hit  → returns vector from Redis in ~5ms (no OpenAI call).
    Cache miss → calls OpenAI (~300ms), writes result to Redis.
    Redis down → falls through to OpenAI on every call (graceful degradation).
    """
    key = _cache_key(text)
    r = _get_redis()

    # ── 1. Try cache ────────────────────────────────────────────────────────
    if r is not None:
        try:
            cached = await r.get(key)
            if cached is not None:
                return json.loads(cached)
        except Exception as exc:
            # Redis read failure — log once at DEBUG, fall through to API.
            logger.debug("Redis cache read failed: %s", exc)

    # ── 2. Call OpenAI API ──────────────────────────────────────────────────
    response = await get_openai_client().embeddings.create(
        model=_EMBED_MODEL,
        input=text,
    )
    vector: list[float] = response.data[0].embedding

    # ── 3. Write cache (non-fatal) ──────────────────────────────────────────
    if r is not None:
        try:
            await r.set(key, json.dumps(vector), ex=_EMBED_CACHE_TTL)
        except Exception as exc:
            logger.debug("Redis cache write failed: %s", exc)

    return vector


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

# Euclidean norm /magnitude (or length) = squaring each individual number in the vector,
# adding them all together, and taking the square root
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that records elapsed milliseconds in self.ms."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
