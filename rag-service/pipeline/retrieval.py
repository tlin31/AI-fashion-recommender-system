# Hybrid retrieval combining BM25 sparse search and Milvus HNSW dense search.
# Results from both are fused with Reciprocal Rank Fusion (RRF, k=60) into a
# single ranked candidate list. Metadata pre-filters are applied inside Milvus
# before ANN search, not as a post-filter.
#
# Design decisions:
#   Pre-filter vs post-filter: Pre-filter reduces the ANN search space before
#     computing distances — 3-5x faster on large catalogs. Post-filter can
#     silently miss results when the ANN top-K is dominated by filtered-out docs.
#
#   RRF over weighted sum: Weighted sum (α·dense + (1-α)·sparse) requires
#     per-query tuning of α and is sensitive to score scale differences between
#     BM25 and cosine. RRF is parameter-free, scale-invariant, and empirically
#     matches or beats weighted sum at typical catalog sizes.
#
#   BM25 at product level, dense at chunk level: BM25 excels at exact keyword
#     matching (brand names, model numbers) over concise product names.
#     Milvus operates on chunk-level embeddings for semantic coverage. Chunks
#     inherit their product's BM25 rank for the sparse RRF contribution.
#
#   BM25 score threshold: Only products whose raw BM25 score is ≥ 5% of the
#     top score receive an RRF boost. Products with near-zero BM25 scores have
#     no meaningful keyword match; boosting them adds noise without improving
#     recall. This does NOT fix semantic vocabulary mismatches (e.g. "minimalist"
#     matching wallets) — that is CRAG's job via query rewriting.
#
# Resilience strategy:
#   embed() failure   → log warning, return [] (CRAG handles fallback)
#   Milvus failure    → log warning, return [] (CRAG handles fallback)
#   embed() timeout   → caught by outer except, treated same as failure
#   Postgres failure  → tenacity retry x3 with exponential backoff; reraise

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import psycopg2
import tiktoken
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.utils import embed

logger = logging.getLogger(__name__)

_POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://gorse:gorse_pass@localhost:5432/gorse",
)

# Number of candidates fetched from each retrieval path before fusion.
_CANDIDATE_POOL = 50

# RRF smoothing constant (Cormack et al., 2009).
_RRF_K = 60

# text-embedding-3-small hard limit is 8191 tokens. We stay safely under it.
# 向 OpenAI 发送查询时的最大 Token 上限（text-embedding-3-small 最多支持 8191 个 Token）
_EMBED_MAX_TOKENS = 8000

# Hard wall-clock cap on the embed() API call inside hybrid_search.
# Guards against hung TCP connections (OpenAI drops without closing).
_EMBED_TIMEOUT_S = 5.0

# Maximum chunks from a single product in the final ranked list.
# Prevents a product with many review chunks from monopolising results.
# 单个商品在最终结果中最多出现的 Chunk 数量，防止同一商品霸占全部名额
_MAX_CHUNKS_PER_PRODUCT = 2

# BM25 score threshold: a product must score at least this fraction of the
# top BM25 score to receive a sparse RRF boost. Eliminates near-zero matches
# that add rank noise without a real keyword relationship to the query.
_BM25_MIN_SCORE_RATIO = 0.05

# Milvus metadata fields that are legal filter keys.
# Allowlisting prevents both typos and expression-injection.
_ALLOWED_FILTER_FIELDS = frozenset(
    {"price_range", "category", "occasion", "brand", "chunk_type"}
)

# Lazily initialised tiktoken encoder (shared across calls — encoding load is ~50ms)
_tiktoken_enc: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _tiktoken_enc
    if _tiktoken_enc is None:
        # cl100k_base is the encoding used by text-embedding-3-small
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer.

    Deliberately simple: BM25Okapi uses the same tokens at index and query time,
    so consistency matters more than linguistic sophistication. A more advanced
    tokenizer (e.g. NLTK word_tokenize) can be swapped in without changing any
    other code, provided build_bm25_index and hybrid_search both call this
    function.
    """
    return text.lower().split()


# ---------------------------------------------------------------------------
# Query validation & sanitisation
# ---------------------------------------------------------------------------

def _validate_and_sanitise_query(query: str) -> str:
    """Strip whitespace; return empty string if the query is blank.

    Callers check for empty return and short-circuit before embed() to avoid
    an openai.BadRequestError on empty input.
    """
    return query.strip()


def _truncate_query(query: str) -> str:
    """Truncate the query to _EMBED_MAX_TOKENS tokens.

    text-embedding-3-small rejects inputs over 8191 tokens with a 400 error.
    We truncate silently and log a warning so very long inputs degrade
    gracefully rather than crashing.
    """
    enc = _get_encoder()
    tokens = enc.encode(query)
    if len(tokens) <= _EMBED_MAX_TOKENS:
        return query
    logger.warning(
        "Query truncated from %d tokens to %d before embedding",
        len(tokens),
        _EMBED_MAX_TOKENS,
    )
    return enc.decode(tokens[:_EMBED_MAX_TOKENS])


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

# PostgreSQL retry: up to 3 attempts with exponential backoff (1s, 2s, 4s).
# reraise=True surfaces the final OperationalError so main.py lifespan can
# decide whether to abort startup or start degraded.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _build_bm25_sync() -> tuple[BM25Okapi, list[str]]:
    """Synchronous helper — runs in a thread-pool executor to avoid blocking
    the event loop during the potentially slow PostgreSQL fetch + index build.
    """
    # 向 PostgreSQL 数据库发起并建立一个同步的物理网络连接（注意：此操作是阻塞的）。
    conn = psycopg2.connect(_POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            # Concatenate (||) name and description so BM25 covers both brand/model
            # keywords (name) and detailed feature keywords (description).
            cur.execute(
                """
                SELECT product_id,
                       COALESCE(name, '') || ' ' || COALESCE(description, '') AS text
                FROM   rag_products
                ORDER  BY product_id
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        # Return an empty-but-valid index so callers don't need to handle None.
        return BM25Okapi([[""]]), []

    # Deduplicate product_ids (guard against re-import without UPSERT).
    # Keep first occurrence to preserve ORDER BY product_id ranking.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for product_id, text in rows:
        if product_id not in seen:
            seen.add(product_id)
            deduped.append((product_id, text))

    if len(deduped) < len(rows):
        logger.warning(
            "BM25 index: dropped %d duplicate product_id rows from rag_products",
            len(rows) - len(deduped),
        )

    product_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []

    for product_id, text in deduped:
        # 特征工程（Feature Engineering）。BM25 算法不认识人类的整段话，它必须吃分割好的词（Tokens）
        tokens = _tokenize(text)
        # Skip products with no usable text (both name and description are empty).
        # An empty token list causes BM25 to silently give the product a zero score
        # on every query, making it permanently invisible.
        if not tokens:
            logger.warning(
                "BM25 index: skipping product_id=%r — name and description are both empty",
                product_id,
            )
            continue
        product_ids.append(product_id)
        tokenized_corpus.append(tokens)

    if not tokenized_corpus:
        return BM25Okapi([[""]]), []

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, product_ids


async def build_bm25_index() -> tuple[BM25Okapi, list[str]]:
    """异步公共接口
    Load all (product_id, text) rows from PostgreSQL and build an in-memory
    BM25 index.

    Called once at startup (see main.py lifespan). ~5 K products ≈ negligible
    RAM. The blocking psycopg2 call is offloaded to a thread-pool executor so
    it does not stall the async event loop during startup.

    Returns:
        (bm25_index, corpus_product_ids) where corpus_product_ids[i] is the
        product_id corresponding to the i-th document in the BM25 corpus.

    Raises:
        psycopg2.OperationalError: if PostgreSQL is unreachable after 3 retries.
    """
    # 拿到当前线程中正在跑的唯一事件循环管理器（Event Loop 实例），为接下来的线程调度做准备。
    loop = asyncio.get_event_loop()
    # loop.run_in_executor(executor, func, *args)：这是 asyncio 桥接同步阻塞代码的核心底层武器。
    # 第一个参数 None：告诉 Python "请直接使用系统默认内置的线程池（ThreadPoolExecutor）"。
    # 第二个参数 _build_bm25_sync：要扔给线程池跑的那个同步阻塞函数的名字（函数指针）。
    #          注意绝对不能写括号 ()，因为我们不是要在当前主线程立刻执行它，而是交给线程池由其他独立的线程去执行。
    return await loop.run_in_executor(None, _build_bm25_sync)


# ---------------------------------------------------------------------------
# Filter expression builder
# ---------------------------------------------------------------------------

def _build_filter_expr(filters: dict | None) -> str:
    """Translate a filter dict into a Milvus boolean expression string.

    Only known metadata fields are accepted; unknown keys are silently ignored
    to prevent expression injection and surface clear errors at integration time.

    Example:
        {"price_range": "budget", "category": "tops"}
        → 'price_range == "budget" && category == "tops"'
    """
    if not filters:
        return ""

    parts: list[str] = []
    for field, value in filters.items():
        if field not in _ALLOWED_FILTER_FIELDS:
            continue
        # Escape backslashes first, then double-quotes, then single-quotes.
        # Milvus expression syntax accepts both " and ' as string delimiters,
        # so an unescaped ' in "O'Reilly" terminates the string early.
        safe_val = (
            str(value)
            .replace("\\", "\\\\")   # must come first
            .replace('"', '\\"')
            .replace("'", "\\'")
        )
        parts.append(f'{field} == "{safe_val}"')

    return " && ".join(parts)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

async def hybrid_search(
    query: str,
    filters: dict | None,
    top_k: int = 20,
    bm25_index: tuple[BM25Okapi, list[str]] | None = None,
    milvus_client: Any = None,
    collection_name: str | None = None,
) -> list[dict]:
    """Retrieve top-k fashion product chunks using hybrid BM25 + dense search.

    Pipeline:
        1. Dense path  — embed query via text-embedding-3-small (5s timeout),
                         search Milvus HNSW with metadata pre-filter (MilvusClient
                         API), take top-50 chunks.
        2. Sparse path — tokenize query, score all products with BM25, take
                         top-50 products whose score ≥ 5% of the max BM25 score.
        3. RRF fusion  — each chunk's RRF score = Σ 1/(k + rank) summed over
                         the lists it appears in. Chunks inherit their product's
                         BM25 rank for the sparse contribution.

    Failure modes:
        embed() error / timeout → returns [] so CRAG falls back to trending.
        milvus search error     → returns [] so CRAG falls back to trending.
        Empty query             → returns [] immediately (no API call made).

    Args:
        query:           Free-text user query.
        filters:         Optional metadata pre-filter dict applied inside Milvus
                         ANN search. Supported keys: price_range, category,
                         occasion, brand, chunk_type. Unknown keys ignored.
        top_k:           Number of results to return (default 20).
        bm25_index:      (BM25Okapi, corpus_product_ids) from build_bm25_index().
        milvus_client:   pymilvus MilvusClient instance.
        collection_name: Milvus collection to search. Defaults to the
                         MILVUS_COLLECTION env var ("fashion_rag").

    Returns:
        List of up to top_k dicts:
            {
                "chunk_id":   str,   # Milvus primary key (chunk UUID)
                "product_id": str,   # FK to PostgreSQL rag_products
                "text":       str,   # raw chunk text
                "score":      float, # RRF fusion score (higher = better)
                "metadata":   dict,  # chunk_type, price_range, category,
                                     # occasion, brand
            }
        Sorted descending by score. Returns [] on empty query or service failure.
    """
    if bm25_index is None or milvus_client is None:
        return []

    if collection_name is None:
        collection_name = os.environ.get("MILVUS_COLLECTION", "fashion_rag")

    # Empty/whitespace-only query guard — must happen before embed().
    # OpenAI raises BadRequestError on "".
    query = _validate_and_sanitise_query(query)
    if not query:
        logger.warning("hybrid_search received an empty query; returning []")
        return []

    # Truncate before embed() to stay within the 8191-token limit.
    query = _truncate_query(query)

    bm25, corpus_product_ids = bm25_index

    # ── 1. Dense path ──────────────────────────────────────────────────────
    # Catch embed() failures (OpenAI outage, rate-limit, auth error, timeout).
    # asyncio.TimeoutError from wait_for is a subclass of Exception and is
    # caught here, logged, and treated as a service failure → return [].
    dense_hits: list[dict] = []
    try:
        # Hard 5s timeout guards against hung TCP connections to OpenAI.
        query_vector = await asyncio.wait_for(embed(query), timeout=_EMBED_TIMEOUT_S)

        expr = _build_filter_expr(filters)
        search_params = {
            "metric_type": "COSINE",
            # ef controls recall vs latency trade-off for HNSW at query time.
            # ef=128 gives >99% recall for most catalog sizes.
            "params": {"ef": 128},
        }

        # Catch Milvus failures (collection evicted, network drop, OOM).
        try:
            # MilvusClient API — replaces deprecated ORM Collection.search().
            # Key differences from ORM style:
            #   param=    → search_params=
            #   expr=     → filter=  (empty string = no filter)
            #   hit.id    → hit["id"]
            #   hit.score → hit["distance"]
            #   hit.entity.get() → hit["entity"].get()
            milvus_results = milvus_client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                search_params=search_params,
                limit=_CANDIDATE_POOL,
                filter=expr,          # "" means no filter in MilvusClient
                output_fields=[
                    "product_id", "text", "chunk_type",
                    "price_range", "category", "occasion", "brand",
                ],
            )

            for hit in milvus_results[0]:
                dense_hits.append(
                    {
                        "chunk_id":    hit["id"],
                        "product_id":  hit["entity"].get("product_id", ""),
                        "text":        hit["entity"].get("text", ""),
                        "score":       float(hit["distance"]),  # overwritten by RRF
                        "milvus_score": float(hit["distance"]), # preserved for CRAG grader
                        "metadata": {
                            "chunk_type":  hit["entity"].get("chunk_type", ""),
                            "price_range": hit["entity"].get("price_range", ""),
                            "category":    hit["entity"].get("category", ""),
                            "occasion":    hit["entity"].get("occasion", ""),
                            "brand":       hit["entity"].get("brand", ""),
                        },
                    }
                )

        except Exception as milvus_err:
            logger.error(
                "Milvus search failed (collection may be unloaded); "
                "returning [] so CRAG falls back to trending. Error: %s",
                milvus_err,
            )
            return []

    except Exception as embed_err:
        logger.error(
            "Dense path failed (embed timeout or API error); "
            "returning [] so CRAG falls back to trending. Error: %s",
            embed_err,
        )
        return []

    # ── 2. Sparse path ─────────────────────────────────────────────────────
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)  # ndarray, length = corpus size

    # Rank products by BM25 score (descending) and take top _CANDIDATE_POOL.
    indexed = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    top_bm25 = indexed[:_CANDIDATE_POOL]

    # BM25 score threshold: skip products below 5% of the top score.
    # Eliminates near-zero keyword matches that add RRF noise without a real
    # relationship to the query. Does NOT prevent vocabulary mismatch (e.g.
    # "minimalist" matching wallet names) — CRAG handles that via query rewriting.
    #
    # Guard: when max_bm25_score == 0, no product contains any query token.
    # Applying the relative threshold (0% of 0 = 0) would include every product,
    # giving all of them an equal but meaningless BM25 boost. Instead, skip
    # sparse boosting entirely so only the dense path influences rankings.
    max_bm25_score = top_bm25[0][1] if top_bm25 else 0.0

    bm25_product_rank: dict[str, int] = {}
    if max_bm25_score > 0:
        min_bm25_score = max_bm25_score * _BM25_MIN_SCORE_RATIO
        for rank, (corpus_idx, score) in enumerate(top_bm25, start=1):
            if score < min_bm25_score:
                break   # list is sorted descending — all remaining are also below threshold
            pid = corpus_product_ids[corpus_idx]
            if pid not in bm25_product_rank:
                bm25_product_rank[pid] = rank

    # ── 3. RRF fusion ──────────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for dense_rank, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        pid = hit["product_id"]

        rrf = 1.0 / (_RRF_K + dense_rank)

        if pid in bm25_product_rank:
            rrf += 1.0 / (_RRF_K + bm25_product_rank[pid])

        rrf_scores[cid] = rrf
        chunk_data[cid] = hit

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Per-product chunk cap: at most _MAX_CHUNKS_PER_PRODUCT chunks from the
    # same product in the final list. Prevents a product with many review chunks
    # from monopolising the top-20.
    results: list[dict] = []
    chunks_per_product: dict[str, int] = {}

    for cid, score in ranked:
        hit = chunk_data[cid]
        pid = hit["product_id"]
        if chunks_per_product.get(pid, 0) >= _MAX_CHUNKS_PER_PRODUCT:
            continue
        chunks_per_product[pid] = chunks_per_product.get(pid, 0) + 1
        results.append({**hit, "score": round(score, 6)})
        if len(results) == top_k:
            break

    if len(results) < top_k:
        logger.info(
            "hybrid_search returned %d results (requested top_k=%d); "
            "catalog may be too small for the applied filters.",
            len(results),
            top_k,
        )

    return results
