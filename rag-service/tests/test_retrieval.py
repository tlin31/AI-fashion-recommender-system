# Tests for pipeline/retrieval.py.
# No external services required — PostgreSQL and Milvus are fully mocked.
#
# Milvus mock format: MilvusClient.search() returns List[List[dict]] where
# each hit dict is {"id": str, "distance": float, "entity": {field: value}}.
# This matches the pymilvus MilvusClient API (not the deprecated ORM Collection).

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from pipeline.retrieval import (
    _BM25_MIN_SCORE_RATIO,
    _MAX_CHUNKS_PER_PRODUCT,
    _RRF_K,
    _build_filter_expr,
    _tokenize,
    _truncate_query,
    _validate_and_sanitise_query,
    build_bm25_index,
    hybrid_search,
)

# ---------------------------------------------------------------------------
# Helpers: MilvusClient hit factory
# ---------------------------------------------------------------------------

def _make_hit(
    chunk_id: str,
    product_id: str,
    text: str,
    score: float,
    *,
    price_range: str = "mid",
    category: str = "tops",
    occasion: str = "casual",
    brand: str = "",
    chunk_type: str = "description",
) -> dict:
    """Return a dict matching the MilvusClient search result format.

    MilvusClient returns List[List[dict]] where each hit is:
        {"id": str, "distance": float, "entity": {output_field: value, ...}}
    """
    return {
        "id": chunk_id,
        "distance": score,
        "entity": {
            "product_id":  product_id,
            "text":        text,
            "chunk_type":  chunk_type,
            "price_range": price_range,
            "category":    category,
            "occasion":    occasion,
            "brand":       brand,
        },
    }


def _make_milvus_client(hits: list[dict]) -> MagicMock:
    """Return a mock MilvusClient whose search() returns one hit-list."""
    client = MagicMock()
    client.search.return_value = [hits]
    return client


def _make_raising_milvus_client(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.search.side_effect = exc
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CORPUS = [
    ("nike_001",  "Nike Air Max 270 running shoe lightweight breathable"),
    ("adidas_001","Adidas Ultraboost trainer performance sport shoe"),
    ("zara_001",  "Zara linen blazer minimalist office formal wear"),
    ("hm_001",    "H&M cotton budget t-shirt casual everyday basics"),
    ("gucci_001", "Gucci premium leather handbag luxury accessory"),
]


@pytest.fixture
def bm25_fixture() -> tuple[BM25Okapi, list[str]]:
    product_ids = [row[0] for row in SMALL_CORPUS]
    corpus      = [_tokenize(row[1]) for row in SMALL_CORPUS]
    return BM25Okapi(corpus), product_ids


@pytest.fixture
def dummy_vector() -> list[float]:
    return [0.0] * 1536


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    assert _tokenize("Nike AIR MAX") == ["nike", "air", "max"]


def test_tokenize_splits_on_whitespace():
    assert _tokenize("blue  linen  trousers") == ["blue", "linen", "trousers"]


def test_tokenize_empty_string():
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _validate_and_sanitise_query
# ---------------------------------------------------------------------------

def test_validate_strips_whitespace():
    assert _validate_and_sanitise_query("  hello  ") == "hello"


def test_validate_empty_returns_empty():
    assert _validate_and_sanitise_query("") == ""


def test_validate_whitespace_only_returns_empty():
    assert _validate_and_sanitise_query("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# _truncate_query
# ---------------------------------------------------------------------------

def test_truncate_short_query_unchanged():
    q = "blue linen trousers"
    assert _truncate_query(q) == q


def test_truncate_long_query_shortened():
    # "the " is a single token in cl100k_base; 9000 reps → 9000 tokens > 8000 limit
    long_query = "the " * 9000
    assert len(_truncate_query(long_query)) < len(long_query)


# ---------------------------------------------------------------------------
# _build_filter_expr
# ---------------------------------------------------------------------------

def test_filter_expr_single_field():
    assert _build_filter_expr({"price_range": "budget"}) == 'price_range == "budget"'


def test_filter_expr_multiple_fields():
    expr = _build_filter_expr({"price_range": "budget", "category": "tops"})
    assert 'price_range == "budget"' in expr
    assert 'category == "tops"' in expr
    assert "&&" in expr


def test_filter_expr_unknown_key_ignored():
    expr = _build_filter_expr({"price_range": "budget", "unknown_field": "x"})
    assert "unknown_field" not in expr
    assert expr == 'price_range == "budget"'


def test_filter_expr_none_returns_empty():
    assert _build_filter_expr(None) == ""


def test_filter_expr_empty_dict_returns_empty():
    assert _build_filter_expr({}) == ""


def test_filter_expr_escapes_double_quotes():
    expr = _build_filter_expr({"brand": 'O"Reilly'})
    assert '\\"' in expr


def test_filter_expr_escapes_single_quotes():
    expr = _build_filter_expr({"brand": "O'Reilly"})
    assert "\\'" in expr


def test_filter_expr_escapes_backslash_first():
    expr = _build_filter_expr({"brand": '\\"'})
    assert '\\\\"' in expr


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------

def _patch_psycopg2(rows: list[tuple]):
    mock_conn = MagicMock()
    mock_cur  = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return patch("pipeline.retrieval.psycopg2.connect", return_value=mock_conn)


@pytest.mark.asyncio
async def test_build_bm25_index_returns_correct_types():
    with _patch_psycopg2(SMALL_CORPUS):
        bm25, product_ids = await build_bm25_index()
    assert isinstance(bm25, BM25Okapi)
    assert product_ids == [row[0] for row in SMALL_CORPUS]


@pytest.mark.asyncio
async def test_build_bm25_index_empty_table():
    with _patch_psycopg2([]):
        bm25, product_ids = await build_bm25_index()
    assert isinstance(bm25, BM25Okapi)
    assert product_ids == []


@pytest.mark.asyncio
async def test_build_bm25_index_deduplicates_product_ids():
    rows = [
        ("nike_001", "Nike Air Max original"),
        ("nike_001", "Nike Air Max duplicate"),
        ("zara_001", "Zara blazer"),
    ]
    with _patch_psycopg2(rows):
        bm25, product_ids = await build_bm25_index()
    assert product_ids.count("nike_001") == 1
    assert len(product_ids) == 2


@pytest.mark.asyncio
async def test_build_bm25_index_skips_empty_text_products():
    rows = [
        ("nike_001", "Nike Air Max"),
        ("empty_001", ""),
        ("zara_001", "Zara blazer"),
    ]
    with _patch_psycopg2(rows):
        bm25, product_ids = await build_bm25_index()
    assert "empty_001" not in product_ids
    assert len(product_ids) == 2


# ---------------------------------------------------------------------------
# hybrid_search — guard cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_none_index_returns_empty():
    client = _make_milvus_client([])
    result = await hybrid_search("any query", {}, bm25_index=None, milvus_client=client)
    assert result == []


@pytest.mark.asyncio
async def test_none_milvus_degrades_to_bm25_only(bm25_fixture):
    """Milvus down is a degradation, not an outage — BM25 needs no external service."""
    sparse = [{"chunk_id": "bm25_p1", "product_id": "p1", "text": "t",
               "score": 0.0, "milvus_score": 0.0, "metadata": {}}]
    degradations: list[str] = []
    with patch("pipeline.retrieval._bm25_only_chunks",
               new_callable=AsyncMock, return_value=sparse):
        result = await hybrid_search(
            "nike air max", {}, bm25_index=bm25_fixture, milvus_client=None,
            degradations=degradations,
        )
    assert result == sparse
    assert "milvus_unavailable" in degradations
    assert "bm25_only" in degradations


async def test_none_bm25_index_still_returns_empty(bm25_fixture):
    """Without the sparse index there is no fallback left."""
    result = await hybrid_search("any query", {}, bm25_index=None, milvus_client=None)
    assert result == []


async def test_degradation_not_recorded_on_healthy_path(bm25_fixture, dummy_vector):
    hits = [_make_hit("c1", "p1", "nike air max shoes", 0.9)]
    client = _make_milvus_client(hits)
    degradations: list[str] = []
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        await hybrid_search("nike air max", {}, bm25_index=bm25_fixture,
                            milvus_client=client, degradations=degradations)
    assert degradations == []


@pytest.mark.asyncio
async def test_hybrid_search_empty_query_returns_empty(bm25_fixture):
    client = _make_milvus_client([])
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock) as mock_embed:
        result = await hybrid_search("", filters={}, bm25_index=bm25_fixture, milvus_client=client)
    assert result == []
    mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_search_whitespace_query_returns_empty(bm25_fixture):
    client = _make_milvus_client([])
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock) as mock_embed:
        result = await hybrid_search("   ", filters={}, bm25_index=bm25_fixture, milvus_client=client)
    assert result == []
    mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# hybrid_search — network failure resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_failure_degrades_to_bm25_only(bm25_fixture):
    """An OpenAI outage kills the dense path but not the in-memory sparse index."""
    client = _make_milvus_client([])
    sparse = [{"chunk_id": "bm25_p1", "product_id": "p1", "text": "t",
               "score": 0.0, "milvus_score": 0.0, "metadata": {}}]
    degradations: list[str] = []
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock,
               side_effect=Exception("OpenAI 503")), \
         patch("pipeline.retrieval._bm25_only_chunks",
               new_callable=AsyncMock, return_value=sparse):
        result = await hybrid_search(
            "nike air max", filters={}, bm25_index=bm25_fixture,
            milvus_client=client, degradations=degradations,
        )
    assert result == sparse
    assert "embedding_failed" in degradations


@pytest.mark.asyncio
async def test_milvus_search_failure_degrades_to_bm25_only(bm25_fixture, dummy_vector):
    """A raising Milvus client degrades rather than emptying the result set."""
    client = _make_raising_milvus_client(Exception("collection not loaded"))
    sparse = [{"chunk_id": "bm25_p1", "product_id": "p1", "text": "t",
               "score": 0.0, "milvus_score": 0.0, "metadata": {}}]
    degradations: list[str] = []
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector), \
         patch("pipeline.retrieval._bm25_only_chunks",
               new_callable=AsyncMock, return_value=sparse):
        result = await hybrid_search(
            "nike air max", filters={}, bm25_index=bm25_fixture,
            milvus_client=client, degradations=degradations,
        )
    assert result == sparse
    assert "milvus_search_failed" in degradations


async def test_no_dense_and_no_bm25_match_returns_empty(bm25_fixture, dummy_vector):
    """Both retrievers empty-handed — nothing to degrade to."""
    client = _make_raising_milvus_client(Exception("collection not loaded"))
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        result = await hybrid_search(
            "zzzz nonexistent token qqqq", filters={},
            bm25_index=bm25_fixture, milvus_client=client,
        )
    assert result == []


@pytest.mark.asyncio
async def test_embed_timeout_returns_empty(bm25_fixture):
    """asyncio.TimeoutError from wait_for is caught and returns []."""
    import asyncio
    client = _make_milvus_client([])
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock,
               side_effect=asyncio.TimeoutError()):
        result = await hybrid_search(
            "blue trousers", filters={}, bm25_index=bm25_fixture, milvus_client=client
        )
    assert result == []


# ---------------------------------------------------------------------------
# hybrid_search — empty Milvus results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_empty_milvus_returns_empty(bm25_fixture, dummy_vector):
    client = _make_milvus_client([])
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        result = await hybrid_search(
            "blue linen trousers", filters={}, bm25_index=bm25_fixture, milvus_client=client
        )
    assert result == []


# ---------------------------------------------------------------------------
# hybrid_search — exact brand query (BM25 should boost)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exact_brand_query_in_top_results(bm25_fixture, dummy_vector):
    """Nike at dense rank 3 but BM25 rank 1 → should appear in top-2 after RRF."""
    hits = [
        _make_hit("chunk_zara",   "zara_001",   "Zara linen blazer",            0.55),
        _make_hit("chunk_adidas", "adidas_001", "Adidas Ultraboost",             0.45),
        _make_hit("chunk_nike",   "nike_001",   "Nike Air Max 270 running shoe", 0.30),
        _make_hit("chunk_hm",     "hm_001",     "H&M cotton t-shirt",            0.20),
        _make_hit("chunk_gucci",  "gucci_001",  "Gucci leather handbag",         0.10),
    ]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "Nike Air Max 270", filters={}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    product_ids = [r["product_id"] for r in results]
    assert "nike_001" in product_ids[:2], f"Expected nike_001 in top-2, got: {product_ids}"


# ---------------------------------------------------------------------------
# hybrid_search — semantic query (dense should dominate)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_query_dense_dominates(bm25_fixture, dummy_vector):
    """No keyword overlap → dense rank 1 (zara_001) wins."""
    hits = [
        _make_hit("chunk_zara", "zara_001", "Zara linen blazer minimalist office", 0.82),
        _make_hit("chunk_hm",   "hm_001",   "H&M cotton t-shirt casual basics",    0.41),
        _make_hit("chunk_nike", "nike_001", "Nike Air Max running shoe",            0.22),
    ]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "something minimalist for a job interview", filters={}, top_k=3,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    assert results[0]["product_id"] == "zara_001", (
        f"Expected zara_001 at rank 1, got: {results[0]['product_id']}"
    )


# ---------------------------------------------------------------------------
# hybrid_search — filtered query (MilvusClient pre-filter)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filtered_query_passes_filter_to_milvus(bm25_fixture, dummy_vector):
    """Filter dict must be forwarded to MilvusClient.search() as 'filter' kwarg."""
    hits = [_make_hit("chunk_hm", "hm_001", "H&M budget t-shirt", 0.75, price_range="budget")]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        await hybrid_search(
            "casual t-shirt", filters={"price_range": "budget"}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    # MilvusClient uses 'filter=', not 'expr='
    call_kwargs = client.search.call_args.kwargs
    filter_used = call_kwargs.get("filter", "")
    assert 'price_range == "budget"' in filter_used, (
        f"Filter not forwarded to Milvus. Got: {filter_used!r}"
    )


@pytest.mark.asyncio
async def test_filtered_query_result_metadata(bm25_fixture, dummy_vector):
    hits = [_make_hit("chunk_hm", "hm_001", "H&M budget basics", 0.75, price_range="budget")]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "affordable top", filters={"price_range": "budget"}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    assert results[0]["metadata"]["price_range"] == "budget"


# ---------------------------------------------------------------------------
# hybrid_search — RRF score formula
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rrf_score_formula(bm25_fixture, dummy_vector):
    """Chunk at dense_rank=1 with product at bm25_rank=1 → score = 2/(k+1)."""
    hits = [_make_hit("chunk_nike", "nike_001", "Nike Air Max 270", 0.90)]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "Nike Air Max 270", filters={}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    expected = round(1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 1), 6)
    assert math.isclose(results[0]["score"], expected, rel_tol=1e-5), (
        f"Expected {expected}, got {results[0]['score']}"
    )


# ---------------------------------------------------------------------------
# hybrid_search — BM25 score threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bm25_threshold_excludes_near_zero_matches(bm25_fixture, dummy_vector):
    """Products with near-zero BM25 scores must not get an RRF boost.

    Uses SMALL_CORPUS (5 docs) so BM25Okapi produces non-zero IDF values.
    With a 2-doc corpus every token has IDF = log(1.5) - log(1.5) = 0, making
    all BM25 scores zero and the threshold meaningless.

    Setup:
      - Query: "nike air max 270"
      - "nike_001" is the only corpus entry containing those tokens → BM25 rank 1
      - "zara_001" has zero BM25 score (no matching tokens)
      - Milvus returns zara_001 at dense rank 1 (score 0.90) and nike_001 at rank 2

    Expected: nike_001 overtakes zara_001 via BM25 boost despite lower dense rank.
    If the threshold were broken and zara_001 also got a BM25 boost, the tie
    would resolve by insertion order and zara_001 might stay first.
    """
    hits = [
        # zara_001: dense rank 1, but ZERO BM25 score on "nike air max 270"
        _make_hit("chunk_zara", "zara_001", "Zara linen blazer formal office", 0.90),
        # nike_001: dense rank 2, BM25 rank 1 (all query tokens in its corpus entry)
        _make_hit("chunk_nike", "nike_001", "Nike Air Max 270 running shoe",    0.60),
    ]
    client = _make_milvus_client(hits)

    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "nike air max 270", filters={}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )

    product_ids = [r["product_id"] for r in results]
    assert product_ids[0] == "nike_001", (
        f"Expected nike_001 at rank 1 (BM25 boosted from rank 2), got: {product_ids}"
    )


# ---------------------------------------------------------------------------
# hybrid_search — per-product chunk cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_product_chunk_cap(bm25_fixture, dummy_vector):
    """No product should appear more than _MAX_CHUNKS_PER_PRODUCT times."""
    hits = [
        _make_hit(f"chunk_nike_{i}", "nike_001", f"Nike chunk {i}", 0.9 - i * 0.05)
        for i in range(6)
    ]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "Nike Air Max", filters={}, top_k=20,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    nike_count = sum(1 for r in results if r["product_id"] == "nike_001")
    assert nike_count <= _MAX_CHUNKS_PER_PRODUCT


# ---------------------------------------------------------------------------
# hybrid_search — top_k cap and result schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_top_k_limits_results(bm25_fixture, dummy_vector):
    hits = [
        _make_hit(f"chunk_{i}", f"product_{i:03d}", f"product text {i}", 1.0 - i * 0.05)
        for i in range(10)
    ]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "generic query", filters={}, top_k=3,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_result_contains_required_fields(bm25_fixture, dummy_vector):
    hits = [
        _make_hit("chunk_zara", "zara_001", "Zara blazer", 0.70,
                  price_range="mid", category="tops", occasion="formal", brand="zara")
    ]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        results = await hybrid_search(
            "formal blazer", filters={}, top_k=5,
            bm25_index=bm25_fixture, milvus_client=client,
        )
    assert results
    r = results[0]
    assert set(r.keys()) >= {"chunk_id", "product_id", "text", "score", "metadata"}
    assert set(r["metadata"].keys()) >= {
        "chunk_type", "price_range", "category", "occasion", "brand"
    }


# ---------------------------------------------------------------------------
# hybrid_search — short-result logging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_short_result_set_logs_info(bm25_fixture, dummy_vector, caplog):
    import logging
    hits = [_make_hit("chunk_zara", "zara_001", "Zara blazer", 0.70)]
    client = _make_milvus_client(hits)
    with patch("pipeline.retrieval.embed", new_callable=AsyncMock, return_value=dummy_vector):
        with caplog.at_level(logging.INFO, logger="pipeline.retrieval"):
            results = await hybrid_search(
                "formal blazer", filters={}, top_k=10,
                bm25_index=bm25_fixture, milvus_client=client,
            )
    assert len(results) < 10
    assert any("returned" in msg and "requested" in msg for msg in caplog.messages)
