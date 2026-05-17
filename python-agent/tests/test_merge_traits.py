"""Pure unit tests for _merge_trait_updates().

No I/O, no mocks, no LangGraph — just the merge logic.
All 7 TraitsData fields are exercised (mirroring Go models.go TraitsData struct).
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from agent.graph import _merge_trait_updates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing(**kwargs) -> dict:
    """Build an existing-traits dict with sensible zero defaults."""
    return {
        "style_preferences": kwargs.get("style_preferences", {}),
        "color_preferences": kwargs.get("color_preferences", {}),
        "price_sensitivity": kwargs.get("price_sensitivity", ""),
        "brand_preferences": kwargs.get("brand_preferences", []),
        "occasions": kwargs.get("occasions", []),
        "keywords": kwargs.get("keywords", []),
        "interests": kwargs.get("interests", []),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_updates_returns_existing_unchanged():
    existing = _existing(style_preferences={"minimalist": 0.8}, price_sensitivity="low")
    result = _merge_trait_updates(existing, [])
    assert result["style_preferences"] == {"minimalist": 0.8}
    assert result["price_sensitivity"] == "low"


def test_style_score_overwrite():
    """A staged score for an existing style key replaces the old score."""
    existing = _existing(style_preferences={"minimalist": 0.5, "casual": 0.7})
    updates = [{"style_preferences": {"minimalist": 0.95}}]
    result = _merge_trait_updates(existing, updates)
    assert result["style_preferences"]["minimalist"] == pytest.approx(0.95)
    # Untouched key is preserved
    assert result["style_preferences"]["casual"] == pytest.approx(0.7)


def test_color_new_key_added():
    """A staged update for a colour not yet in existing should be added."""
    existing = _existing(color_preferences={"black": 0.9})
    updates = [{"color_preferences": {"white": 0.6}}]
    result = _merge_trait_updates(existing, updates)
    assert result["color_preferences"]["black"] == pytest.approx(0.9)
    assert result["color_preferences"]["white"] == pytest.approx(0.6)


def test_price_sensitivity_override():
    """The most recent non-empty price_sensitivity wins."""
    existing = _existing(price_sensitivity="high")
    updates = [
        {"price_sensitivity": "medium"},   # first update
        {"price_sensitivity": "low"},      # second update wins
    ]
    result = _merge_trait_updates(existing, updates)
    assert result["price_sensitivity"] == "low"


def test_list_fields_deduplicated():
    """brand_preferences, occasions, keywords, and interests are unioned without dupes."""
    existing = _existing(
        brand_preferences=["ZARA"],
        occasions=["work"],
        keywords=["简约"],
        interests=["旅行"],
    )
    updates = [
        {
            "brand_preferences": ["ZARA", "UNIQLO"],   # ZARA already in existing
            "occasions": ["casual", "work"],            # work already in existing
            "keywords": ["舒适", "简约"],               # 简约 already in existing
            "interests": ["运动", "旅行"],              # 旅行 already in existing
        }
    ]
    result = _merge_trait_updates(existing, updates)
    assert result["brand_preferences"] == ["ZARA", "UNIQLO"]
    assert result["occasions"] == ["work", "casual"]
    assert result["keywords"] == ["简约", "舒适"]
    assert result["interests"] == ["旅行", "运动"]


def test_multiple_updates_applied_in_order():
    """When two updates touch the same key, the later one wins (dict semantics)."""
    existing = _existing()
    updates = [
        {"style_preferences": {"formal": 0.4}},
        {"style_preferences": {"formal": 0.9}},  # should win
    ]
    result = _merge_trait_updates(existing, updates)
    assert result["style_preferences"]["formal"] == pytest.approx(0.9)


def test_empty_existing_works_cleanly():
    """merge() must not crash when the user has no prior traits."""
    result = _merge_trait_updates({}, [{"style_preferences": {"casual": 0.7}, "interests": ["音乐"]}])
    assert result["style_preferences"]["casual"] == pytest.approx(0.7)
    assert "音乐" in result["interests"]
    # All other list fields should be empty lists, not raise KeyError
    assert result["brand_preferences"] == []
    assert result["occasions"] == []
