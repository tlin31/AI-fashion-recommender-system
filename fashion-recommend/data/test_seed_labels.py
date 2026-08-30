# Regression tests for the Gorse item-label schema built by seed_gorse.py.
#
# Pure unit tests — no Postgres, no Gorse, no network.
#
#   pytest fashion-recommend/data/test_seed_labels.py -v
#
# The invariant these exist to protect: a label must be a LOW-CARDINALITY,
# CANONICAL string. Gorse indexes a label only on its second occurrence
# (master/tasks.go:307-330), so any label that is near-unique per item is
# silently dropped and never reaches the CTR model. Embedding a float — the
# way traits/gorse_sync.go emits "style:minimalist:0.8" — is the fastest way
# to turn one working feature into hundreds of dropped singletons.

import re

import pytest

from seed_gorse import (
    COLOUR_KEYWORDS,
    STYLE_KEYWORDS,
    _colour_labels,
    _style_labels,
    build_item_labels,
)

# A canonical row with everything populated. Individual tests override keys.
BASE_ROW = {
    "product_id":   "B000TEST01",
    "name":         "Plain Cotton Tee",
    "brand":        "TestBrand",
    "category":     "t-shirt",
    "colour":       "",
    "occasion":     "casual",
    "material":     "cotton",
    "price":        29.99,
    "price_range":  "mid",
    "avg_rating":   4.3,
    "rating_count": 120,
    "description":  "A simple everyday shirt.",
    "image_url":    "",
}


def row(**overrides) -> dict:
    return {**BASE_ROW, **overrides}


def labels_of(**overrides) -> list[str]:
    return build_item_labels(row(**overrides))[0]


# ── The core invariant ────────────────────────────────────────────────────────

# Prefixes whose value is a category name. A number appended to any of these is
# the trait-sync defect reappearing on the item side.
CATEGORICAL_PREFIXES = ("style", "color", "occasion", "brand", "material",
                        "price_range")


def test_no_categorical_label_has_a_score_appended():
    """No category label may carry a decimal score in its string.

    This is the item-side guard against the exact defect in the user-side trait
    sync, which emits "style:minimalist:0.8" and thereby makes almost every
    trait label a singleton that Gorse drops.
    """
    got = labels_of(name="Minimalist Black Dress", description="simple and clean")
    for label in got:
        prefix = label.split(":", 1)[0]
        if prefix not in CATEGORICAL_PREFIXES:
            continue
        assert not re.search(r":\d+(\.\d+)?$", label), (
            f"{label!r} has a score appended — this fragments one feature into "
            f"many near-unique ones, which Gorse drops entirely"
        )


def test_avg_rating_is_bounded_to_one_decimal():
    """avg_rating is the one label that legitimately holds a number.

    It stays a usable feature only while its cardinality is small, so the
    precision is pinned here rather than inherited from the source column.
    """
    assert "avg_rating:4.3" in labels_of(avg_rating=4.3)
    assert "avg_rating:4.3" in labels_of(avg_rating=4.333333)
    assert not [l for l in labels_of(avg_rating=4.333333)
                if re.search(r":\d+\.\d{2,}$", l)]


def test_price_label_is_integer_truncated():
    assert "price:29" in labels_of(price=29.99)
    assert "price:29.99" not in labels_of(price=29.99)


def test_style_and_colour_labels_carry_no_score():
    got = labels_of(name="Minimalist Black Dress", description="simple and clean")
    style = [l for l in got if l.startswith("style:")]
    colour = [l for l in got if l.startswith("color:")]
    assert style and colour, "fixture should produce both kinds of label"
    for label in style + colour:
        assert label.count(":") == 1, f"{label!r} has an extra field appended"


# ── price_range must not conflate 'unknown' with 'mid' ────────────────────────

def test_null_price_is_labelled_unknown_not_mid():
    """A missing price must not masquerade as a mid-range price.

    normalize.py assigns 'mid' to every price-less product, which at N=1000
    makes 672 of 803 'mid' items actually mean 'no data'.
    """
    got = labels_of(price=None, price_range="mid")
    assert "price_range:unknown" in got
    assert "price_range:mid" not in got


def test_null_price_emits_no_price_carrier_label():
    got = labels_of(price=None)
    assert not [l for l in got if l.startswith("price:")]


def test_real_price_keeps_its_band():
    assert "price_range:budget" in labels_of(price=5.0, price_range="budget")


def test_price_carrier_label_survives_for_api_enrichment():
    """api/server.go:246 parses `price:` to populate the API response.

    Removing it would silently zero out item.Price in the frontend, so it is
    kept deliberately even though the model drops it.
    """
    assert "price:29" in labels_of(price=29.99)
    assert "item_name:Plain Cotton Tee" in labels_of()


# ── Colour normalisation ──────────────────────────────────────────────────────

def test_colour_labels_are_always_from_the_canonical_palette():
    """Free-text colour must never reach Gorse verbatim."""
    got = _colour_labels("Smokey Quartz / Black+Orange", "Beaded Necklace", None)
    assert all(c in COLOUR_KEYWORDS for c in got)
    assert "black" in got


def test_colour_synonyms_collapse_onto_one_label():
    assert _colour_labels("navy", "Shirt", None) == ["blue"]
    assert _colour_labels("burgundy", "Shirt", None) == ["red"]
    assert _colour_labels(None, "Charcoal Hoodie", None) == ["gray"]


def test_uncontrolled_colour_text_yields_no_label_rather_than_junk():
    """'3x' and 'no' are sizes/nulls in the source data, not colours."""
    assert _colour_labels("3x", "Basic Legging", None) == []
    assert _colour_labels("no", "Basic Legging", None) == []


def test_colour_read_from_product_text_when_column_is_empty():
    """941 of 1000 items have an empty colour column; text is the real source."""
    assert _colour_labels("", "Black Midi Dress", None) == ["black"]


def test_colour_matching_respects_word_boundaries():
    # 'coral' contains no standalone colour; 'incorporated' must not match 'coral'
    assert _colour_labels(None, "Incorporated Design", None) == []
    # 'goldfish' must not match 'gold'
    assert _colour_labels(None, "Goldfish Print Tee", None) == []


# ── Style tagging ─────────────────────────────────────────────────────────────

def test_style_labels_are_always_from_the_declared_taxonomy():
    got = _style_labels("Vintage Retro Floral Blazer", "elegant and romantic")
    assert all(s in STYLE_KEYWORDS for s in got)
    assert {"vintage", "formal", "romantic"} <= set(got)


def test_style_matching_respects_word_boundaries():
    # 'transportation' must not match 'sport'; 'elbow' must not match 'bow'
    assert _style_labels("Transportation Themed Elbow Patch", None) == []


def test_untagged_item_yields_no_style_label():
    """~31% of the catalogue gets no style label — that is expected, not a bug."""
    assert _style_labels("Zx9 Replacement Part", None) == []


def test_style_labels_are_deduplicated_and_sorted():
    got = _style_labels("Casual casual relaxed everyday comfy", None)
    assert got == ["casual"]


# ── Missing data must not crash label building ────────────────────────────────

@pytest.mark.parametrize("field", ["brand", "occasion", "material", "name",
                                   "description", "colour"])
def test_missing_optional_field_is_skipped_not_stringified(field):
    got = labels_of(**{field: None})
    assert not any("None" in l for l in got), (
        f"a null {field} leaked the string 'None' into a label"
    )
