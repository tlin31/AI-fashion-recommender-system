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

import json
import re

import pytest

from seed_gorse import (
    COLOUR_KEYWORDS,
    STYLE_KEYWORDS,
    _colour_labels,
    _attribute_labels,
    _product_type,
    _style_labels,
    build_item_comment,
    build_item_labels,
    flatten_labels,
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


def labels_of(**overrides) -> dict:
    """The whole label map."""
    return build_item_labels(row(**overrides))[0]


def features_of(**overrides) -> list[str]:
    """Only the branch tags item-to-item can see (column = "item.Labels.f")."""
    return labels_of(**overrides)["f"]


def all_label_strings(**overrides) -> list[str]:
    """Every label as "prefix:value", features and attributes alike."""
    return [label for _, label, _ in flatten_labels(labels_of(**overrides))]


# ── The core invariant ────────────────────────────────────────────────────────

# Prefixes whose value is a category name. A number appended to any of these is
# the trait-sync defect reappearing on the item side.
CATEGORICAL_PREFIXES = ("style", "color", "occasion", "brand", "material",
                        "price_range")


def test_no_categorical_label_has_a_score_appended():
    """No category label may carry a decimal score in its string.

    This is the item-side guard against the exact defect in the user-side trait
    sync, which emitted "style:minimalist:0.8" and thereby made almost every
    trait label a singleton that Gorse drops.
    """
    got = all_label_strings(name="Minimalist Black Dress", description="simple and clean")
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
    assert labels_of(avg_rating=4.3)["avg_rating"] == "4.3"
    assert labels_of(avg_rating=4.333333)["avg_rating"] == "4.3"


def test_avg_rating_is_a_string_not_a_number():
    """A JSON number under a map key reaches neither model.

    ctr.convertLabels (model/ctr/data.go:44) switches on string / []any /
    map / json.Number — and gorm's json serializer decodes into float64, not
    json.Number, so the numeric branch never fires. logics' flatten ignores
    numeric leaves too. A float here would look like a feature and be inert.
    """
    assert isinstance(labels_of(avg_rating=4.3)["avg_rating"], str)


def test_style_and_colour_labels_carry_no_score():
    got = features_of(name="Minimalist Black Dress", description="simple and clean")
    style = [l for l in got if l.startswith("style:")]
    colour = [l for l in got if l.startswith("color:")]
    assert style and colour, "fixture should produce both kinds of label"
    for label in style + colour:
        assert label.count(":") == 1, f"{label!r} has an extra field appended"


# ── The similarity branch: what may and may not be in `f` ────────────────────

def test_similarity_branch_excludes_the_labels_that_caused_compression():
    """`f` is what column = "item.Labels.f" sees, and these four must not be in it.

    Measured on the flat schema, all 330 sampled item-to-item scores fell in
    0.500689-0.510655 with a within-list spread of 0.000339. Two arbitrary items
    shared price_range:mid plus avg_rating:4.5 and nothing else: the labels that
    discriminate (item_name, brand — near-unique) never matched, and the labels
    that matched did not discriminate.
    """
    features = features_of()
    for prefix in ("brand:", "price_range:", "avg_rating:", "item_name:", "price:"):
        assert not [f for f in features if f.startswith(prefix)], (
            f"{prefix!r} is on the similarity path again"
        )


def test_attributes_survive_outside_the_similarity_branch():
    """Off the similarity path is not the same as deleted.

    brand / price_range / avg_rating stay in the map, so ctr.ConvertLabels still
    flattens them into CTR features (brand.TestBrand, price_range.mid).
    """
    labels = labels_of()
    assert labels["brand"] == "TestBrand"
    assert labels["price_range"] == "mid"
    assert labels["avg_rating"] == "4.3"


def test_every_item_gets_a_feature_even_with_no_text_signal():
    """Zero empty feature sets is the point of adding type: and cat:.

    On the old schema style covered 48.7% and color 62.1% of the 95,335-product
    eval catalogue, leaving 21.0% of items with an empty feature set — items
    tags item-to-item cannot place at all, and pairs ILD has to skip.
    """
    bare = features_of(name="zzz", description="zzz", colour="",
                       occasion="", material="")
    assert not [f for f in bare if f.startswith(("style:", "color:"))], (
        "fixture should defeat both keyword heuristics"
    )
    assert bare == ["type:t-shirt", "cat:tops"]


def test_taxonomy_has_two_levels_so_jaccard_has_a_middle_rung():
    """Same type > same category > unrelated, which a flat vocabulary cannot say."""
    def bare(category):
        return set(features_of(category=category, name="x", description="x",
                               colour="", occasion="", material=""))

    tee, tee2 = bare("t-shirt"), bare("t-shirt")
    blouse, ring = bare("blouse"), bare("necklace")

    assert len(tee & tee2) > len(tee & blouse) > len(tee & ring)
    assert tee & ring == set()


def test_product_type_slugifies_and_never_embeds_a_space():
    assert _product_type("Crop Top") == "crop-top"
    assert _product_type("  DRESS ") == "dress"
    assert _product_type("") == ""
    assert _product_type(None) == ""


# ── price_range must not conflate 'unknown' with 'mid' ────────────────────────

def test_null_price_is_labelled_unknown_not_mid():
    """A missing price must not masquerade as a mid-range price.

    normalize.py assigns 'mid' to every price-less product, which at N=1000
    makes 672 of 803 'mid' items actually mean 'no data'.
    """
    assert labels_of(price=None, price_range="mid")["price_range"] == "unknown"


def test_real_price_keeps_its_band():
    assert labels_of(price=5.0, price_range="budget")["price_range"] == "budget"


# ── Carriers live in Comment, not Labels ─────────────────────────────────────

def test_carriers_are_in_the_comment_payload_not_the_labels():
    """name/price are what the frontend needs and no model should charge for.

    They were labels, and on the tags path they were the worst kind: both
    near-unique, so their IDF is about log(95335) = 11.5 each against about 1.5
    for a shared style tag — and the distance function divides by the sqrt of
    the sum over ALL of an item's tags.
    """
    carrier = json.loads(build_item_comment(row()))
    assert carrier["name"] == "Plain Cotton Tee"
    assert carrier["price"] == pytest.approx(29.99)
    assert carrier["desc"]

    for label in all_label_strings():
        assert not label.startswith(("item_name:", "price:"))


def test_null_price_carries_none_not_zero():
    """A price-less product must be distinguishable from a free one.

    api/server.go leaves item.Price at 0 either way, but the payload has to keep
    the distinction rather than launder it at seed time.
    """
    assert json.loads(build_item_comment(row(price=None)))["price"] is None


def test_flatten_labels_marks_the_similarity_path():
    """The audit's `sim` column decides whether a prefix reaches item-to-item."""
    on_sim = {prefix: sim for prefix, _, sim in flatten_labels(labels_of())}
    assert on_sim["type"] and on_sim["cat"] and on_sim["style"]
    assert not on_sim["brand"] and not on_sim["price_range"] and not on_sim["avg_rating"]


def test_flatten_labels_still_reads_the_old_flat_schema():
    """So the audit can be pointed at pre-restructure data for a before/after."""
    got = flatten_labels(["style:minimalist", "brand:zara"])
    assert got == [("style", "style:minimalist", True), ("brand", "brand:zara", True)]


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


# ── Garment attributes ───────────────────────────────────────────────────────

def test_men_does_not_fire_inside_women():
    """The word-boundary trap that makes these regexes and not substring checks.

    "men" is a substring of "women", so a naive `in` test labels every women's
    garment as menswear — and it would look like a coverage win, not a bug.
    """
    got = _attribute_labels("Women's Long Sleeve Blouse", None)
    assert "audience:women" in got
    assert "audience:men" not in got

    assert "audience:men" in _attribute_labels("Men's Cotton Polo", None)


def test_attributes_are_canonicalised_not_taken_verbatim():
    """Synonyms must collapse, or one attribute fragments into several labels.

    That is the same defect as the raw `colour` column: uncontrolled free text
    produces near-unique values that never match.
    """
    for title in ("Long Sleeve Tee", "Long-Sleeve Tee", "Long Sleeved Tee"):
        assert "sleeve:long" in _attribute_labels(title, None)
    for title in ("V Neck Top", "V-Neck Top", "VNeck Top"):
        assert "neck:v-neck" in _attribute_labels(title, None)


def test_attributes_reach_the_similarity_branch():
    got = features_of(name="Women's High Waisted Wide Leg Striped Palazzo Pants",
                      description="", colour="", occasion="", material="")
    assert "audience:women" in got
    assert "fit:high-waist" in got
    assert "fit:wide-leg" in got
    assert "pattern:striped" in got


def test_attributes_break_the_ties_type_and_cat_leave_behind():
    """type+cat+style+color alone put the median item in a class of 66.

    Two dresses identical on the taxonomy must still be separable when the title
    says they differ — otherwise a top-10 is one equivalence class at distance 0,
    ordered arbitrarily by the ANN index.
    """
    common = dict(category="dress", description="", colour="",
                  occasion="", material="")
    a = set(features_of(name="Women's Long Sleeve Maxi Dress", **common))
    b = set(features_of(name="Women's Sleeveless Mini Dress", **common))
    assert a != b, "titles differ on sleeve and length; the labels must too"
    assert a & b, "both are still women's dresses"


def test_absent_attributes_add_nothing_rather_than_guessing():
    """Uneven coverage is fine; inventing a default is not.

    length is present on ~10% of the catalogue. An item whose title says nothing
    about length must carry no length label, not a "length:unknown" that every
    such item would share — that is exactly the price_range:mid failure.
    """
    got = _attribute_labels("Plain Shirt", None)
    assert not [g for g in got if g.startswith("length:")]
