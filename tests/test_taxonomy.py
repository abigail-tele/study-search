"""Tests for backend.taxonomy. The detect_* tests are xfail(strict) until a human
implements detect_cancer_types; remove the marker then (strict=True will remind you)."""
import pytest

from backend import taxonomy

HUMAN = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                          reason="HUMAN-WRITTEN: detect_cancer_types")

BLOOD = {"leukemia_all", "leukemia_aml", "leukemia_cll", "leukemia_cml",
         "lymphoma_hodgkin", "lymphoma_nhl", "multiple_myeloma"}


# ---- loaders (must pass now) ----
def test_load_cancer_types_shape():
    types = taxonomy.load_cancer_types()
    assert len(types) >= 25
    assert all({"id", "label", "lay_terms", "clinical_terms"} <= set(t) for t in types)


def test_umbrella_ids_exist():
    ids = {t["id"] for t in taxonomy.load_cancer_types()}
    for phrase, members in taxonomy.load_umbrella_terms().items():
        assert members and set(members) <= ids, phrase


def test_chips_match_types():
    chips = taxonomy.cancer_type_chips()
    assert len(chips) == len(taxonomy.load_cancer_types())
    assert all(set(c) == {"id", "label"} for c in chips)


def test_detect_is_stubbed_or_returns_list():
    # Passes both before and after the human implements it.
    try:
        assert isinstance(taxonomy.detect_cancer_types("breast cancer"), list)
    except NotImplementedError:
        pass


# ---- detection (human) ----
OFF_TOPIC = ["knee pain", "diabetes", "all of my family", "flu shot", "how do I bake bread",
             "high blood pressure", "best pizza near me", "my back hurts", "asthma inhaler",
             "broken arm", "", "all", "hello"]


@HUMAN
@pytest.mark.parametrize("q", OFF_TOPIC)
def test_off_topic_returns_empty(q):
    assert taxonomy.detect_cancer_types(q) == []


@HUMAN
@pytest.mark.parametrize("q,expected", [
    ("breast cancer", {"breast"}),
    ("lump in my breast", {"breast"}),
    ("Breast Cancer clinical trials", {"breast"}),
    ("BREAST CANCER", {"breast"}),
    ("brest cancer", {"breast"}),
    ("prostate cancer", {"prostate"}),
    ("prostrate cancer", {"prostate"}),
    ("pancreatic cancer", {"pancreatic"}),
    ("melanoma", {"melanoma"}),
    ("nsclc", {"lung_nsclc"}),
    ("cancer in my lungs", {"lung_nsclc"}),
    ("glioblastoma", {"brain"}),
    ("multiple myeloma", {"multiple_myeloma"}),
])
def test_single_type_queries(q, expected):
    assert set(taxonomy.detect_cancer_types(q)) == expected


@HUMAN
def test_lung_cancer_is_both_lung_types():
    assert set(taxonomy.detect_cancer_types("lung cancer")) == {"lung_nsclc", "lung_sclc"}


@HUMAN
def test_blood_cancer_is_seven_types():
    assert set(taxonomy.detect_cancer_types("blood cancer")) == BLOOD
    assert len(taxonomy.detect_cancer_types("blood cancer")) == 7


@HUMAN
def test_lymphoma_umbrella():
    assert set(taxonomy.detect_cancer_types("lymphoma")) == {"lymphoma_hodgkin", "lymphoma_nhl"}


@HUMAN
def test_no_duplicates_and_known_ids():
    ids = {t["id"] for t in taxonomy.load_cancer_types()}
    out = taxonomy.detect_cancer_types("blood cancer and leukemia")
    assert len(out) == len(set(out)) and set(out) <= ids


@HUMAN
def test_bare_all_never_matches_leukemia():
    assert "leukemia_all" not in taxonomy.detect_cancer_types("all of my family has diabetes")
    assert taxonomy.detect_cancer_types("all") == []


@HUMAN
def test_all_leukemia_phrase_matches():
    assert "leukemia_all" in taxonomy.detect_cancer_types("all leukemia")
