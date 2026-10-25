"""Tests for backend.taxonomy. The detect_* tests are xfail(strict) until a human
implements detect_cancer_types; remove the marker then (strict=True will remind you)."""
import pytest

from backend import taxonomy
from tests.fixtures import tiny_taxonomy

HUMAN = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                          reason="HUMAN-WRITTEN: detect_cancer_types")


@pytest.fixture
def semantic_taxonomy(monkeypatch):
    """Tiny taxonomy + fake encoder so semantic tests never load MiniLM."""
    tiny_taxonomy.install(monkeypatch)
    yield
    taxonomy.cancer_type_term_matrix.cache_clear()


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


# ---- semantic term matrix (scaffolding, must pass now) ----
def test_term_matrix_shape_and_ids(semantic_taxonomy):
    terms, ids, matrix = taxonomy.cancer_type_term_matrix()
    assert len(terms) == len(ids) == matrix.shape[0]
    assert set(ids) == {t["id"] for t in tiny_taxonomy.TYPES}
    assert matrix.shape[1] == 5


def test_term_matrix_is_cached(semantic_taxonomy):
    assert taxonomy.cancer_type_term_matrix() is taxonomy.cancer_type_term_matrix()


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


@HUMAN
def test_mixed_query_unions_types():
    assert set(taxonomy.detect_cancer_types("breast and lung cancer")) == {
        "breast", "lung_nsclc", "lung_sclc"}


@HUMAN
def test_small_cell_lung_cancer_includes_sclc():
    out = taxonomy.detect_cancer_types("small cell lung cancer")
    assert "lung_sclc" in out


@HUMAN
@pytest.mark.parametrize("q", ["small", "camel", "ball", "a small camel and a ball"])
def test_short_acronyms_not_matched_inside_words(q):
    out = set(taxonomy.detect_cancer_types(q))
    assert not out & {"leukemia_aml", "leukemia_cml", "leukemia_all"}


@HUMAN
def test_aml_whole_word_matches():
    assert "leukemia_aml" in taxonomy.detect_cancer_types("AML")


# ---- semantic fallback (human, after exact/umbrella/fuzzy) ----
@HUMAN
def test_semantic_fallback_matches_paraphrase(semantic_taxonomy):
    assert taxonomy.detect_cancer_types("a growth in the milk ducts") == ["breast"]


@HUMAN
def test_semantic_fallback_below_threshold_is_off_topic(semantic_taxonomy):
    assert taxonomy.detect_cancer_types("weak match") == []


@HUMAN
def test_semantic_fallback_off_topic_is_empty(semantic_taxonomy):
    assert taxonomy.detect_cancer_types("knee pain") == []


@HUMAN
def test_exact_match_wins_before_semantic(semantic_taxonomy):
    assert taxonomy.detect_cancer_types("breast cancer") == ["breast"]


@HUMAN
def test_semantic_fallback_keeps_close_runners_up(semantic_taxonomy):
    assert set(taxonomy.detect_cancer_types("close pair")) == {"breast", "lung"}


@HUMAN
def test_semantic_fallback_drops_distant_runner_up(semantic_taxonomy):
    assert taxonomy.detect_cancer_types("clear winner") == ["breast"]


@HUMAN
def test_semantic_fallback_caps_types(semantic_taxonomy, monkeypatch):
    monkeypatch.setattr(taxonomy.config, "TYPE_DETECT_EMBED_THRESHOLD", 0.45)
    assert taxonomy.detect_cancer_types("four types") == ["breast", "lung", "prostate"]


@HUMAN
def test_semantic_fallback_is_deterministic(semantic_taxonomy):
    first = taxonomy.detect_cancer_types("close pair")
    second = taxonomy.detect_cancer_types("close pair")
    assert first == second and len(first) == len(set(first))
