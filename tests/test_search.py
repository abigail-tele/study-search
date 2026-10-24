"""Tests for backend.search and the db read helpers.

RRF and detect_cancer_types are HUMAN-WRITTEN stubs; integration tests swap in
simple fakes defined here (never in backend/) so the pipeline is still covered.
"""
import pytest

from backend import config, db, search, taxonomy
from tests.fixtures import tiny_db

HUMAN_RRF = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                              reason="HUMAN-WRITTEN: reciprocal_rank_fusion")


# ------------------------------------------------------------- test-only fakes
def fake_rrf(rankings, k=config.RRF_K):
    scores: dict[str, float] = {}
    for r in rankings:
        for pos, nct in enumerate(r, start=1):
            scores[nct] = scores.get(nct, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def fake_detect(q):
    q = q.lower()
    if "lung" in q:
        return ["lung_nsclc", "lung_sclc"]
    if "breast" in q:
        return ["breast"]
    if "prostate" in q:
        return ["prostate"]
    return []


@pytest.fixture
def env(monkeypatch):
    conn = tiny_db.build_conn()
    index, ids = tiny_db.build_index(conn)
    monkeypatch.setattr(search, "_get_index", lambda: (index, ids))
    monkeypatch.setattr(search, "get_encoder", lambda: tiny_db.fake_encode)
    monkeypatch.setattr(search, "reciprocal_rank_fusion", fake_rrf)
    monkeypatch.setattr(taxonomy, "detect_cancer_types", fake_detect)
    return conn


# ------------------------------------------------------------- pure functions
def test_build_citation_exact_format():
    t = {"nct_id": "NCT1", "sponsor": "Acme", "official_title": "Long Title",
         "brief_title": "Short", "last_updated": "2026-09-14"}
    assert search.build_citation(t) == (
        "Acme. Long Title. ClinicalTrials.gov ID: NCT1. Last updated 2026-09-14. "
        "Available from: https://clinicaltrials.gov/study/NCT1")


def test_build_citation_falls_back_to_brief_title():
    t = {"nct_id": "NCT1", "sponsor": "Acme", "official_title": None,
         "brief_title": "Short", "last_updated": "2026-01-02"}
    assert ". Short. ClinicalTrials.gov ID: NCT1." in search.build_citation(t)


@pytest.mark.parametrize("raw,expected", [
    ('["PHASE1","PHASE2"]', "PHASE1|PHASE2"), ('["PHASE3"]', "PHASE3"), ('["NA"]', "NA"),
    ("[]", "NA"), (None, "NA"), ("garbage", "NA")])
def test_normalize_phase(raw, expected):
    assert search.normalize_phase(raw) == expected


def test_allowed_statuses():
    assert search.allowed_statuses(None) == config.INDEX_STATUSES
    assert search.allowed_statuses("any") == config.INDEX_STATUSES
    assert search.allowed_statuses("recruiting") == ("RECRUITING",)
    with pytest.raises(ValueError):
        search.allowed_statuses("COMPLETED")


@pytest.mark.parametrize("mn,mx,age,ok", [
    (18, None, 30, True), (18, None, 17, False), (None, 65, 66, False), (None, None, 5, True),
    (18, 65, 18, True), (18, 65, 65, True), (18, 65, None, True), (0.5, 17, 10, True)])
def test_age_ok(mn, mx, age, ok):
    assert search.age_ok(mn, mx, age) is ok


def test_bounding_box_contains_center():
    lo_la, hi_la, lo_lo, hi_lo = search.bounding_box((40.0, -74.0), 50)
    assert lo_la < 40.0 < hi_la and lo_lo < -74.0 < hi_lo
    assert hi_la - lo_la == pytest.approx(2 * 50 / 69.0)


def test_fts_query_sanitizes():
    assert search.fts_query('breast "cancer" AND (near) me; DROP') == '"breast" OR "drop"'
    assert search.fts_query("the of") == ""
    assert search.fts_query("") == ""


# ------------------------------------------------------------- RRF (human)
@HUMAN_RRF
def test_rrf_basic_formula():
    out = dict(search.reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)


@HUMAN_RRF
def test_rrf_item_in_both_lists_beats_single():
    out = search.reciprocal_rank_fusion([["a", "b"], ["c", "a"]], k=60)
    assert out[0][0] == "a"


@HUMAN_RRF
def test_rrf_item_in_only_one_list():
    out = dict(search.reciprocal_rank_fusion([["a"], ["b"]], k=60))
    assert out == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 61)}


@HUMAN_RRF
def test_rrf_rank_starts_at_one():
    out = dict(search.reciprocal_rank_fusion([["a"]], k=0))
    assert out["a"] == pytest.approx(1.0)


@HUMAN_RRF
def test_rrf_k_effect():
    small = dict(search.reciprocal_rank_fusion([["a", "b"]], k=1))
    big = dict(search.reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert small["a"] / small["b"] > big["a"] / big["b"]


@HUMAN_RRF
def test_rrf_sorted_descending():
    out = search.reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]], k=60)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


@HUMAN_RRF
def test_rrf_ties_are_deterministic():
    first = search.reciprocal_rank_fusion([["b"], ["a"]], k=60)
    again = search.reciprocal_rank_fusion([["b"], ["a"]], k=60)
    assert first == again
    assert [n for n, _ in first] == ["a", "b"]  # tie broken by id ascending


@HUMAN_RRF
def test_rrf_empty():
    assert search.reciprocal_rank_fusion([]) == []
    assert search.reciprocal_rank_fusion([[], []]) == []


@HUMAN_RRF
def test_rrf_default_k_comes_from_config():
    out = dict(search.reciprocal_rank_fusion([["a"]]))
    assert out["a"] == pytest.approx(1 / (config.RRF_K + 1))


# ------------------------------------------------------------- integration
def test_search_shape_and_results(env):
    r = search.search("breast cancer", status="ANY", conn=env)
    assert set(r) == {"query", "detected_types", "off_topic", "suggested_types", "results"}
    assert r["detected_types"] == ["breast"] and r["off_topic"] is False
    assert r["suggested_types"] == []
    assert {x["nct_id"] for x in r["results"]} == {"NCT00000001", "NCT00000002", "NCT00000003"}
    item = r["results"][0]
    assert set(item) == {"nct_id", "title", "status", "phase", "cancer_tags", "sponsor",
                         "last_updated", "nearest_site", "url", "citation", "score"}
    assert item["nearest_site"] is None
    assert item["url"] == f"https://clinicaltrials.gov/study/{item['nct_id']}"
    assert item["citation"].startswith("Test Sponsor. Official: ")


def test_search_only_inside_detected_types(env):
    r = search.search("lung cancer", status="ANY", conn=env)
    assert {x["nct_id"] for x in r["results"]} == {"NCT00000004", "NCT00000005"}


def test_off_topic_returns_chips_and_never_searches(env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("index must not be searched for off-topic queries")
    monkeypatch.setattr(search, "candidate_ids", boom)
    r = search.search("knee pain", conn=env)
    assert r["off_topic"] is True and r["results"] == [] and r["detected_types"] == []
    assert r["suggested_types"] == taxonomy.cancer_type_chips()


def test_cancer_param_overrides_detection(env):
    r = search.search("knee pain", cancer="prostate", conn=env)
    assert r["off_topic"] is False and r["detected_types"] == ["prostate"]
    assert [x["nct_id"] for x in r["results"]] == ["NCT00000006"]


def test_unknown_cancer_param_falls_back_to_detection(env):
    assert search.search("knee pain", cancer="not_a_type", conn=env)["off_topic"] is True


def test_status_filter(env):
    r = search.search("breast cancer", status="NOT_YET_RECRUITING", conn=env)
    assert [x["nct_id"] for x in r["results"]] == ["NCT00000003"]
    r = search.search("breast cancer", status="RECRUITING", conn=env)
    assert {x["nct_id"] for x in r["results"]} == {"NCT00000001", "NCT00000002"}


def test_age_filter(env):
    r = search.search("breast cancer", age=10, status="ANY", conn=env)
    assert [x["nct_id"] for x in r["results"]] == ["NCT00000003"]
    r = search.search("breast cancer", age=70, status="ANY", conn=env)
    assert [x["nct_id"] for x in r["results"]] == ["NCT00000001"]


def test_zip_radius_filters_and_reports_nearest_site(env):
    r = search.search("breast cancer", zip="10001", radius_mi=50, status="ANY", conn=env)
    assert {x["nct_id"] for x in r["results"]} == {"NCT00000001", "NCT00000003"}  # LA dropped
    by_id = {x["nct_id"]: x for x in r["results"]}
    assert by_id["NCT00000001"]["nearest_site"]["city"] == "New York"
    assert by_id["NCT00000001"]["nearest_site"]["distance_mi"] < 5


def test_zip_without_radius_keeps_all_and_computes_distance(env):
    r = search.search("breast cancer", zip="10001", status="ANY", conn=env)
    assert len(r["results"]) == 3
    assert all(x["nearest_site"] for x in r["results"])


def test_zip_drops_trials_with_no_sites(env):
    r = search.search("lung cancer", zip="10001", radius_mi=100, status="ANY", conn=env)
    assert [x["nct_id"] for x in r["results"]] == ["NCT00000004"]


def test_result_limit(env, monkeypatch):
    for i in range(20):
        nct = f"NCT9{i:07d}"
        db.upsert_trial(env, {"nct_id": nct, "brief_title": "breast extra",
                              "status": "RECRUITING", "phases": ["PHASE1"], "conditions": []})
        db.add_cancer_tags(env, nct, ["breast"])
    env.commit()
    index, ids = tiny_db.build_index(env)
    monkeypatch.setattr(search, "_get_index", lambda: (index, ids))
    assert len(search.search("breast", conn=env)["results"]) == config.RESULT_LIMIT


def test_phase_normalized_in_results(env):
    r = search.search("breast cancer", status="ANY", conn=env)
    phases = {x["nct_id"]: x["phase"] for x in r["results"]}
    assert phases["NCT00000002"] == "PHASE1|PHASE2" and phases["NCT00000003"] == "NA"


def test_does_not_log_query(env, caplog):
    with caplog.at_level("DEBUG"):
        search.search("breast cancer", zip="10001", conn=env)
    assert "10001" not in caplog.text and "breast cancer" not in caplog.text


# ------------------------------------------------------------- db read helpers
def test_get_study_shape(env):
    s = db.get_study(env, "NCT00000001")
    assert set(s) == {"nct_id", "title", "official_title", "status", "phase", "conditions",
                      "cancer_tags", "min_age", "max_age", "sex", "sponsor", "start_date",
                      "last_updated", "enrollment", "sites", "url", "citation", "brief_summary",
                      "eligibility_text"}
    assert s["min_age"] == "18 Years" and s["max_age"] is None
    assert s["cancer_tags"] == ["breast"] and s["sites"][0]["city"] == "New York"
    assert set(s["sites"][0]) == {"facility", "city", "state", "zip", "lat", "lon"}
    assert s["citation"].endswith("https://clinicaltrials.gov/study/NCT00000001")


def test_get_study_missing(env):
    assert db.get_study(env, "NCT404") is None


def test_get_summary(env):
    assert db.get_summary(env, "NCT00000001") is None
    env.execute("INSERT INTO summaries VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("NCT00000001", "Plain text", '["a"]', '["b"]', 1, 0, 14.2, 8.1, "v1",
                 "2026-10-24T00:00:00Z"))
    s = db.get_summary(env, "NCT00000001")
    assert s["who_can_join"] == {"include": ["a"], "exclude": ["b"]}
    assert s["faithful"] is True and s["fallback_used"] is False
    assert s["reading_level"] == {"before": 14.2, "after": 8.1}
    assert s["original"]["summary"].startswith("A study of")


@pytest.mark.parametrize("years,text", [(18.0, "18 Years"), (1.0, "1 Year"), (0.5, "6 Months"),
                                        (None, None), (0.0192, "1 Week")])
def test_format_age(years, text):
    assert db.format_age(years) == text
