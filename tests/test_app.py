"""API tests. search/db/taxonomy are monkeypatched, so these pass without a built database (rule 4: no network)."""
import os
os.environ.setdefault("STUDY_SEARCH_SKIP_WARMUP", "1")  # tests never load real models
import logging

import pytest
from fastapi.testclient import TestClient

from backend import app as app_mod
from backend import db, geo, search as search_mod, taxonomy as tax


class FakeConn:
    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    calls = {}

    def fake_search(q, cancer=None, zip=None, radius_mi=None, status=None, age=None, conn=None):
        calls.update(q=q, cancer=cancer, zip=zip, radius_mi=radius_mi, status=status, age=age)
        if "pizza" in q:
            return {"query": q, "detected_types": [], "off_topic": True,
                    "suggested_types": [{"id": "breast", "label": "Breast cancer"}], "results": []}
        return {"query": q, "detected_types": ["breast"], "off_topic": False,
                "suggested_types": [], "results": [{"nct_id": "NCT05000001"}]}

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(tax, "load_cancer_types", lambda: [
        {"id": "breast", "label": "Breast cancer", "lay_terms": ["breast"], "organ_words": ["breast"]}])
    monkeypatch.setattr(db, "connect_ro", lambda path=None: FakeConn())
    monkeypatch.setattr(geo, "lookup", lambda z: None if z == "00000" else (40.0, -74.0))
    monkeypatch.setattr(db, "get_study", lambda c, n: {"nct_id": n} if n == "NCT05000001" else None)
    monkeypatch.setattr(db, "get_interventions", lambda c, n: [{"type": "DRUG", "name": "pembrolizumab"}])
    monkeypatch.setattr(db, "get_summary", lambda c, n: {"nct_id": n, "faithful": True} if n == "NCT05000001" else None)
    from backend import match
    monkeypatch.setattr(match, "match_for_nct", lambda *a, **k: {"summary_label": "Possible match"})
    c = TestClient(app_mod.app)
    c.calls = calls
    return c


def test_cancer_types(client):
    r = client.get("/api/cancer-types")
    assert r.status_code == 200
    assert r.json() == [{"id": "breast", "label": "Breast cancer", "lay_terms": ["breast"]}]


def test_search_happy_passes_parsed_params(client):
    r = client.get("/api/search", params={"q": "breast", "zip": "77030", "radius_mi": "50",
                                          "status": "RECRUITING", "age": "40", "cancer": "breast"})
    assert r.status_code == 200 and r.json()["off_topic"] is False
    assert client.calls == dict(q="breast", cancer="breast", zip="77030", radius_mi=50, status="RECRUITING", age=40)


def test_search_any_status_and_cancer_only(client):
    assert client.get("/api/search", params={"cancer": "breast", "status": "ANY"}).status_code == 200
    assert client.calls["status"] is None


def test_search_off_topic_is_200(client):
    r = client.get("/api/search", params={"q": "pizza near me"})
    assert r.status_code == 200
    assert r.json()["off_topic"] is True and r.json()["suggested_types"]


@pytest.mark.parametrize("params", [
    {}, {"q": "  "}, {"q": "x" * 201}, {"q": "a", "zip": "1234"}, {"q": "a", "zip": "abcde"},
    {"q": "a", "zip": "123456"}, {"q": "a", "radius_mi": "30"}, {"q": "a", "radius_mi": "x"},
    {"q": "a", "status": "COMPLETED"}, {"q": "a", "age": "121"}, {"q": "a", "age": "-1"}, {"q": "a", "age": "ab"},
])
def test_search_validation_errors(client, params):
    r = client.get("/api/search", params=params)
    assert r.status_code == 400
    assert set(r.json()) == {"error"}


def test_study(client):
    body = client.get("/api/study/NCT05000001").json()
    assert body["nct_id"] == "NCT05000001" and body["match"] is None
    assert set(body["overview"]) == {"sentence", "facts"} and body["criteria"] == {"include": [], "exclude": []}
    r = client.get("/api/study/NCT09999999")
    assert r.status_code == 404 and r.json() == {"error": "Study not found"}
    assert client.get("/api/study/NCT123").status_code == 400
    assert client.get("/api/study/nct05000001").status_code == 400


def test_summary(client):
    assert client.get("/api/summary/NCT05000001").json()["faithful"] is True
    r = client.get("/api/summary/NCT09999999")
    assert r.status_code == 404 and r.json() == {"error": "Summary not available yet"}
    assert client.get("/api/summary/bad").status_code == 400


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.json()


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/api.js").status_code == 200


def test_no_query_or_zip_in_logs(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.get("/api/search?q=secretquery&zip=12345")
        # Simulate a server access-log line too: the redact filter must scrub it.
        logging.getLogger("uvicorn.access").info('GET /api/search?q=secretquery&zip=12345 HTTP/1.1 200')
    text = caplog.text
    assert "secretquery" not in text and "12345" not in text


def test_unknown_zip_is_400(client):
    r = client.get("/api/search", params={"q": "a", "zip": "00000"})
    assert r.status_code == 400 and "ZIP" in r.json()["error"]


def test_search_value_error_is_400(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("unknown status: X")
    monkeypatch.setattr(search_mod, "search", boom)
    assert client.get("/api/search?q=a").status_code == 400


def test_not_implemented_is_503(client, monkeypatch):
    def boom(*a, **k):
        raise NotImplementedError("HUMAN-WRITTEN")
    monkeypatch.setattr(search_mod, "search", boom)
    r = client.get("/api/search?q=a")
    assert r.status_code == 503 and r.json() == {"error": "Search ranking is not implemented yet"}


# --- /api/summary live-generation switch (STUDY_SEARCH_LIVE_SUMMARY); no model is ever loaded ---------
class RowConn(FakeConn):
    """Fake connection whose execute() returns one trial row for NCT05000003 and nothing otherwise."""
    def execute(self, sql, params=()):
        row = {"nct_id": "NCT05000003", "brief_summary": "A study.", "eligibility_text": "Age 18+"}
        class Cur:
            def fetchone(_):
                return row if params and params[0] == "NCT05000003" else None
        return Cur()


@pytest.fixture
def live_client(client, monkeypatch, tmp_path):
    from backend import config, summarize
    saved = []
    monkeypatch.setattr(db, "connect_ro", lambda path=None: RowConn())
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(summarize, "summarize_trial", lambda trial: {"nct_id": trial["nct_id"], "faithful": True})
    monkeypatch.setattr(summarize, "save_summary", lambda conn, s: saved.append(s["nct_id"]))
    client.saved = saved
    return client


def test_summary_miss_default_is_404_and_never_generates(live_client, monkeypatch):
    from backend import summarize
    monkeypatch.delenv("STUDY_SEARCH_LIVE_SUMMARY", raising=False)
    monkeypatch.setattr(summarize, "summarize_trial", lambda t: pytest.fail("model must not run when switch is 0"))
    assert live_client.get("/api/summary/NCT05000003").status_code == 404


def test_summary_miss_live_generates_and_stores(live_client, monkeypatch):
    monkeypatch.setenv("STUDY_SEARCH_LIVE_SUMMARY", "1")
    r = live_client.get("/api/summary/NCT05000003")
    assert r.status_code == 200 and r.json()["nct_id"] == "NCT05000003"
    assert live_client.saved == ["NCT05000003"]


def test_summary_live_unknown_trial_404(live_client, monkeypatch):
    monkeypatch.setenv("STUDY_SEARCH_LIVE_SUMMARY", "1")
    assert live_client.get("/api/summary/NCT09999999").status_code == 404


def test_summary_live_model_failure_is_503(live_client, monkeypatch):
    from backend import summarize
    monkeypatch.setenv("STUDY_SEARCH_LIVE_SUMMARY", "1")
    def boom(t): raise RuntimeError("no checkpoint")
    monkeypatch.setattr(summarize, "summarize_trial", boom)
    assert live_client.get("/api/summary/NCT05000003").status_code == 503


# --- "About you" profile: validation, match objects, sort, log scrubbing (rule 8) -------------------
@pytest.fixture
def mclient(client, monkeypatch):
    """client + a fake matcher: trial ...01 is a 'Possible match', ...02 'Likely not a match'; records what it was given."""
    from backend import match
    seen = []

    def fake_match(conn, nct, age, sex, terms, encoder=None):
        seen.append((nct, age, sex, tuple(terms)))
        label = match.LABEL_NOT if nct.endswith("2") else match.LABEL_POSSIBLE
        return {"age": "unknown", "sex": "unknown", "mentioned_in_requirements": [], "mentioned_in_exclusions": [],
                "possible_conflicts": [], "summary_label": label, "label_text": label}

    def fake_search(q, **kw):
        return {"query": q, "detected_types": ["breast"], "off_topic": False, "suggested_types": [],
                "results": [{"nct_id": "NCT05000002"}, {"nct_id": "NCT05000001"}]}

    monkeypatch.setattr(match, "match_for_nct", fake_match)
    monkeypatch.setattr(search_mod, "search", fake_search)
    client.seen = seen
    return client


def test_no_profile_means_no_match_objects(mclient):
    r = mclient.get("/api/search", params={"q": "breast"}).json()
    assert all("match" not in x for x in r["results"]) and mclient.seen == []


def test_profile_adds_match_and_parses_terms(mclient):
    r = mclient.get("/api/search", params={"q": "breast", "sex": "female", "age": "40",
                                           "conditions": "HER2 positive, stage 3"}).json()
    assert [x["match"]["summary_label"] for x in r["results"]] == ["Likely not a match", "Possible match"]
    assert mclient.seen[0] == ("NCT05000002", 40, "female", ("her2 positive", "stage 3"))


def test_sort_match_puts_possible_first_default_unchanged(mclient):
    base = {"q": "breast", "conditions": "x"}
    assert [x["nct_id"] for x in mclient.get("/api/search", params=base).json()["results"]] == ["NCT05000002", "NCT05000001"]
    ordered = mclient.get("/api/search", params={**base, "sort": "match"}).json()["results"]
    assert [x["nct_id"] for x in ordered] == ["NCT05000001", "NCT05000002"]


@pytest.mark.parametrize("params", [
    {"sex": "other"}, {"conditions": "x" * 601}, {"sort": "random"}])
def test_profile_validation_400_and_no_echo(client, params):
    r = client.get("/api/search", params={"q": "a", **params})
    assert r.status_code == 400 and set(r.json()) == {"error"}
    for v in params.values():
        assert v not in r.json()["error"] or len(v) < 3


def test_conditions_at_limit_ok(mclient):
    assert mclient.get("/api/search", params={"q": "a", "conditions": "x" * 600}).status_code == 200


def test_study_profile_params(mclient):
    assert mclient.get("/api/study/NCT05000001", params={"sex": "x"}).status_code == 400
    assert mclient.get("/api/study/NCT05000001", params={"conditions": "y" * 601}).status_code == 400
    assert mclient.get("/api/study/NCT05000001", params={"age": "200"}).status_code == 400
    r = mclient.get("/api/study/NCT05000001", params={"age": "30", "conditions": "diabetes, hiv"})
    assert r.status_code == 200 and r.json()["match"]["summary_label"] == "Possible match"
    assert mclient.seen[-1] == ("NCT05000001", 30, None, ("diabetes", "hiv"))


def test_no_profile_values_in_logs(mclient, caplog):
    with caplog.at_level(logging.DEBUG):
        mclient.get("/api/search?q=a&age=47&sex=female&conditions=secretdiagnosis")
        mclient.get("/api/study/NCT05000001?age=47&sex=female&conditions=secretdiagnosis")
        logging.getLogger("uvicorn.access").info(
            'GET /api/search?q=a&age=47&sex=female&conditions=secretdiagnosis HTTP/1.1 200')
    assert "secretdiagnosis" not in caplog.text and "age=47" not in caplog.text and "sex=female" not in caplog.text
