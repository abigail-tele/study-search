"""API tests. search/db/taxonomy are monkeypatched, so these pass without a built database (rule 4: no network)."""
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
    monkeypatch.setattr(db, "get_summary", lambda c, n: {"nct_id": n, "faithful": True} if n == "NCT05000001" else None)
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
    assert client.get("/api/study/NCT05000001").json() == {"nct_id": "NCT05000001"}
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
