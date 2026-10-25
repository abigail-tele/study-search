"""Interventions: parse, backfill (mocked network), read helper, overview wording."""
import httpx
import pytest

from backend import db, overview
from pipeline import indexer
from tests.fixtures import tiny_db


def _study(nct, items):
    return {"protocolSection": {"identificationModule": {"nctId": nct},
                                "armsInterventionsModule": {"interventions": items}}}


def test_parse_interventions():
    s = _study("NCT1", [{"type": "DRUG", "name": "Pembrolizumab"}, {"type": "OTHER"}])
    assert indexer.parse_interventions(s) == [{"type": "DRUG", "name": "Pembrolizumab"}]
    assert indexer.parse_interventions({"protocolSection": {}}) == []


def test_db_helpers_roundtrip_and_missing_table():
    conn = tiny_db.build_conn()
    assert db.get_interventions(conn, "NCT00000001") == []          # table not created yet
    db.ensure_interventions_table(conn)
    db.replace_interventions(conn, "NCT00000001", [{"type": "DRUG", "name": "A"}, {"type": "RADIATION", "name": "B"}])
    db.replace_interventions(conn, "NCT00000001", [{"type": "DRUG", "name": "C"}])
    assert db.get_interventions(conn, "NCT00000001") == [{"type": "DRUG", "name": "C"}]


def test_backfill_uses_filter_ids_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer.time, "sleep", lambda s: None)
    path = tmp_path / "t.db"
    conn = db.connect(path)
    db.init_schema(conn)
    for n in ("NCT00000001", "NCT00000002", "NCT00000003"):
        db.upsert_trial(conn, {"nct_id": n, "brief_title": n})
    conn.commit()
    conn.close()
    reqs = []

    def handler(req):
        reqs.append(req)
        assert "InterventionName" in req.url.params["fields"] and "DetailedDescription" not in req.url.params["fields"]
        return httpx.Response(200, json={"studies": [_study("NCT00000001", [{"type": "DRUG", "name": "X"}]),
                                                     _study("NCT00000003", [])]})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = indexer.backfill_interventions(path, client, batch=10)
    assert out["with_interventions"] == 1 and out["rows"] == 1 and len(reqs) == 1
    assert reqs[0].url.params["filter.ids"] == "NCT00000001,NCT00000002,NCT00000003"
    assert indexer.backfill_interventions(path, client, batch=10)["requested"] == 0   # nothing left to ask
    c = db.connect(path)
    assert db.get_interventions(c, "NCT00000001") == [{"type": "DRUG", "name": "X"}]


@pytest.mark.parametrize("items,expected", [
    ([], None),
    ([{"type": "DRUG", "name": "Pembrolizumab"}], "Pembrolizumab (drug)"),
    ([{"type": "RADIATION", "name": "radiation"}, {"type": "DRUG", "name": "pembrolizumab"}],
     "pembrolizumab (drug) and radiation (radiation)"),
    ([{"type": "OTHER", "name": "Survey"}, {"type": "DRUG", "name": "A"}, {"type": "DRUG", "name": "B"},
      {"type": "DRUG", "name": "C"}, {"type": "DRUG", "name": "D"}], "A (drug), B (drug), C (drug) and 2 more"),
])
def test_intervention_phrase(items, expected):
    assert overview.intervention_phrase(items) == expected


def test_overview_mentions_interventions():
    study = {"status": "RECRUITING", "phase": "PHASE2", "sex": "ALL", "min_age": "18 Years", "max_age": None,
             "conditions": ["Breast Cancer"], "sites": [], "interventions": [{"type": "DRUG", "name": "pembrolizumab"}]}
    o = overview.study_overview(study)
    assert "It is testing pembrolizumab (drug)." in o["sentence"]
    assert {"label": "Being tested", "value": "pembrolizumab (drug)"} in o["facts"]
    study["interventions"] = []
    assert "testing" not in overview.study_overview(study)["sentence"]
