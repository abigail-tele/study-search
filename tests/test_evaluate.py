"""Tests for pipeline.evaluate metric functions and pipeline.make_ship_db."""
import os
os.environ.setdefault("STUDY_SEARCH_SKIP_WARMUP", "1")
import sqlite3

import pytest

from backend import config, db
from pipeline import evaluate as ev
from pipeline import make_ship_db


TAGS = {"A": {"breast"}, "B": {"lung"}, "C": {"breast", "lung"}, "D": set()}


def test_type_precision():
    assert ev.type_precision_at_k(["A", "B", "C", "D"], TAGS, {"breast"}, 4) == 0.5
    assert ev.type_precision_at_k(["A", "B", "C", "D"], TAGS, {"breast"}, 2) == 0.5
    assert ev.type_precision_at_k([], TAGS, {"breast"}, 10) == 0.0


def test_type_hit():
    assert ev.type_hit_at_k(["B", "D", "A"], TAGS, {"breast"}, 3) == 1.0
    assert ev.type_hit_at_k(["B", "D", "A"], TAGS, {"breast"}, 2) == 0.0


def test_true_metrics():
    ranked = ["x", "y", "z", "w"]
    assert ev.hit_at_k(ranked, {"z"}, 3) == 1.0
    assert ev.hit_at_k(ranked, {"w"}, 3) == 0.0
    assert ev.reciprocal_rank(ranked, {"y", "w"}) == 0.5
    assert ev.reciprocal_rank(ranked, {"q"}) == 0.0
    assert ev.recall_at_k(ranked, {"x", "w", "q", "r"}, 2) == 0.25
    assert ev.recall_at_k(ranked, set(), 2) == 0.0


def test_percentile():
    assert ev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) in (5, 6)
    assert ev.percentile([7], 95) == 7


def test_replace_section_keeps_neighbours():
    doc = "# E\n\n## Retrieval (old)\n\nTODO\n\n## Other\n\nkeep me\n"
    out = ev.replace_section(doc, "## Retrieval", "## Retrieval (new)", "body")
    assert "body" in out and "TODO" not in out and "keep me" in out
    again = ev.replace_section(out, "## Retrieval", "## Retrieval (new)", "body2")
    assert again.count("## Retrieval") == 1 and "body2" in again


def test_pending_when_stubs(monkeypatch):
    def boom(_q):
        raise NotImplementedError
    monkeypatch.setattr(ev.taxonomy, "detect_cancer_types", boom)
    rows = ev.eval_off_topic([{"query": "a", "expected_types": []}], [{"query": "b"}])
    assert all(r["value"] == "" and r["note"] == ev.PENDING_HUMAN for r in rows)


@pytest.fixture
def full_db(tmp_path):
    p = tmp_path / "full.db"
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    db.upsert_trial(conn, {"nct_id": "NCT00000001", "brief_title": "T", "status": "RECRUITING",
                           "detailed_description": "long text", "brief_summary": "s", "eligibility_text": "e"})
    db.add_cancer_tags(conn, "NCT00000001", ["breast"])
    conn.execute("CREATE TABLE IF NOT EXISTS model_outputs (k TEXT, v TEXT)")
    conn.commit()
    conn.close()
    return p


def test_ship_db_schema(full_db, tmp_path):
    dst = tmp_path / "ship.db"
    make_ship_db.make_ship_db(full_db, dst)
    c = sqlite3.connect(dst)
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"trials", "trial_cancer_types", "locations", "fts", "summaries", "meta"} <= tables
    assert "model_outputs" not in tables
    cols = {r[1] for r in c.execute("PRAGMA table_info(trials)")}
    assert "detailed_description" not in cols and "eligibility_text" in cols
    c.close()
    ro = db.connect_ro(dst)
    study = db.get_study(ro, "NCT00000001")
    assert study["cancer_tags"] == ["breast"] and study["brief_summary"] == "s"
    ro.close()


def test_db_path_env(monkeypatch):
    import importlib
    monkeypatch.setenv("STUDY_SEARCH_DB", "/tmp/x.db")
    importlib.reload(config)
    try:
        assert str(config.DB_PATH).replace("\\", "/").endswith("/tmp/x.db")
    finally:
        monkeypatch.delenv("STUDY_SEARCH_DB")
        importlib.reload(config)
