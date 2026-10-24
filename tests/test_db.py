import pytest

from backend import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


@pytest.mark.parametrize("text,expected", [
    ("18 Years", 18.0), ("1 Year", 1.0), ("6 Months", 0.5), ("N/A", None),
    (None, None), ("", None), ("52 Weeks", pytest.approx(0.9971, abs=1e-3)),
    ("65 years", 65.0), ("abc", None),
])
def test_parse_age_years(text, expected):
    assert db.parse_age_years(text) == expected


def test_upsert_and_fts(conn):
    rec = {"nct_id": "NCT1", "brief_title": "Breast study", "conditions": ["Breast Cancer"],
           "keywords": ["HER2"], "phases": ["PHASE1"], "brief_summary": "tamoxifen"}
    assert db.upsert_trial(conn, rec) is True
    assert db.upsert_trial(conn, {**rec, "brief_title": "Breast study v2"}) is False
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
    hits = conn.execute("SELECT nct_id FROM fts WHERE fts MATCH 'tamoxifen'").fetchall()
    assert [h["nct_id"] for h in hits] == ["NCT1"]
    assert conn.execute("SELECT COUNT(*) FROM fts").fetchone()[0] == 1  # no duplicate FTS row


def test_tags_idempotent(conn):
    assert db.add_cancer_tags(conn, "NCT1", ["breast", "lung"]) == 2
    assert db.add_cancer_tags(conn, "NCT1", ["breast"]) == 0


def test_replace_locations(conn):
    db.replace_locations(conn, "NCT1", [{"city": "A", "lat": 1.0, "lon": 2.0}, {"city": "B"}])
    db.replace_locations(conn, "NCT1", [{"city": "C"}])
    rows = conn.execute("SELECT city FROM locations WHERE nct_id='NCT1'").fetchall()
    assert [r["city"] for r in rows] == ["C"]


def test_meta(conn):
    assert db.get_meta(conn, "last_index_run") is None
    ts = db.mark_index_run(conn)
    assert db.get_meta(conn, "last_index_run") == ts


def test_summaries_table_exists(conn):
    conn.execute("INSERT INTO summaries (nct_id, faithful) VALUES ('NCT1', 1)")
