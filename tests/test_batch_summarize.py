"""Tests for pipeline/batch_summarize.py. No model is downloaded: generation is a fake function."""
from __future__ import annotations

import json

import pytest

from backend import config, db, summarize
from pipeline import batch_summarize as bs
from tests.test_summarize import TRIAL, _fake_encode

VERSION = "fake@0000"


def fake_gen(texts: list[str]) -> list[str]:
    return ["PLAIN: " + t for t in texts]


def _rec(nct: str, status: str = "RECRUITING", **kw) -> dict:
    base = {"nct_id": nct, "status": status, "brief_summary": TRIAL["brief_summary"],
            "eligibility_text": TRIAL["eligibility_text"], "brief_title": "t"}
    return base | kw


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    for r in (_rec("NCT00000003", "COMPLETED"), _rec("NCT00000002", "NOT_YET_RECRUITING"),
              _rec("NCT00000001", "RECRUITING"), _rec("NCT00000004", "RECRUITING")):
        db.upsert_trial(c, r)
    c.commit()
    return c


@pytest.fixture
def words(monkeypatch):
    """Word-count tokens, fake encoder and fake model version for the check stage."""
    monkeypatch.setattr(summarize, "_token_counter", lambda: summarize._word_count_tokens)
    monkeypatch.setattr(summarize, "model_version", lambda *a, **k: VERSION)
    monkeypatch.setattr(summarize, "_encode", _fake_encode)


def _unimplemented(source, output):
    raise NotImplementedError("faithfulness_check")


@pytest.fixture
def stub(monkeypatch):
    """Pretend faithfulness_check is not written yet (every model piece must then fall back)."""
    monkeypatch.setattr(summarize, "faithfulness_check", _unimplemented)


def _generate_all(conn, **kw):
    todo = bs.pending_trials(conn, VERSION, **kw)
    return bs.generate_slab(conn, todo, VERSION, fake_gen, summarize._word_count_tokens)


# --- piece split parity with the live path ------------------------------------
def test_piece_split_matches_what_summarize_trial_sends_to_the_model(words, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(summarize, "_generate_batch", lambda prompts: sent.extend(prompts) or ["" for _ in prompts])
    summarize.summarize_trial(TRIAL)
    planned = [config.TASK_PREFIX + p["source_text"] for p in summarize.plan_pieces(TRIAL)]
    assert sorted(sent) == sorted(planned) and len(planned) >= 3


def test_piece_ids_are_unique_and_kinds_are_right():
    pieces = summarize.plan_pieces(TRIAL)
    assert len({p["piece_id"] for p in pieces}) == len(pieces)
    kinds = {p["piece_id"][0]: p["kind"] for p in pieces}
    assert kinds == {"S": "summary", "I": "include", "E": "exclude"}
    assert not any(p["source_text"] == "Pregnant" for p in pieces)   # below the minimum-words floor


# --- generate stage ------------------------------------------------------------
def test_pending_order_recruiting_first_then_not_yet_then_rest(conn):
    assert [t["nct_id"] for t in bs.pending_trials(conn, VERSION)] == \
        ["NCT00000001", "NCT00000004", "NCT00000002", "NCT00000003"]


def test_pending_status_filter(conn):
    got = bs.pending_trials(conn, VERSION, ["NOT_YET_RECRUITING"])
    assert [t["nct_id"] for t in got] == ["NCT00000002"]


def test_generate_stores_raw_outputs_and_is_resumable(conn):
    pieces, trials = _generate_all(conn, statuses=["RECRUITING"])
    assert trials == 2 and pieces > 0
    rows = conn.execute("SELECT * FROM model_outputs").fetchall()
    assert len(rows) == pieces and all(r["output_text"] == "PLAIN: " + r["source_text"] for r in rows)
    assert {r["model_version"] for r in rows} == {VERSION}
    # resume: finished trials are not pending again, the others still are
    assert [t["nct_id"] for t in bs.pending_trials(conn, VERSION)] == ["NCT00000002", "NCT00000003"]
    _generate_all(conn)
    assert bs.pending_trials(conn, VERSION) == []
    n = conn.execute("SELECT COUNT(*) FROM model_outputs").fetchone()[0]
    _generate_all(conn)                                   # nothing pending: no new rows
    assert conn.execute("SELECT COUNT(*) FROM model_outputs").fetchone()[0] == n


def test_new_model_version_regenerates(conn):
    _generate_all(conn)
    assert len(bs.pending_trials(conn, "other@1")) == 4


# --- check stage -----------------------------------------------------------------
def _run_check(tmp_path, conn, *extra):
    stats_path = tmp_path / "stats.json"
    rc = bs.main(["check", "--db", str(tmp_path / "t.db"), "--model-version", VERSION,
                  "--stats", str(stats_path), *extra])
    return rc, json.loads(stats_path.read_text())


def test_check_with_stub_everything_falls_back_and_guard_only_is_measured(conn, tmp_path, words, stub):
    _generate_all(conn)
    rc, stats = _run_check(tmp_path, conn)
    assert rc == 0 and stats["faithfulness_check_implemented"] is False
    real, guard = stats["real_run"], stats["guard_only"]
    assert real["shown_as_model_text"] == 0 and real["fell_back"]["faithfulness_check"] == real["pieces_total"] > 0
    assert guard["pieces_total"] > 0 and guard["shown_as_model_text"] > 0           # what WOULD show
    assert 0 < guard["model_text_rate"] <= 1
    # summaries rows: original wording only, never model text (rule 3), PLAN section 2 shape
    c = db.connect(tmp_path / "t.db")
    assert c.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 4
    s = db.get_summary(c, "NCT00000001")
    assert set(s) == {"nct_id", "what_is_tested", "who_can_join", "original", "faithful", "fallback_used",
                      "reading_level", "model_version", "generated_at"}
    assert s["fallback_used"] is True and s["faithful"] is False and s["model_version"] == VERSION
    assert "PLAIN" not in json.dumps(s)
    assert "metastatic (has spread" in s["what_is_tested"]
    assert s["who_can_join"]["include"][0] == "Age 18 years or older"
    assert set(s["reading_level"]) == {"before", "after"}


def test_guard_stat_is_never_written_to_summaries(conn, tmp_path, words, stub):
    _generate_all(conn)
    _run_check(tmp_path, conn)
    rows = db.connect(tmp_path / "t.db").execute("SELECT what_is_tested, include_json FROM summaries").fetchall()
    assert not any("PLAIN" in r["what_is_tested"] or "PLAIN" in r["include_json"] for r in rows)


def test_check_matches_live_summarize_when_check_passes(conn, tmp_path, words, monkeypatch):
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    monkeypatch.setattr(summarize, "_generate_batch", lambda prompts: fake_gen([p[len(config.TASK_PREFIX):] for p in prompts]))
    _generate_all(conn)
    rc, stats = _run_check(tmp_path, conn, "--no-guard-stat")
    assert rc == 0 and stats["faithfulness_check_implemented"] is True
    live = summarize.summarize_trial(_rec("NCT00000001"))
    batch = db.get_summary(db.connect(tmp_path / "t.db"), "NCT00000001")
    for key in ("what_is_tested", "who_can_join", "faithful", "fallback_used", "reading_level", "model_version"):
        assert batch[key] == live[key]
    assert stats["real_run"]["shown_as_model_text"] == stats["real_run"]["pieces_total"]


def test_recheck_after_check_implemented_refreshes_summaries_without_regenerating(conn, tmp_path, words, monkeypatch, stub):
    _generate_all(conn)
    _run_check(tmp_path, conn, "--no-guard-stat")
    n_out = conn.execute("SELECT COUNT(*) FROM model_outputs").fetchone()[0]
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    _run_check(tmp_path, conn, "--no-guard-stat")
    c = db.connect(tmp_path / "t.db")
    assert "PLAIN" in db.get_summary(c, "NCT00000001")["what_is_tested"]
    assert c.execute("SELECT COUNT(*) FROM model_outputs").fetchone()[0] == n_out


def test_check_skips_incomplete_and_stale_trials(conn, tmp_path, words, stub):
    _generate_all(conn)
    conn.execute("DELETE FROM model_outputs WHERE nct_id='NCT00000001' AND piece_id LIKE 'S:%'")        # incomplete
    conn.execute("UPDATE trials SET brief_summary='Completely different text now.' WHERE nct_id='NCT00000002'")  # stale
    conn.commit()
    _, stats = _run_check(tmp_path, conn, "--no-guard-stat")
    assert stats["trials"]["skipped_incomplete_or_stale"] == 2 and stats["trials"]["checked"] == 2
    got = {r[0] for r in db.connect(tmp_path / "t.db").execute("SELECT nct_id FROM summaries")}
    assert got == {"NCT00000003", "NCT00000004"}


def test_check_without_cached_outputs_returns_error(conn, tmp_path, words):
    assert bs.main(["check", "--db", str(tmp_path / "t.db"), "--model-version", VERSION,
                    "--stats", str(tmp_path / "s.json")]) == 1


def test_delete_trial_removes_cached_outputs(conn):
    _generate_all(conn)
    db.delete_trial(conn, "NCT00000001")
    assert conn.execute("SELECT COUNT(*) FROM model_outputs WHERE nct_id='NCT00000001'").fetchone()[0] == 0
