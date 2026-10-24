from backend.embedding import trial_text
from tests.fixtures import tiny_db


def test_trial_text_combines_fields():
    row = {"brief_title": "T", "conditions": '["A","B"]', "keywords": '["K"]', "brief_summary": "S"}
    assert trial_text(row) == "T. A; B. K. S"


def test_trial_text_handles_missing():
    row = {"brief_title": "T", "conditions": None, "keywords": "not json", "brief_summary": None}
    assert trial_text(row) == "T"


def test_index_matches_trials():
    conn = tiny_db.build_conn()
    index, ids = tiny_db.build_index(conn)
    assert index.ntotal == len(ids) == len(tiny_db.TRIALS)
    assert ids == sorted(ids)


def test_index_searchable():
    conn = tiny_db.build_conn()
    index, ids = tiny_db.build_index(conn)
    scores, rows = index.search(tiny_db.fake_encode(["prostate cancer radiation"]), 1)
    assert ids[rows[0][0]] == "NCT00000006"
    assert scores[0][0] <= 1.0001


def test_get_encoder_missing_local_model_raises_not_downloads(monkeypatch, tmp_path):
    """Rule 4: a missing local model must be a clear error, never a Hub download."""
    import pytest
    from backend import config, embedding
    monkeypatch.setattr(config, "EMBED_MODEL_DIR", tmp_path / "nope")
    monkeypatch.setattr(embedding, "_encoder", None)
    with pytest.raises(RuntimeError, match="fetch_models"):
        embedding.get_encoder()
