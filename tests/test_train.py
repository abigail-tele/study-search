"""Tests for the pure parts of pipeline/train.py (no model download)."""
import json

from backend import config
from pipeline import train


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _row(i, split, sim):
    return {"id": f"x-{i}", "source": "s", "target": "t", "split": split, "similarity": sim}


def test_similarity_filter_and_splits(tmp_path):
    p = _write(tmp_path, [_row(1, "train", 0.9), _row(2, "train", 0.65), _row(3, "val", 0.8),
                          _row(4, "val", 0.5), _row(5, "test", 0.99)])
    tr, va = train.load_pairs(p, 0.70)
    assert [r["id"] for r in tr] == ["x-1"]
    assert [r["id"] for r in va] == ["x-3"]


def test_threshold_is_inclusive(tmp_path):
    p = _write(tmp_path, [_row(1, "train", 0.60)])
    assert len(train.load_pairs(p, 0.60)[0]) == 1


def test_test_split_never_loaded(tmp_path):
    p = _write(tmp_path, [_row(1, "test", 1.0)])
    assert train.load_pairs(p, 0.0) == ([], [])


def test_prefix_added():
    assert train.with_prefix("Hello.") == "simplify: Hello."


def test_real_file_only_train_and_val():
    if not config.PDQ_PAIRS_PATH.exists():
        return
    tr, va = train.load_pairs(config.PDQ_PAIRS_PATH, config.TRAIN_MIN_SIMILARITY)
    assert tr and va and all(r["split"] in ("train", "val") for r in tr + va)
