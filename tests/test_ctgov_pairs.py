"""pipeline/ctgov_pairs.py with a fake bag-of-words encoder (no model download)."""
import numpy as np

from backend import config, support
from pipeline import ctgov_pairs as cp


def fake_encode(texts):
    out = np.zeros((len(texts), 512))
    for i, t in enumerate(texts):
        for w in support.content_words(t):
            out[i, sum(map(ord, w)) % 512] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


DETAIL = ("Patients with confirmed metastatic adenocarcinoma of the colon undergoing randomization will receive "
          "oxaliplatin-based chemotherapeutic intervention concomitantly with radiographic surveillance. "
          "Secondary endpoints encompass progression-free survival and pharmacokinetic characterization. "
          "Unrelated administrative sentence about sponsorship and monitoring procedures here.")
BRIEF = "Patients with colon cancer that has spread get chemotherapy and scans. Doctors will also check how long people live."


def test_split_is_deterministic_and_by_nct():
    assert cp.split_of("NCT00000001") == cp.split_of("NCT00000001")
    assert {cp.split_of(f"NCT{i:08d}") for i in range(200)} == {"train", "val"}


def test_windows():
    assert cp.windows(["a", "b", "c"], 2) == [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]


def test_low_similarity_dropped():
    assert cp.drop_reason("a b c d e f g h i j", "k l m n o p q r s t", 0.1) == "low_similarity"


def test_new_number_rejected(monkeypatch):
    monkeypatch.setattr(cp, "fk_grade", lambda t: 20.0 if len(t) > 90 else 5.0)
    src = "Participants receive pembrolizumab 5 mg intravenously every three weeks during randomized evaluation."
    assert cp.drop_reason(src, "People get pembrolizumab 9 mg every three weeks in the study.", 0.8) == "new_key_term"
    assert cp.drop_reason(src, "People get pembrolizumab 5 mg every three weeks in the study.", 0.8) is None


def test_build_excludes_eval_and_counts(monkeypatch):
    monkeypatch.setattr(cp, "fk_grade", lambda t: 20.0 if len(t) > 100 else 5.0)
    monkeypatch.setattr(config, "CTGOV_PAIR_SIM", 0.2)
    trials = [("NCT00000001", BRIEF, DETAIL), ("NCT00000002", BRIEF, DETAIL)]
    recs, stats = cp.build_pairs(trials, fake_encode, {"NCT00000002"})
    assert stats["excluded_eval_nct"] == 1 and stats["trials_used"] == 1
    assert all(r["nct_id"] == "NCT00000001" and r["fk_drop"] >= config.CTGOV_MIN_FK_DROP for r in recs)
    assert stats["candidates"] >= stats["final"]
