"""pipeline/clean_pairs.py with a fake encoder (no model download)."""
import numpy as np

from pipeline import clean_pairs as C
from backend import support


def fake_encode(texts):
    out = np.zeros((len(texts), 512))
    for i, t in enumerate(texts):
        for w in support.content_words(t):
            out[i, sum(map(ord, w)) % 512] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def _r(src, tgt, split="train", gran="paragraph", sim=0.8):
    return {"source": src, "target": tgt, "split": split, "granularity": gran, "similarity": sim}


def test_is_generic_only_when_source_lacks_it():
    assert C.is_generic("There are different types of treatment for colon cancer.", "Surgery is common.")
    assert not C.is_generic("There are different types of treatment.", "There are different types of treatment here.")
    assert not C.is_generic("Surgery removes the tumor.", "anything")


def test_clean_removes_generic_and_unsupported_sentences():
    src = "Surgery removes the tumor from the colon today. Chemotherapy kills cancer cells after surgery is done."
    tgt = ("There are different types of treatment for colon cancer. Surgery removes the tumor from the colon today. "
           "Smoking raises the risk of heart attacks in older people.")
    out, st = C.clean_records([_r(src, tgt)], fake_encode)
    assert out[0]["target"] == "Surgery removes the tumor from the colon today."
    assert st["sentences_removed_generic"] == 1 and st["sentences_removed_unsupported"] == 1


def test_clean_drops_pairs_left_too_short_and_dedupes():
    src = "Patients receive radiation therapy to the pelvis every day for five weeks."
    short = _r(src, "There are different types of treatment for rectal cancer, and many more exist today.")
    good = _r("Doctors remove the tumor with surgery and check nearby lymph nodes carefully.",
              "Doctors remove the tumor with surgery and check nearby lymph nodes carefully.")
    twin = dict(good)
    out, st = C.clean_records([short, good, twin], fake_encode)
    assert st["dropped_too_short"] == 1 and len(out) == 1 and st["dedupe_dropped"]["exact_pair"] == 1
