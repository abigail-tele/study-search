"""backend/support.py with a fake bag-of-words encoder (no model download)."""
import numpy as np

from backend import config, support


def fake_encode(texts):
    out = np.zeros((len(texts), 512))
    for i, t in enumerate(texts):
        for w in support.content_words(t):
            out[i, sum(map(ord, w)) % 512] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


SRC = "Patients must be 18 years or older. Patients must have breast cancer that has spread to the bone."


def test_supported_sentence_passes():
    ok, bad = support.sentence_support(SRC, "Patients must have breast cancer that has spread to the bone.", fake_encode)
    assert ok and bad == []


def test_invented_sentence_fails():
    ok, bad = support.sentence_support(SRC, "Women are more likely to develop lung disease than men.", fake_encode)
    assert not ok and len(bad) == 1


def test_mixed_output_reports_only_unsupported():
    out = "Patients must have breast cancer that has spread to the bone. Smoking causes many diseases of the heart."
    ok, bad = support.sentence_support(SRC, out, fake_encode)
    assert not ok and bad == ["Smoking causes many diseases of the heart."]


def test_word_fraction_blocks_similar_but_new_content(monkeypatch):
    monkeypatch.setattr(config, "SUPPORT_SIM_THRESHOLD", 0.0)   # similarity alone would pass
    monkeypatch.setattr(support, "glossary_words", lambda: frozenset())
    ok, _ = support.sentence_support(SRC, "Doctors prescribe immunotherapy chemotherapy radiation surgery.", fake_encode)
    assert not ok


def test_glossary_words_are_allowed(monkeypatch):
    monkeypatch.setattr(support, "glossary_words", lambda: frozenset({"spread", "bone", "metastatic"}))
    monkeypatch.setattr(config, "SUPPORT_SIM_THRESHOLD", 0.0)
    ok, _ = support.sentence_support("Metastatic disease present.", "Metastatic spread bone.", fake_encode)
    assert ok


def test_empty_output_is_trivially_supported():
    assert support.sentence_support(SRC, "", fake_encode) == (True, [])


def test_boilerplate_only_when_absent_from_source():
    assert support.has_boilerplate("Age criterion.", "There are different types of treatment for cancer.")
    assert not support.has_boilerplate("There are different types of treatment.", "There are different types of treatment.")
    assert not support.has_boilerplate(SRC, "Patients must be 18 or older.")
