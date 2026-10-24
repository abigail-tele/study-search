"""backend/support.py with a fake bag-of-words encoder (no model download)."""
import numpy as np

import pytest

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


# ---- inference-time hard-drop rules (generic claims, tautology, risk words)
@pytest.mark.parametrize("phrase", ["more likely to", "less likely to", "is a type of", "is a kind of", "is a disease in which",
                                    "are a group of", "is used to treat", "may cause", "can cause", "most common", "is called",
                                    "forms in", "there are different types of", "signs of", "symptoms of"])
def test_generic_phrase_dropped_unless_in_source(phrase):
    out = f"Patients {phrase} something."
    assert support.boilerplate_sentences("Age 18 or older.", out) == [out]
    assert support.boilerplate_sentences(f"Text with {phrase} inside.", out) == []


def test_tautology_detected():
    assert support.is_tautology("Radiation therapy is a type of radiation therapy that uses beams.")
    assert not support.is_tautology("Radiation therapy is a type of cancer treatment.")
    assert not support.is_tautology("Patients must have breast cancer.")
    assert support.boilerplate_sentences("Radiation was given.", "Radiation therapy is a type of radiation therapy.")


def test_new_risk_claim_dropped_only_if_absent_from_source():
    assert support.adds_risk_claim("Age 18 or older.", "Patients with this have a higher risk of relapse.")
    assert not support.adds_risk_claim("Patients at high risk of relapse.", "Patients with high risk of relapse can join.")
    assert support.adds_risk_claim("Age 18.", "This treatment is safe and effective.")
    assert not support.adds_risk_claim("Age 18.", "Patients must be 18 or older.")


# ---- general unbacked-definition rule
def test_unbacked_definition_dropped():
    src = "Prostate motion during radiotherapy is measured using gold markers."
    assert support.is_unbacked_definition(src, "Prostate Specific Antigen (PSA) is a protein that is found in the prostate.")
    assert support.boilerplate_sentences(src, "Prostate Specific Antigen (PSA) is a protein that is found in the prostate.")


def test_may_include_with_new_content_dropped():
    src = "Patients with colon adenocarcinoma stage I-III."
    assert support.is_unbacked_definition(src, "Treatment of colon adenocarcinoma may include surgery, radiation and immunotherapy.")


def test_source_backed_definition_kept():
    src = "A biomarker is a molecule that is found in blood and tissue. Patients must have a biomarker."
    assert not support.is_unbacked_definition(src, "A biomarker is a molecule that is found in blood and tissue.")
    assert not support.is_unbacked_definition(src, "Patients must have a biomarker.")


# ------------------------------------------------------------ direction_check
@pytest.mark.parametrize("source,output", [
    ("Age at least 18 years", "Age under 18 years"),
    ("Patients must be 18 years or older", "You must be younger than 18 years"),
    ("At least 2 prior lines", "At most 2 prior lines"),
    ("\u2265 2 lines of therapy", "\u2264 2 lines of therapy"),
    ("No prior chemotherapy", "Prior chemotherapy"),
    ("Patients must not be pregnant", "Patients must be pregnant"),
    ("Patients with brain metastases are excluded", "Patients with brain metastases are included"),
])
def test_direction_flips_are_caught(source, output):
    ok, problems = support.direction_check(source, output)
    assert not ok and problems


@pytest.mark.parametrize("source,output", [
    ("Patients must be 18 years or older", "You must be at least 18 years old"),
    ("Age 18+", "At least 18 years old"),
    ("No prior chemo", "You have not had chemotherapy before"),
    ("No prior chemotherapy", "You have not had prior chemotherapy"),
    ("Life expectancy of no more than 6 months", "Life expectancy of less than 6 months"),
    ("Patients must be 18 years or older and must not be pregnant", "You must be 18 or older. You must not be pregnant."),
    ("Patients must have breast cancer. Patients must not have had chemotherapy.", "You must have breast cancer."),
])
def test_direction_safe_paraphrases_pass(source, output):
    assert support.direction_check(source, output) == (True, [])


def test_direction_unrelated_sentences_stay_silent():
    assert support.direction_check("No prior chemotherapy", "You must be 18 years old")[0]


def test_direction_flipped_sentences_reports_only_bad_ones():
    src = "Patients must be at least 18 years old. Patients must have breast cancer."
    bad = support.direction_flipped_sentences(src, "You must have breast cancer. You must be under 18 years old.")
    assert bad == ["You must be under 18 years old."]
