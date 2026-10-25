"""Tests for backend/match.py and backend/overview.py (pure functions; a fake encoder stands in for MiniLM)."""
import numpy as np
import pytest

from backend import config, match, overview

TRIAL = {"nct_id": "NCT05000001", "min_age_years": 18.0, "max_age_years": 75.0, "sex": "ALL"}
ELIG = """Inclusion Criteria:
* HER2-positive breast cancer, Stage IIIA
* No prior chemotherapy for this cancer
* Adequate kidney function

Exclusion Criteria:
* Pregnant or breastfeeding
* Uncontrolled diabetes mellitus
* Active infection"""

CONCEPTS = {"sugar": 0, "glucose": 0, "diabetes": 0, "kidney": 1, "renal": 1, "infection": 2}


def fake_encoder(texts):
    """Each text becomes a unit vector on the first concept word it contains (else a 'nothing' axis)."""
    out = np.zeros((len(texts), 4), dtype="float32")
    for i, t in enumerate(texts):
        axis = next((a for w, a in CONCEPTS.items() if w in t.lower()), 3)
        out[i, axis] = 1.0
    return out


def run(**kw):
    kw.setdefault("encoder", None)
    return match.match_trial(TRIAL, ELIG, **kw)


# --- parse_conditions ---------------------------------------------------------------------------
def test_parse_splits_lowercases_and_trims():
    got = match.parse_conditions("HER2 positive, Stage 3; had chemotherapy and Diabetes\n  pregnant ")
    assert got == ["her2 positive", "stage 3", "had chemotherapy", "diabetes", "pregnant"]


def test_parse_limits_and_dedupes():
    assert len(match.parse_conditions(",".join(f"t{i}" for i in range(30)))) == config.MATCH_MAX_TERMS
    assert all(len(t) <= config.MATCH_MAX_TERM_CHARS for t in match.parse_conditions("x" * 200))
    assert match.parse_conditions("a, A, a") == ["a"]
    assert match.parse_conditions(None) == [] and match.parse_conditions(" ,; ") == []


# --- age / sex ----------------------------------------------------------------------------------
@pytest.mark.parametrize("age,expected", [(None, "unknown"), (17, "outside"), (18, "ok"), (75, "ok"), (76, "outside")])
def test_age(age, expected):
    assert run(age=age)["age"] == expected


def test_age_missing_limits_mean_no_limit():
    assert match.age_status(None, None, 5) == "ok"
    assert match.age_status(None, 10, 11) == "outside"


@pytest.mark.parametrize("trial_sex,sex,expected", [
    ("ALL", "female", "ok"), ("FEMALE", "female", "ok"), ("FEMALE", "male", "no"),
    ("MALE", "female", "no"), ("FEMALE", "any", "unknown"), ("FEMALE", None, "unknown")])
def test_sex(trial_sex, sex, expected):
    assert match.sex_status(trial_sex, sex) == expected


def test_age_outside_or_sex_no_is_likely_not():
    assert run(age=10)["summary_label"] == match.LABEL_NOT
    r = match.match_trial({**TRIAL, "sex": "FEMALE"}, ELIG, sex="male", encoder=None)
    assert r["sex"] == "no" and r["summary_label"] == match.LABEL_NOT


# --- phrase / synonym matching ------------------------------------------------------------------
def test_phrase_match_with_stage_and_her2_spellings():
    r = run(conditions=["her2+", "stage 3"])
    terms = {m["term"] for m in r["mentioned_in_requirements"]}
    assert terms == {"her2+", "stage 3"}
    assert run(conditions=["stage iii"])["mentioned_in_requirements"]


def test_stage_3_does_not_match_stage_4():
    r = match.match_trial(TRIAL, "Inclusion Criteria:\n* Stage IV disease", conditions=["stage 3"], encoder=None)
    assert not r["mentioned_in_requirements"]


def test_whole_word_only():
    r = match.match_trial(TRIAL, "Inclusion Criteria:\n* Pregnancy test negative", conditions=["gnan"], encoder=None)
    assert not r["mentioned_in_requirements"] and not r["possible_conflicts"]


def test_synonym_hits_exclusion_and_label_check():
    r = run(conditions=["pregnant", "diabetes"])
    assert [m["criterion"] for m in r["mentioned_in_exclusions"]] == ["Pregnant or breastfeeding", "Uncontrolled diabetes mellitus"]
    assert r["summary_label"] == match.LABEL_CHECK


def test_synonym_pregnancy_word():
    assert run(conditions=["pregnancy"])["mentioned_in_exclusions"]


def test_no_terms_possible_match():
    r = run(age=40, sex="female")
    assert r["summary_label"] == match.LABEL_POSSIBLE and r["mentioned_in_requirements"] == []


# --- embedding matching (fake encoder) ----------------------------------------------------------
def test_meaning_match_when_no_shared_words():
    r = run(conditions=["high blood sugar"], encoder=fake_encoder)   # 'sugar' and 'diabetes' share a concept axis
    assert any("diabetes" in m["criterion"].lower() for m in r["mentioned_in_exclusions"])


def test_meaning_match_needs_threshold(monkeypatch):
    monkeypatch.setattr(config, "MATCH_SIM_THRESHOLD", 1.01)
    assert not run(conditions=["high blood sugar"], encoder=fake_encoder)["mentioned_in_exclusions"]


def test_digit_terms_never_use_meaning_match():
    enc = lambda texts: np.ones((len(texts), 2), dtype="float32") / np.sqrt(2)   # everything identical
    r = match.match_trial(TRIAL, "Inclusion Criteria:\n* Stage IV disease", conditions=["stage 3"], encoder=enc)
    assert not r["mentioned_in_requirements"]


def test_criteria_encoded_once_per_trial():
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_encoder(texts)
    t = {**TRIAL, "nct_id": "NCT07777777"}
    for terms in (["a b"], ["c d"], ["e f"]):
        match.match_trial(t, ELIG, conditions=terms, encoder=counting)
    # each call encodes the 1 term; criteria batches (include 3, exclude 3) happen on the first call only
    assert calls.count(3) == 2


# --- negation -----------------------------------------------------------------------------------
def test_negated_inclusion_is_a_conflict():
    r = run(conditions=["had chemotherapy"])
    assert r["possible_conflicts"] == [{"term": "had chemotherapy", "criterion": "No prior chemotherapy for this cancer"}]
    assert r["mentioned_in_requirements"] == [] and r["summary_label"] == match.LABEL_CHECK


def test_user_absence_is_ignored_not_matched():
    r = run(conditions=["no chemotherapy", "never smoked"])
    assert r["ignored_terms"] == ["no chemotherapy", "never smoked"] and not r["possible_conflicts"]


def test_flag_criteria_marks_icons():
    r = run(conditions=["her2 positive", "had chemotherapy", "pregnant"])
    inc, exc = match.split_criteria(ELIG)
    flags = match.flag_criteria(inc, exc, r)
    assert [c["flag"] for c in flags["include"]] == ["match", "concern", None]
    assert [c["flag"] for c in flags["exclude"]] == ["concern", None, None]
    assert flags["include"][0]["original"] == inc[0]


# --- rule 7: wording -----------------------------------------------------------------------------
def test_labels_never_say_eligible():
    for text in list(match.LABEL_TEXT.values()) + list(match.LABEL_ORDER):
        assert "eligible" not in text.lower() and "qualify" not in text.lower()
    for kw in ({}, {"age": 10}, {"conditions": ["pregnant"]}):
        r = run(**kw)
        assert "eligible" not in (r["summary_label"] + r["label_text"]).lower()
        assert "study team" in r["label_text"].lower()


def test_sort_by_match_is_stable_within_groups():
    rs = [{"nct_id": c, "match": {"summary_label": lab}} for c, lab in
          [("A", match.LABEL_NOT), ("B", match.LABEL_POSSIBLE), ("C", match.LABEL_CHECK), ("D", match.LABEL_POSSIBLE)]]
    assert [r["nct_id"] for r in match.sort_by_match(rs)] == ["B", "D", "C", "A"]


# --- overview -----------------------------------------------------------------------------------
STUDY = {"phase": "PHASE2", "status": "RECRUITING", "min_age": "18 Years", "max_age": "75 Years", "sex": "ALL",
         "conditions": ["Breast Cancer"], "sponsor": "Example Cancer Center", "enrollment": 120,
         "sites": [{"city": "Springfield", "state": "IL", "lat": 39.78, "lon": -89.65, "facility": "x", "zip": "1"},
                   {"city": "Peoria", "state": "IL", "lat": 40.69, "lon": -89.59, "facility": "y", "zip": "2"}]}


def test_overview_shape_and_sentence():
    o = overview.study_overview(STUDY)
    assert set(o) == {"sentence", "facts"}
    assert o["sentence"] == ("A Phase 2 study, now recruiting, for people ages 18–75 with Breast Cancer. "
                             "Run by Example Cancer Center at 2 sites.")
    assert {"label": "Phase", "value": "Phase 2"} in o["facts"]
    assert all(set(f) == {"label", "value"} for f in o["facts"])


def test_overview_nearest_site_with_zip(monkeypatch):
    monkeypatch.setattr(overview.geo, "nearest_site", lambda sites, z: {"city": "Springfield", "state": "IL", "distance_mi": 12.4})
    o = overview.study_overview(STUDY, "62701")
    assert "(nearest: Springfield, IL — 12 mi)" in o["sentence"]


def test_overview_handles_missing_fields():
    o = overview.study_overview({"phase": "NA", "status": "UNKNOWN", "sex": "FEMALE"})
    assert o["sentence"].startswith("A study, unknown, for women all ages")
    assert "Phase" not in [f["label"] for f in o["facts"]]
