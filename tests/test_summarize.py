"""Tests for backend/summarize.py. No model is ever downloaded: generation is faked."""
import json
import re
import sqlite3
from pathlib import Path

import pytest

from backend import config, db, summarize
from backend.readability import fk_grade, reading_grade

FIX = Path(__file__).parent / "fixtures" / "eligibility_samples.json"
SAMPLES = {s["nct_id"]: s for s in json.loads(FIX.read_text(encoding="utf-8"))}
HUMAN = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                          reason="HUMAN-WRITTEN: faithfulness_check")


# ----------------------------------------------------------------- split_eligibility
def test_split_real_bulleted_with_numbered_exclusions():
    inc, exc = summarize.split_eligibility(SAMPLES["NCT02453282"]["eligibility_text"])
    assert "Aged at least 18 years" in inc
    assert len(inc) == 5 and len(exc) == 4
    assert not any(c.endswith(":") for c in inc + exc)          # lead-in lines dropped
    assert exc[0].startswith("Mixed small-cell")                 # "1. " marker stripped


def test_split_real_unescapes_markdown_and_unicode():
    inc, exc = summarize.split_eligibility(SAMPLES["NCT04541212"]["eligibility_text"])
    assert any("\u226550%" in c for c in inc)
    assert "Known cardiomyopathy and/or LVEF <50%" in exc          # "\<" unescaped
    assert len(inc) == 6


def test_split_real_headers_glued_to_text_no_bullets():
    inc, exc = summarize.split_eligibility(SAMPLES["NCT07855627"]["eligibility_text"])
    assert len(inc) == 1 and inc[0].startswith("All women")
    assert "Prior breast/axillary surgery." in exc and len(exc) == 4


def test_split_real_with_context_prefixed_exclusions():
    inc, exc = summarize.split_eligibility(SAMPLES["NCT03289195"]["eligibility_text"])
    assert inc[0] == "Women age 18 years or older"
    assert len(inc) == 10 and len(exc) == 3
    assert exc[0].startswith("Medical history and concurrent disease: ")


def test_split_no_headers_is_all_inclusion():
    inc, exc = summarize.split_eligibility("* Age 18 or older\n* Able to sign consent")
    assert inc == ["Age 18 or older", "Able to sign consent"] and exc == []


def test_split_odd_casing_and_dash_bullets():
    inc, exc = summarize.split_eligibility("INCLUSION CRITERIA\n- ECOG 0-1\n- Adequate organ function\n"
                                           "exclusion criteria:\n- Pregnant\n- Active infection")
    assert inc == ["ECOG 0-1", "Adequate organ function"]
    assert exc == ["Pregnant", "Active infection"]


def test_split_nested_bullets_get_parent_context():
    inc, _ = summarize.split_eligibility(
        "Inclusion Criteria:\n* Adequate organ function:\n  * ANC > 1500\n  * Platelets > 100k\n* Age 18+")
    assert inc == ["Adequate organ function: ANC > 1500", "Adequate organ function: Platelets > 100k", "Age 18+"]


def test_split_wrapped_lines_are_joined():
    inc, _ = summarize.split_eligibility("Inclusion Criteria:\n* Histologically confirmed\n  breast cancer\n* Age 18+")
    assert inc[0] == "Histologically confirmed breast cancer"


def test_split_empty_and_none():
    assert summarize.split_eligibility("") == ([], [])
    assert summarize.split_eligibility(None) == ([], [])


def test_split_inclusion_word_inside_bullet_is_not_a_header():
    inc, exc = summarize.split_eligibility("Inclusion Criteria:\n* Meets all inclusion criteria of the parent study\n* Age 18+")
    assert len(inc) == 2 and exc == []


# ------------------------------------------------------------------------- chunking
def test_chunks_fit_budget_and_keep_all_text():
    text = " ".join(f"This is sentence number {i} about the study." for i in range(60))
    chunks = summarize.chunk_sentences(text, max_tokens=100, prefix="simplify: ")
    assert len(chunks) > 1
    for c in chunks:
        assert summarize._word_count_tokens("simplify: " + c) <= 100
    assert " ".join(chunks) == text


def test_chunks_short_text_is_one_chunk():
    assert summarize.chunk_sentences("One short sentence.") == ["One short sentence."]


def test_chunks_giant_sentence_is_cut_by_words():
    big = " ".join(["word"] * 400) + "."
    chunks = summarize.chunk_sentences(big, max_tokens=50)
    assert len(chunks) > 1 and all(summarize._word_count_tokens(c) <= 50 for c in chunks)


def test_chunks_use_supplied_counter():
    chunks = summarize.chunk_sentences("Alpha beta. Gamma delta. Epsilon zeta.",
                                       count=lambda s: len(s.split()), max_tokens=5, prefix="x ")
    assert chunks == ["Alpha beta. Gamma delta.", "Epsilon zeta."] or len(chunks) >= 2


def test_decimals_do_not_split_sentences():
    assert summarize.split_sentences("Dose is 2.5 mg daily. Next sentence.") == ["Dose is 2.5 mg daily.", "Next sentence."]


# ------------------------------------------------------------------ model dir / version
def test_resolve_falls_back_when_best_missing(tmp_path, monkeypatch):
    fallback = tmp_path / "flan-t5-pdq"; fallback.mkdir()
    monkeypatch.setattr(config, "SUMMARY_MODEL_DIR", tmp_path / "flan-t5-pdq-best")
    monkeypatch.setattr(config, "SUMMARY_FALLBACK_MODEL_DIR", fallback)
    assert summarize.resolve_model_dir() == fallback


def test_resolve_prefers_best(tmp_path, monkeypatch):
    best = tmp_path / "flan-t5-pdq-best"; best.mkdir()
    monkeypatch.setattr(config, "SUMMARY_MODEL_DIR", best)
    assert summarize.resolve_model_dir() == best


def test_model_version_reads_commit(tmp_path):
    d = tmp_path / "flan-t5-pdq"; d.mkdir()
    (d / "train_metadata.json").write_text(json.dumps({"git_commit": "abcdef1234567890"}))
    assert summarize.model_version(d) == "flan-t5-pdq@abcdef12"
    d2 = tmp_path / "other"; d2.mkdir()
    assert summarize.model_version(d2) == "other"


# --------------------------------------------------------------------------- glossary
def test_glossary_has_attribution_and_size():
    data = json.loads(config.GLOSSARY_PATH.read_text(encoding="utf-8"))
    assert "NCI Dictionary of Cancer Terms" in data["source"]
    assert len(data["terms"]) >= 60
    assert all(v["plain"] and v["url"].startswith("https://www.cancer.gov/") for v in data["terms"].values())
    assert not any(ch.isdigit() for v in data["terms"].values() for ch in v["plain"])  # swap never adds numbers


def test_swap_basic_and_case_insensitive():
    out = summarize.glossary_swap("Patients with Metastatic disease.")
    assert out.startswith("Patients with Metastatic (has spread")


def test_swap_first_occurrence_only():
    out = summarize.glossary_swap("Hepatic function. Severe hepatic failure.")
    assert out.count("(related to the liver)") == 1 and out.endswith("Severe hepatic failure.")


def test_swap_word_boundary():
    assert summarize.glossary_swap("The nonrenal precardiac") == "The nonrenal precardiac"
    assert summarize.glossary_swap("Renal-sparing") == "Renal-sparing"


def test_swap_prefers_longest_term():
    out = summarize.glossary_swap("prior radiation therapy")
    assert "radiation therapy (treatment that uses" in out


def test_swap_plural_form():
    assert "metastases (" in summarize.glossary_swap("Brain metastases present")


def test_swap_never_alters_numbers_or_original_text():
    src = "ECOG performance status 0-1; LVEF >=50%; creatinine clearance <60 mL/min; hepatic AST 2.5 x ULN"
    out = summarize.glossary_swap(src)
    assert re.findall(r"\d+(?:\.\d+)?", out) == re.findall(r"\d+(?:\.\d+)?", src)
    # original text is a subsequence: removing the inserted parentheticals restores it exactly
    stripped = re.sub(r" \((?:a score|related to)[^)]*\)", "", out)
    assert stripped == src


def test_swap_empty():
    assert summarize.glossary_swap("") == ""


# -------------------------------------------------------------------------- readability
def test_fk_grade_orders_text_difficulty():
    hard = "Histologically confirmed unresectable hepatocellular carcinoma refractory to antineoplastic chemotherapy."
    easy = "The cancer is in the liver. Surgery cannot remove it."
    assert fk_grade(hard) > fk_grade(easy)
    assert fk_grade("") == 0.0 and reading_grade("") == 0.0


# ---------------------------------------------------------------------- summarize_trial
def _fake_encode(texts):
    """Bag-of-words hashing vectors (normalised): similar wording -> high cosine, no model download."""
    import numpy as np
    out = np.zeros((len(texts), 256), dtype="float32")
    for r, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[r, hash(w) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def fake_model(monkeypatch):
    """Word-count tokens + a fake generator that returns 'PLAIN: <text>' for every chunk."""
    monkeypatch.setattr(summarize, "_token_counter", lambda: summarize._word_count_tokens)
    monkeypatch.setattr(summarize, "model_version", lambda *a, **k: "fake@0000")
    calls = []

    def gen(prompts):
        calls.append(prompts)
        return ["PLAIN: " + p[len(config.TASK_PREFIX):] for p in prompts]
    monkeypatch.setattr(summarize, "_generate_batch", gen)
    monkeypatch.setattr(summarize, "_encode", _fake_encode)
    return calls


TRIAL = {"nct_id": "NCT00000001",
         "brief_summary": "This trial tests whether drug X helps people with metastatic breast cancer.",
         "eligibility_text": "Inclusion Criteria:\n* Age 18 years or older\n* Hepatic function adequate\n"
                             "Exclusion Criteria:\n* Pregnant\n* Prior radiation therapy"}


def test_summarize_trial_all_pass(fake_model, monkeypatch):
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    s = summarize.summarize_trial(TRIAL)
    assert s["nct_id"] == "NCT00000001" and s["model_version"] == "fake@0000"
    assert s["what_is_tested"].startswith("PLAIN: This trial tests")
    assert s["who_can_join"]["include"][0] == "PLAIN: Age 18 years or older"
    # "Pregnant" is below the min-words floor: not sent to the model, glossary-swapped only
    assert s["who_can_join"]["exclude"][0].startswith("Pregnant (expecting a baby)")
    assert s["fallback_used"] is False and s["faithful"] is True
    assert s["original"]["summary"] == TRIAL["brief_summary"]
    assert set(s["reading_level"]) == {"before", "after"}
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", s["generated_at"])


def test_summarize_trial_failure_falls_back_to_glossary_original(fake_model, monkeypatch, caplog):
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (False, ["99 mg"]))
    with caplog.at_level("WARNING"):
        s = summarize.summarize_trial(TRIAL)
    assert s["fallback_used"] is True and s["faithful"] is True
    assert "PLAIN" not in json.dumps(s["who_can_join"]) and "PLAIN" not in s["what_is_tested"]
    assert "metastatic (has spread" in s["what_is_tested"]
    assert "NCT00000001" in caplog.text and "99 mg" in caplog.text


def test_summarize_trial_per_piece_fallback(fake_model, monkeypatch):
    monkeypatch.setattr(summarize, "faithfulness_check",
                        lambda s, o: (False, ["x"]) if "18 years" in s else (True, []))
    s = summarize.summarize_trial(TRIAL)
    assert s["who_can_join"]["include"][0] == "Age 18 years or older"      # fell back, numbers intact
    assert s["who_can_join"]["include"][1].startswith("PLAIN:")             # other pieces keep model text
    assert s["fallback_used"] is True


def test_summarize_trial_unimplemented_check_is_conservative(fake_model, caplog):
    # no monkeypatch: the real HUMAN-WRITTEN stub raises NotImplementedError -> treated as FAILED
    s = summarize.summarize_trial(TRIAL)
    assert s["fallback_used"] is True
    assert "PLAIN" not in json.dumps(s)
    assert "metastatic (has spread" in s["what_is_tested"]


def test_summarize_trial_empty_model_output_falls_back(monkeypatch):
    monkeypatch.setattr(summarize, "_token_counter", lambda: summarize._word_count_tokens)
    monkeypatch.setattr(summarize, "model_version", lambda *a, **k: "fake@0000")
    monkeypatch.setattr(summarize, "_generate_batch", lambda ps: ["" for _ in ps])
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    s = summarize.summarize_trial(TRIAL)
    assert s["fallback_used"] is True and s["what_is_tested"].startswith("This trial tests")


def test_summarize_trial_missing_text(fake_model):
    s = summarize.summarize_trial({"nct_id": "NCT2", "brief_summary": None, "eligibility_text": None})
    assert s["what_is_tested"] == "" and s["who_can_join"] == {"include": [], "exclude": []}
    assert s["fallback_used"] is False


def test_criteria_cap_shows_overflow_as_original(fake_model, monkeypatch):
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    monkeypatch.setattr(config, "SUMMARY_MAX_CRITERIA_PER_SIDE", 2)
    text = "Inclusion Criteria:\n" + "\n".join(f"* Criterion number {i} here" for i in range(4))
    s = summarize.summarize_trial({"nct_id": "N", "brief_summary": "", "eligibility_text": text})
    inc = s["who_can_join"]["include"]
    assert inc[0].startswith("PLAIN") and inc[2] == "Criterion number 2 here" and s["fallback_used"] is True


# ------------------------------------------------------------------------------ storage
def test_save_summary_roundtrips_through_get_summary(fake_model, monkeypatch):
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    db.upsert_trial(conn, {"nct_id": TRIAL["nct_id"], "brief_summary": TRIAL["brief_summary"],
                           "eligibility_text": TRIAL["eligibility_text"]})
    s = summarize.summarize_trial(TRIAL)
    summarize.save_summary(conn, s)
    got = db.get_summary(conn, TRIAL["nct_id"])
    assert got == s


# ------------------------------------------- faithfulness_check (HUMAN-WRITTEN) tests
# These document the rule for the teammate who writes it (CLAUDE.md rule 3).
# They are xfail(strict) so they flip to FAIL-until-fixed once the function exists:
# remove the marker as each behavior starts passing.
def fc(source, output):
    return summarize.faithfulness_check(source, output)


@HUMAN
def test_fc_identical_text_passes():
    assert fc("Age 18 years or older.", "Age 18 years or older.") == (True, [])


@HUMAN
def test_fc_words_without_protected_terms_may_change_freely():
    assert fc("Patients must have adequate organ function.", "You need healthy organs.")[0] is True


@HUMAN
def test_fc_integer_in_source_ok():
    assert fc("ANC >= 1500 per microliter", "Your ANC must be at least 1500.")[0] is True


@HUMAN
def test_fc_invented_number_fails_and_is_reported():
    ok, missing = fc("Age 18 or older", "You must be 21 or older")
    assert ok is False and any("21" in m for m in missing)


@HUMAN
def test_fc_decimals():
    assert fc("hemoglobin >= 9.5 g/dL", "hemoglobin of at least 9.5 g/dL")[0] is True
    assert fc("hemoglobin >= 9.5 g/dL", "hemoglobin of at least 9 g/dL")[0] is False   # 9 is not 9.5


@HUMAN
def test_fc_percentages():
    assert fc("LVEF >=50%", "heart pumping (LVEF) of 50% or more")[0] is True
    assert fc("LVEF >=50%", "heart pumping (LVEF) of 55% or more")[0] is False


@HUMAN
def test_fc_ranges():
    assert fc("Age 18-75 years", "between 18 and 75 years old")[0] is True
    assert fc("Age 18-75 years", "between 18-80 years old")[0] is False


@HUMAN
def test_fc_ages_with_units():
    assert fc("18 years or older", "at least 18 years old")[0] is True
    assert fc("18 years or older", "at least 18 months old")[0] is False   # wrong unit


@HUMAN
def test_fc_durations():
    assert fc("no chemotherapy within 6 months", "no chemo in the last 6 months")[0] is True
    assert fc("no chemotherapy within 6 months", "no chemo in the last 6 weeks")[0] is False


@HUMAN
def test_fc_dose_units():
    assert fc("dose 75 mg/m2 every 3 weeks", "75 mg/m2 every 3 weeks")[0] is True
    assert fc("dose 75 mg/m2 every 3 weeks", "75 mg/kg every 3 weeks")[0] is False
    assert fc("creatinine clearance <60 mL/min", "kidney function under 60 mL/min")[0] is True


@HUMAN
def test_fc_drug_names_in_source_ok_case_insensitive():
    assert fc("Prior Pembrolizumab allowed", "You may have had pembrolizumab before")[0] is True


@HUMAN
def test_fc_invented_drug_fails():
    ok, missing = fc("Treatment with trastuzumab", "Treatment with trastuzumab and pembrolizumab")
    assert ok is False and any("pembrolizumab" in m.lower() for m in missing)


@HUMAN
def test_fc_markers():
    assert fc("HER2-positive breast cancer", "breast cancer with too much HER2 protein")[0] is True
    assert fc("HER2-positive breast cancer", "breast cancer with EGFR changes")[0] is False
    assert fc("EGFR exon 19 deletion", "an EGFR exon 19 change")[0] is True
    assert fc("EGFR exon 19 deletion", "an EGFR exon 21 change")[0] is False
    assert fc("BRCA1 mutation", "a BRCA1 gene change")[0] is True
    assert fc("BRCA1 mutation", "a BRCA2 gene change")[0] is False
    assert fc("PD-L1 expression >= 1%", "PD-L1 level of 1% or more")[0] is True
    assert fc("KRAS G12C mutation", "a KRAS G12C change")[0] is True
    assert fc("KRAS G12C mutation", "a KRAS G12D change")[0] is False


@HUMAN
def test_fc_stage_roman_arabic_equivalence():
    assert fc("stage III disease", "stage 3 disease")[0] is True
    assert fc("stage 3 disease", "stage III disease")[0] is True
    assert fc("stage IV", "stage 4")[0] is True
    assert fc("stage III disease", "stage 2 disease")[0] is False


@HUMAN
def test_fc_stage_letter_suffix_must_match():
    assert fc("stage IIIB or IIIC", "stage 3B or 3C")[0] is True
    assert fc("stage IIIB", "stage IIIC")[0] is False
    assert fc("stage III", "stage IIIB")[0] is False        # model made it more specific


@HUMAN
def test_fc_case_insensitive():
    assert fc("HER2 NEGATIVE, AGE 18 YEARS", "her2 negative, age 18 years")[0] is True


@HUMAN
def test_fc_numbers_are_whole_tokens():
    assert fc("ANC 1500", "ANC 150")[0] is False            # not a substring match
    assert fc("ANC 150", "ANC 1500")[0] is False


@HUMAN
def test_fc_number_words_in_output_not_required_but_digits_checked():
    assert fc("ECOG 0-1", "ECOG score of 0 or 1")[0] is True


@HUMAN
def test_fc_empty_output_is_not_faithful():
    assert fc("Age 18 or older", "")[0] is False
    assert fc("Age 18 or older", "   ")[0] is False


@HUMAN
def test_fc_empty_source_with_no_terms_in_output():
    assert fc("", "You can join.")[0] is True


@HUMAN
def test_fc_returns_missing_list_empty_when_ok():
    ok, missing = fc("take 5 mg", "take 5 mg")
    assert ok is True and missing == []


@HUMAN
def test_fc_reports_every_missing_term():
    ok, missing = fc("Age 18", "Age 21, taking 5 mg of pembrolizumab")
    assert ok is False and len(missing) >= 3


# ------------------------------------------------------- support check + boilerplate guard
SRC = "Patients must have breast cancer. Patients must not have taken chemotherapy in the past. Patients must be healthy enough to join."


def _run_screen(monkeypatch, model_output, source=SRC):
    """Run one piece through _rewrite_pieces with a fake model that returns `model_output`."""
    monkeypatch.setattr(summarize, "_token_counter", lambda: summarize._word_count_tokens)
    monkeypatch.setattr(summarize, "_generate_batch", lambda ps: [model_output for _ in ps])
    monkeypatch.setattr(summarize, "_encode", _fake_encode)
    monkeypatch.setattr(summarize, "faithfulness_check", lambda s, o: (True, []))
    (out,), fb = summarize._rewrite_pieces([source], "NCT9")
    return out, fb


def test_supported_output_is_kept(monkeypatch):
    out, fb = _run_screen(monkeypatch, "You must have breast cancer. You must not have taken chemotherapy before.")
    assert out.startswith("You must have breast cancer") and fb is False


def test_unsupported_sentence_is_dropped_rest_kept(monkeypatch, caplog):
    bad = "Many women enjoy gardening on sunny weekends."
    with caplog.at_level("WARNING"):
        out, fb = _run_screen(monkeypatch, f"You must have breast cancer. You must not have taken chemotherapy before. {bad}")
    assert bad not in out and "breast cancer" in out and "chemotherapy" in out
    assert fb is False                                  # shown text is still checked model text
    assert "gardening" in caplog.text and "NCT9" in caplog.text


def test_fully_unsupported_falls_back_to_glossary_original(monkeypatch):
    out, fb = _run_screen(monkeypatch, "Many women enjoy gardening on sunny weekends. Bananas are yellow fruit.")
    assert fb is True and out.startswith("Patients must have breast cancer")
    assert "gardening" not in out


def test_mostly_unsupported_falls_back_whole_piece(monkeypatch):
    # 1 supported of 3 sentences is below SUMMARY_MIN_KEPT_FRACTION (0.5): whole piece falls back
    out, fb = _run_screen(monkeypatch, "You must have breast cancer. Bananas are yellow fruit. Gardening is relaxing on weekends.")
    assert fb is True and "Bananas" not in out and out.startswith("Patients must have")


def test_boilerplate_opener_sentence_is_dropped(monkeypatch):
    src = "Patients must have breast cancer. Patients must not have taken chemotherapy."
    out, fb = _run_screen(monkeypatch,
                          "There are different types of breast cancer. You must have breast cancer. You must not have taken chemotherapy.",
                          source=src)
    assert "different types" not in out and "chemotherapy" in out and fb is False


def test_boilerplate_allowed_when_source_has_it(monkeypatch):
    src = "There are different types of breast cancer. Patients must have breast cancer."
    out, _ = _run_screen(monkeypatch, "There are different types of breast cancer. You must have breast cancer.", source=src)
    assert "different types" in out


def test_faithfulness_stub_still_forces_fallback_even_if_supported(monkeypatch):
    monkeypatch.setattr(summarize, "_token_counter", lambda: summarize._word_count_tokens)
    monkeypatch.setattr(summarize, "_generate_batch", lambda ps: ["You must have breast cancer." for _ in ps])
    monkeypatch.setattr(summarize, "_encode", _fake_encode)
    (out,), fb = summarize._rewrite_pieces([SRC], "NCT9")   # real stub raises NotImplementedError
    assert fb is True and out.startswith("Patients must have breast cancer. Patients must not")
