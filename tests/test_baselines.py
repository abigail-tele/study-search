"""Metric functions in pipeline/baselines.py (no model downloads, no GPU)."""
import json

import pytest

from backend import config
from pipeline import baselines as B


def test_key_terms_numbers_units_stages_genes_drugs():
    t = B.key_terms("Women aged 18 years or older with stage IIIA HER2-positive disease got 50% cisplatin, 1,200 mg, PD-L1 high.")
    assert {"n:18", "u:years", "stage:3", "gene:her2", "n:50", "u:%", "drug:cisplatin", "n:1200", "u:mg", "gene:pdl1"} <= t
    assert "n:2" not in t and "n:1" not in t          # digits glued to HER2 / PD-L1 are not plain numbers


def test_stage_roman_and_arabic_are_the_same():
    assert B.key_terms("stage III") == B.key_terms("Stage 3") == {"stage:3"}
    assert B.key_terms("stages I and II") == {"stage:1", "stage:2"}


def test_drug_suffix_heuristic_and_plain_words():
    assert "drug:pembrolizumab" in B.key_terms("treated with pembrolizumab")
    assert B.key_terms("The doctor may use surgery and radiation") == set()


def test_prf_edges():
    assert B.prf(0, 0, 0) == (1.0, 1.0, 1.0)
    p, r, f = B.prf(2, 4, 8)
    assert (p, r) == (0.5, 0.25) and f == pytest.approx(1 / 3)
    assert B.prf(0, 3, 3)[2] == 0.0


def test_keyterm_prf_flags_invented_terms():
    src = ["Patients aged 18 years or older with stage II disease."]
    good = B.keyterm_prf(src, ["People 18 years or older with stage 2 disease."])
    assert (good["p"], good["r"], good["zero_halluc_pct"]) == (1.0, 1.0, 100.0)
    bad = B.keyterm_prf(src, ["People 21 years or older with stage 2 disease."])
    assert bad["p"] < 1.0 and bad["hallucinated"] == [["n:21"]] and bad["zero_halluc_pct"] == 0.0
    copy = B.keyterm_prf(src, src)
    assert copy["f"] == 1.0                      # copy-source is trivially faithful
    assert B.keyterm_prf(src, ["Nothing here."])["r"] == 0.0   # dropped every key term


def test_token_prf_micro_average():
    p, r, f = B.token_prf(["the cat sat", "a dog"], ["the cat ran", "a dog barked loudly"])
    assert p == pytest.approx(4 / 7) and r == pytest.approx(4 / 5)
    assert B.token_prf(["x y"], ["x y"]) == (1.0, 1.0, 1.0)


def test_rouge_prf_identical_and_disjoint():
    same = B.rouge_prf(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert same["rougeL"] == pytest.approx((1.0, 1.0, 1.0))
    none = B.rouge_prf(["alpha beta gamma"], ["delta epsilon zeta"])
    assert none["rouge1"] == (0.0, 0.0, 0.0)


def test_fewshot_prompt_drops_shots_until_it_fits():
    shots = [{"source": "s1 " * 10, "target": "t1"}, {"source": "s2 " * 10, "target": "t2"}, {"source": "s3 " * 10, "target": "t3"}]
    count = lambda t: len(t.split())  # noqa: E731
    full, n3 = B.fewshot_prompt("input text", shots, count, limit=10**6)
    assert n3 == 3 and full.endswith("Professional: input text\nPlain:")
    _, n_small = B.fewshot_prompt("input text", shots, count, limit=count(full) - 5)
    assert n_small < 3
    _, n0 = B.fewshot_prompt("input text", shots, count, limit=1)
    assert n0 == 0


def test_pick_shots_is_seeded_and_short():
    rows = [{"source": "word " * (i + 5), "target": "word " * 5} for i in range(100)]
    a, b = B.pick_shots(rows, 3), B.pick_shots(rows, 3)
    assert a == b and all(len(r["source"].split()) <= config.FEWSHOT_MAX_SOURCE_WORDS for r in a)


def test_replace_sections_keeps_other_splits(tmp_path):
    p = tmp_path / "e.txt"
    B._replace_sections(p, "val", "A")
    B._replace_sections(p, "trials", "B")
    B._replace_sections(p, "val", "A2")
    text = p.read_text()
    assert "A2" in text and "B" in text and text.count("=== split: val ===") == 1


def test_discover_checkpoints_skips_best_copies(tmp_path):
    for name in ("m1", "m2-best", "empty"):
        (tmp_path / name).mkdir()
    (tmp_path / "m1" / "model.safetensors").write_bytes(b"x")
    (tmp_path / "m2-best" / "model.safetensors").write_bytes(b"x")
    assert [p.name for p in B.discover_checkpoints(tmp_path)] == ["m1"]


def test_test_split_needs_opt_in():
    with pytest.raises(SystemExit):
        B.cmd_generate(B.argparse.Namespace(splits="val,test", allow_test=False, systems="copy-source", cpu=True))


def test_effective_grade_blends_output_and_source():
    assert B.effective_grade(100, 8.0, 15.0) == 8.0
    assert B.effective_grade(0, 8.0, 15.0) == 15.0
    assert abs(B.effective_grade(50, 8.0, 15.0) - 11.5) < 1e-9


def test_strip_think_removes_reasoning_block():
    assert B.strip_think("<think>\nhmm\n</think>\n\nPlain text.") == "Plain text."
    assert B.strip_think("Plain text.") == "Plain text."
