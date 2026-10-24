"""Tests for pipeline/pdq_pairs.py. No real network, no real embedding model."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from backend import config as cfg
from pipeline import pdq_pairs as P

FIX = Path(__file__).parent / "fixtures" / "pdq"
HP = (FIX / "hp_sample.html").read_text(encoding="utf-8")
PT = (FIX / "patient_sample.html").read_text(encoding="utf-8")


def fake_encoder(texts):
    """Bag-of-words hashed vectors: shared words => higher cosine. Deterministic."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, sum(map(ord, w)) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# --- citation stripping ------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("Surgery works.[1,2] It helps.", "Surgery works. It helps."),
    ("Studies [3-5] agree.", "Studies agree."),
    ("Good outcomes [Level of evidence: 1iiA] were seen.", "Good outcomes were seen."),
    ("Cryosurgery is used.[ ]; [, ] Evidence is limited.", "Cryosurgery is used. Evidence is limited."),
    ("No markers here at all.", "No markers here at all."),
])
def test_strip_citations(raw, expected):
    assert P.strip_citations(raw) == expected


def test_strip_citations_keeps_real_brackets():
    # Brackets with words (not just numbers) are real content.
    assert "[HNPCC]" in P.strip_citations("Lynch syndrome [HNPCC] is hereditary.")


# --- parsing -----------------------------------------------------------------
def test_parse_hp_drops_junk_and_boilerplate():
    secs = {s.heading: s for s in P.parse_sections(HP, is_patient=False)}
    assert "References" not in secs
    general = secs["General Information About Colon Cancer"].paragraphs
    assert len(general) == 1                      # evidence tag, lead-in ':', too-short all dropped
    assert "[" not in general[0] and "cit" not in general[0]
    # "Current Clinical Trials" boilerplate must not leak into stage IV
    assert not any("enrolling" in p for p in secs["Treatment of Stage IV and Recurrent Colon Cancer"].paragraphs)


def test_parse_patient_keeps_keypoint_sentences():
    secs = {s.heading: s for s in P.parse_sections(PT, is_patient=True)}
    paras = secs["General Information About Colon Cancer"].paragraphs
    assert any(p.startswith("Health history affects") for p in paras)
    assert "To Learn More About Colon Cancer" not in secs


# --- heading matching --------------------------------------------------------
def test_heading_score_exact_and_filler():
    assert P.heading_score("Treatment Option Overview for Colon Cancer", "Treatment Option Overview") >= cfg.PDQ_HEADING_MATCH_THRESHOLD
    assert P.heading_score("Stage Information for Colon Cancer", "Stages of Colon Cancer") >= 0.95


def test_stage_numerals_are_a_hard_gate():
    assert P.heading_score("Treatment of Stage I Colon Cancer", "Treatment of Stages I and II Colon Cancer") > 0.9
    assert P.heading_score("Treatment of Stage III Colon Cancer", "Treatment of Stages I and II Colon Cancer") == 0.0
    assert P.heading_score("Treatment of Stage I Colon Cancer", "Stages of Colon Cancer") == 0.0


def test_match_sections_many_to_one():
    hp = P.parse_sections(HP, False)
    pt = P.parse_sections(PT, True)
    matches = P.match_sections(hp, pt)
    by_pt = {p.heading: [h.heading for h in hs] for p, hs, _ in matches}
    assert by_pt["Treatment of Stages I and II Colon Cancer"] == ["Treatment of Stage I Colon Cancer"]
    assert "General Information About Colon Cancer" in by_pt


# --- alignment & filters -----------------------------------------------------
def test_greedy_align_is_one_to_one():
    sim = np.array([[0.9, 0.8], [0.85, 0.1]])
    pairs = P.greedy_align(sim, 0.5)
    assert pairs == [(0, 0, pytest.approx(0.9))]   # row 1 only had 0.85 with a used column
    assert P.greedy_align(sim, 0.95) == []


def test_pair_drop_reasons():
    src = " ".join(["word"] * 20)
    assert P.pair_drop_reason(src, " ".join(["w"] * 25), 0.7) is None
    assert P.pair_drop_reason(src, " ".join(["w"] * 31), 0.7) == "length_ratio"   # > 1.5x
    assert P.pair_drop_reason(src, " ".join(["w"] * 30), 0.7) is None             # exactly 1.5x ok
    assert P.pair_drop_reason(src, "tiny", 0.7) == "too_short"
    assert P.pair_drop_reason(src, " ".join(["w"] * 20), 0.99) == "near_identical"
    assert P.pair_drop_reason(" ".join(["x"] * (cfg.PDQ_MAX_SOURCE_WORDS + 1)), src, 0.7) == "source_too_long"


# --- split by page -----------------------------------------------------------
def test_split_has_no_page_leakage():
    counts = {f"page{i}": 10 + i * 3 for i in range(8)}
    split_of = P.assign_splits(counts)
    assert set(split_of) == set(counts)
    assert set(split_of.values()) == {"train", "val", "test"}   # exactly one split per page


def test_split_fractions_close_with_enough_pages():
    counts = {f"p{i}": 10 for i in range(10)}
    split_of = P.assign_splits(counts)
    got = {s: sum(counts[p] for p, v in split_of.items() if v == s) for s in ("train", "val", "test")}
    assert got == {"train": 80, "val": 10, "test": 10}


def test_split_is_deterministic():
    counts = {"a": 5, "b": 9, "c": 1, "d": 7, "e": 3}
    assert P.assign_splits(counts) == P.assign_splits(dict(counts))


# --- fetch (mocked) ----------------------------------------------------------
def test_fetch_retries_then_succeeds_and_caches(tmp_path):
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "<html>ok</html>"

    sleeps: list[float] = []
    html = P.load_html("https://x.gov/a", None, tmp_path, True, flaky, sleeps.append)
    assert html == "<html>ok</html>" and calls["n"] == 3
    assert sleeps[:2] == [cfg.PDQ_SLEEP_SECONDS, cfg.PDQ_SLEEP_SECONDS * 2]  # exponential backoff
    # second call is served from cache: getter must not run again
    P.load_html("https://x.gov/a", None, tmp_path, True, lambda u: 1 / 0, sleeps.append)


def test_no_fetch_requires_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_html("https://x.gov/missing", None, tmp_path, fetch=False)


def test_fetch_gives_up_after_max_retries():
    with pytest.raises(RuntimeError):
        P.fetch_with_retry("u", lambda u: 1 / 0, lambda s: None)


# --- end to end (fake network, fake encoder) ----------------------------------
def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_ALIGN_SIM_THRESHOLD", 0.25)
    pages = tmp_path / "pages.json"
    spec = [{"cancer_type": t, "hp_url": f"https://x/{t}/hp", "patient_url": f"https://x/{t}/pt"}
            for t in ("c1", "c2", "c3", "bad")]
    pages.write_text(json.dumps(spec))

    def getter(url):
        if "/bad/" in url:
            raise RuntimeError("404")
        return HP if url.endswith("/hp") else PT

    out, stats_p = tmp_path / "pairs.jsonl", tmp_path / "stats.json"
    stats = P.run(None, True, fake_encoder, pages, out, stats_p, tmp_path / "raw", getter, lambda s: None,
                 borderline_path=tmp_path / "border.json")
    assert "bad" in stats["failed_pages"]                       # failure recorded, run continued
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert recs and set(recs[0]) == {"id", "source", "target", "split", "cancer_type", "hp_url",
                                     "patient_url", "section", "similarity", "source_paragraphs"}
    splits_by_type: dict[str, set] = {}
    for r in recs:
        splits_by_type.setdefault(r["cancer_type"], set()).add(r["split"])
    assert all(len(v) == 1 for v in splits_by_type.values())    # no page spans two splits
    assert json.loads(stats_p.read_text())["total_pairs"] == len(recs)


# --- newer page layout & page-wide fallback ------------------------------------
def test_empty_h2_heading_takes_text_from_next_paragraph():
    html = ("<main><h2 id='s'></h2><p>Surgery</p>"
            "<p>Surgery is an operation to remove the cancer and some healthy tissue around it.</p></main>")
    secs = P.parse_sections(html, is_patient=True)
    assert secs[0].heading == "Surgery" and len(secs[0].paragraphs) == 1


def test_fallback_pairs_unmatched_patient_sections(monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_FALLBACK_SIM_THRESHOLD", 0.3)
    hp = ("<main><h2>Treatment Option Overview</h2><p>Surgery removes the tumor together with a margin "
          "of normal tissue around it during an operation.</p></main>")
    pt = ("<main><h2>Surgery</h2><p>Surgery removes the tumor together with some normal tissue around "
          "it during an operation.</p></main>")
    spec = P.PageSpec("x", "hp", "pt")
    stats: dict = {}
    recs = P.build_page_pairs(spec, hp, pt, fake_encoder, stats)
    assert stats["sections_matched"] == 0 and stats["pairs_kept_fallback"] == 1 and len(recs) == 1


def test_split_many_pages_uses_fast_path_and_stays_balanced():
    counts = {f"p{i}": 5 + (i * 7) % 23 for i in range(24)}
    split_of = P.assign_splits(counts)            # must not hang (3^24 is far too many)
    assert set(split_of) == set(counts) and set(split_of.values()) == {"train", "val", "test"}
    total = sum(counts.values())
    train = sum(c for p, c in counts.items() if split_of[p] == "train") / total
    assert 0.7 <= train <= 0.9


# --- many-to-one windows --------------------------------------------------------
def _items(*texts, group=0):
    return [(t, group, k) for k, t in enumerate(texts)]


def test_make_windows_only_adjacent_and_capped():
    items = _items("a b c d e f g h", "i j k l m n o p", "q r s t u v w x", "y z a b c d e f")
    idxs = [w[0] for w in P.make_windows(items, 3)]
    assert (0, 1, 2) in idxs and (0, 1, 2, 3) not in idxs       # cap of 3
    # a gap in positions (paragraph 1 already used) breaks adjacency
    gap = [("a b c d e f g h", 0, 0), ("i j k l m n o p", 0, 2)]
    assert all(len(ix) == 1 for ix, _ in P.make_windows(gap, 3))
    # different sections are never glued
    other = [("a b c d e f g h", 0, 0), ("i j k l m n o p", 1, 1)]
    assert all(len(ix) == 1 for ix, _ in P.make_windows(other, 3))


def test_make_windows_respects_max_source_words(monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_MAX_SOURCE_WORDS", 12)
    items = _items("one two three four five six seven eight", "nine ten eleven twelve thirteen fourteen fifteen sixteen")
    assert all(len(ix) == 1 for ix, _ in P.make_windows(items, 3))


def test_window_wins_when_target_covers_two_paragraphs(monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_ALIGN_SIM_THRESHOLD", 0.3)
    s1 = "Surgery removes the tumor together with a margin of normal tissue around it during an operation."
    s2 = "Chemotherapy uses drugs that travel through the blood to kill cancer cells that may have spread."
    target = ("Surgery removes the tumor and some normal tissue around it. Chemotherapy uses drugs in the "
              "blood to kill cancer cells that may have spread to other places in the body.")
    matches, _ = P.align_windows(_items(s1, s2), [target], fake_encoder, 0.3)
    assert matches[0][0] == (0, 1)               # both paragraphs joined


def test_window_not_used_when_single_is_as_good():
    s1 = "Surgery removes the tumor together with a margin of normal tissue around it during an operation."
    s2 = "Completely unrelated sentence about parking lots and bicycles near the hospital entrance today."
    matches, _ = P.align_windows(_items(s1, s2), [s1], fake_encoder, 0.3)
    assert matches[0][0] == (0,)                 # joining s2 would only dilute the match


def test_window_candidates_respect_length_ratio():
    src = " ".join(["alpha"] * 10)
    long_target = " ".join(["alpha"] * 40)       # 40 > 1.5 x (10+10) so even the pair is rejected
    matches, _ = P.align_windows(_items(src, src), [long_target], fake_encoder, 0.1)
    assert all(len(ix) == 1 or len(long_target.split()) <= 1.5 * 20 for ix, _, _ in matches)
    assert not any(len(ix) == 2 for ix, _, _ in matches)


def test_records_carry_source_paragraph_count(monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_ALIGN_SIM_THRESHOLD", 0.3)
    hp = ("<main><h2>Treatment Option Overview</h2>"
          "<p>Surgery removes the tumor together with a margin of normal tissue around it during an operation.</p>"
          "<p>Chemotherapy uses drugs that travel through the blood to kill cancer cells that may have spread.</p></main>")
    pt = ("<main><h2>Treatment Option Overview</h2><p>Surgery removes the tumor and some normal tissue around it. "
          "Chemotherapy uses drugs in the blood to kill cancer cells that may have spread to other places in the body.</p></main>")
    stats: dict = {}
    recs = P.build_page_pairs(P.PageSpec("x", "hp", "pt"), hp, pt, fake_encoder, stats)
    assert [r["source_paragraphs"] for r in recs] == [2] and stats["pairs_kept_multi_source"] == 1
    assert P.pair_drop_reason(recs[0]["source"], recs[0]["target"], recs[0]["similarity"]) is None


def test_sim_threshold_override_and_borderline_samples(tmp_path, monkeypatch):
    pages = tmp_path / "p.json"
    pages.write_text(json.dumps([{"cancer_type": "c", "hp_url": "https://x/c/hp", "patient_url": "https://x/c/pt"}]))
    getter = lambda u: HP if u.endswith("/hp") else PT          # noqa: E731
    monkeypatch.setattr(cfg, "PDQ_ALIGN_SIM_THRESHOLD", 0.99)  # default: nothing passes
    monkeypatch.setattr(cfg, "PDQ_BORDERLINE_BANDS", ((0.2, 0.6), (0.6, 1.01)))
    border = tmp_path / "b.json"
    stats = P.run(None, True, fake_encoder, pages, tmp_path / "o.jsonl", tmp_path / "s.json", tmp_path / "raw",
                  getter, lambda s: None, sim_threshold=0.2, borderline_path=border)
    assert stats["sim_threshold"] == 0.2
    bands = json.loads(border.read_text())
    assert set(bands) == {"0.20-0.60", "0.60-1.01"}
    for key, samples in bands.items():
        lo, hi = map(float, key.split("-"))
        assert len(samples) <= cfg.PDQ_BORDERLINE_COUNT
        assert all(lo <= s["similarity"] < hi for s in samples)
    assert any(bands.values())


# --- family split, near-duplicates, cross-references, numbers ---------------------
def test_run_splits_by_family_not_page(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PDQ_ALIGN_SIM_THRESHOLD", 0.25)
    spec = [{"cancer_type": f"p{i}", "family": f"fam{i // 2}", "hp_url": f"https://x/p{i}/hp",
             "patient_url": f"https://x/p{i}/pt"} for i in range(8)]
    pages = tmp_path / "pages.json"
    pages.write_text(json.dumps(spec))
    getter = lambda u: HP if u.endswith("/hp") else PT          # noqa: E731
    stats = P.run(None, True, fake_encoder, pages, tmp_path / "o.jsonl", tmp_path / "s.json", tmp_path / "raw",
                  getter, lambda s: None, borderline_path=tmp_path / "b.json")
    sp = stats["split"]["page_to_split"]
    for i in range(0, 8, 2):                                    # siblings always share a split
        assert sp[f"p{i}"] == sp[f"p{i + 1}"]
    assert set(sp.values()) == {"train", "val", "test"}


def _rec(target: str, split: str) -> dict:
    return {"target": target, "split": split}


def test_drop_near_duplicates_only_val_test_against_train():
    base = "Surgery removes the tumor together with some normal tissue around it during an operation."
    recs = [_rec(base, "train"), _rec(base.replace("tumor", "growth"), "val"),
            _rec("Radiation uses high energy beams to shrink cancer cells in a targeted area.", "test"),
            _rec(base, "train")]
    dropped = P.drop_near_duplicates(recs, 0.85)
    assert dropped == {"val": 1, "test": 0}
    assert [r["split"] for r in recs] == ["train", "test", "train"]    # train never dropped


def test_strip_xref_sentences():
    text = ("Surgery removes the tumor. See the Treatment Options section for details. "
            "Radiation kills cells. Visit our website to learn about clinical trials.")
    out, n = P.strip_xref_sentences(text)
    assert out == "Surgery removes the tumor. Radiation kills cells." and n == 2
    assert P.strip_xref_sentences("Nothing to strip here.") == ("Nothing to strip here.", 0)


def test_strip_learn_more_sentences():
    for line in ("Learn more about radiation therapy.", "Read more about it online.", "To learn more, ask your team.",
                 "Find out more on the site.", "Visit the NCI page for details."):
        assert P.strip_xref_sentences("Surgery removes the tumor. " + line) == ("Surgery removes the tumor.", 1), line
    assert P.strip_xref_sentences("Patients learn more slowly with age.")[1] == 0


def test_xref_strip_that_leaves_too_little_drops_pair():
    src = " ".join(["surgery"] * 20)
    stats = {"pairs_dropped": {}, "pairs_kept_heading": 0, "pairs_kept_multi_source": 0,
             "xref_sentences_stripped": 0, "pairs_with_xref_stripped": 0}
    tgt = "Surgery helps. For more information, see the booklet about surgery for patients."
    rec = P._emit(P.PageSpec("x", "h", "p", family="x"), src, tgt, 0.8, "s", "heading", 1, stats)
    assert rec is None and stats["pairs_dropped"] == {"xref_too_short": 1}
    assert stats["xref_sentences_stripped"] == 1


@pytest.mark.parametrize("src,tgt,bad", [
    ("Patients aged 50 to 75 may take 10 mg.", "People between 50 and 75 may take 10 mg.", set()),
    ("The dose is 10 mg.", "The dose is 20 mg.", {"20"}),
    ("It affects 1,000 people.", "It affects 1000 people.", set()),
    ("Cancer in stage III has spread.", "Cancer in stage 3 has spread.", set()),
    ("Cancer in stages I and II is small.", "Stage III cancer has spread.", {"stage 3"}),
    ("About half respond.", "About 50% respond.", {"50"}),
    ("No numbers at all.", "Still no numbers.", set()),
])
def test_unsupported_numbers(src, tgt, bad):
    assert P.unsupported_numbers(src, tgt) == bad


def test_pair_dropped_when_target_adds_number():
    src = " ".join(["chemotherapy"] * 20)
    assert P.pair_drop_reason(src, " ".join(["chemotherapy"] * 12) + " 5 cycles", 0.7) == "number_not_in_source"


def test_cached_encoder_embeds_each_text_once(tmp_path):
    calls: list[list[str]] = []

    def counting(texts):
        calls.append(list(texts))
        return fake_encoder(texts)

    cache = tmp_path / "c.pkl"
    enc = P.cached_encoder(counting, cache)
    a = enc(["alpha beta", "gamma delta", "alpha beta"])
    enc(["alpha beta", "new text"])
    assert [len(c) for c in calls] == [2, 1]               # duplicates and repeats not re-embedded
    enc.save()
    enc2 = P.cached_encoder(counting, cache)                # reload from disk
    assert np.allclose(enc2(["gamma delta"]), a[1:2], atol=1e-6) and len(calls) == 2
