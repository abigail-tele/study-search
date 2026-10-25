"""Build in-domain training pairs from ClinicalTrials.gov (public domain, NLM).

Why: PDQ pairs teach "professional cancer text -> patient text" but PDQ is review-article prose, not trial
prose. Many trials have BOTH a technical detailed description and a simpler brief summary that restates parts
of it. We align each brief-summary sentence (or 2-sentence window) to the best 1-3 consecutive detailed-
description sentences by MiniLM cosine, then keep only pairs where the target is easier to read, no longer,
adds no number/marker/stage/drug, and has no PDQ-style boilerplate. This is a weak but in-domain signal; the
filters are what keep it faithful (CLAUDE.md rule 3). No eligibility pairs: there is no plain reference for them.

Leakage: every NCT in data/trial_eval_set.jsonl is excluded; train/val is split by NCT id (hash).
Run (reads the DB read-only):  python -m pipeline.ctgov_pairs
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Callable

from backend import config
from backend.readability import fk_grade
from backend.support import has_boilerplate, split_sentences
from pipeline.baselines import key_terms
from pipeline.clean_pairs import is_generic
from pipeline.pdq_pairs import dedupe_records, unsupported_numbers


def eval_ncts(path: Path = config.TRIAL_EVAL_PATH) -> set[str]:
    """NCT ids used by the fixed trial evaluation set (must never appear in training)."""
    if not path.exists():
        return set()
    return {json.loads(l)["nct_id"] for l in path.open(encoding="utf-8") if l.strip()}


def split_of(nct_id: str, val_fraction: float = config.CTGOV_VAL_FRACTION) -> str:
    """Deterministic train/val assignment by NCT id, so all pairs of one trial land on the same side."""
    h = int(hashlib.md5(nct_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def windows(sents: list[str], max_n: int) -> list[tuple[int, int]]:
    """(start, n) for every run of 1..max_n consecutive sentences."""
    return [(i, n) for i in range(len(sents)) for n in range(1, max_n + 1) if i + n <= len(sents)]


def candidate_pairs(brief: list[str], detail: list[str], encode: Callable) -> list[dict]:
    """Best detailed-description window for each brief-summary piece.

    Selection uses mean-pooled sentence vectors (one encode per sentence, fast); the winner is then re-encoded
    exactly so the stored similarity is a true MiniLM cosine of the two texts.
    """
    import numpy as np

    if not brief or not detail:
        return []
    bw = windows(brief, config.CTGOV_TARGET_WINDOW)
    dw = windows(detail, config.CTGOV_MAX_WINDOW)
    bv, dv = np.asarray(encode(brief)), np.asarray(encode(detail))

    def pooled(vecs, spans):
        m = np.stack([vecs[i:i + n].mean(axis=0) for i, n in spans])
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    sim = pooled(bv, bw) @ pooled(dv, dw).T
    out = []
    for r, (bi, bn) in enumerate(bw):
        di, dn = dw[int(sim[r].argmax())]
        out.append({"target": " ".join(brief[bi:bi + bn]), "source": " ".join(detail[di:di + dn]),
                    "granularity": "sentence" if bn == 1 else "window"})
    texts = list(dict.fromkeys([o[k] for o in out for k in ("source", "target")]))
    vec = dict(zip(texts, np.asarray(encode(texts))))
    for o in out:
        o["similarity"] = float(vec[o["source"]] @ vec[o["target"]])
    return out


def drop_reason(source: str, target: str, sim: float) -> str | None:
    """Why a candidate is rejected, or None. Order only affects which filter gets the count."""
    sw, tw = len(source.split()), len(target.split())
    if sim < config.CTGOV_PAIR_SIM:
        return "low_similarity"
    if min(sw, tw) < config.CTGOV_MIN_TARGET_WORDS or sw > config.CTGOV_MAX_SOURCE_WORDS:
        return "length_bounds"
    if sim > config.PDQ_MAX_ALIGN_SIM:
        return "near_identical"
    if tw > config.CTGOV_MAX_LEN_RATIO * sw:
        return "length_ratio"
    if fk_grade(source) - fk_grade(target) < config.CTGOV_MIN_FK_DROP:
        return "fk_drop"
    if has_boilerplate(source, target) or is_generic(target, source):
        return "boilerplate"
    if key_terms(target) - key_terms(source) or unsupported_numbers(source, target):
        return "new_key_term"
    return None


def build_pairs(trials: list[tuple[str, str, str]], encode: Callable, exclude: set[str]) -> tuple[list[dict], dict]:
    """trials = [(nct_id, brief_summary, detailed_description)] -> (records, stats with per-filter counts)."""
    stats: Counter = Counter()
    recs: list[dict] = []
    for nct, brief, detail in trials:
        if nct in exclude:
            stats["excluded_eval_nct"] += 1
            continue
        b, d = split_sentences(brief or ""), split_sentences(detail or "")
        if not b or len(d) < 2:
            stats["trial_without_usable_text"] += 1
            continue
        stats["trials_used"] += 1
        for c in candidate_pairs(b, d, encode):
            stats["candidates"] += 1
            why = drop_reason(c["source"], c["target"], c["similarity"])
            if why:
                stats["dropped_" + why] += 1
                continue
            recs.append({"id": f"{nct}-{len(recs)}", "nct_id": nct, "source": c["source"], "target": c["target"],
                         "similarity": round(c["similarity"], 4), "granularity": c["granularity"],
                         "fk_drop": round(fk_grade(c["source"]) - fk_grade(c["target"]), 2),
                         "split": split_of(nct), "source_type": "ctgov", "page": nct})
    stats["kept_before_dedupe"] = len(recs)
    out = dict(stats)
    out["dedupe"] = dedupe_records(recs)
    out["final"] = len(recs)
    out["per_split"] = dict(Counter(r["split"] for r in recs))
    out["distinct_trials"] = len({r["nct_id"] for r in recs})
    return recs, out


def default_encode() -> Callable:
    """Local MiniLM (models/all-MiniLM-L6-v2), on a GPU if torch sees one (tiny model; fine to share)."""
    from sentence_transformers import SentenceTransformer

    path = config.EMBED_MODEL_DIR if config.EMBED_MODEL_DIR.exists() else config.EMBED_MODEL_NAME
    m = SentenceTransformer(str(path))
    return lambda t: m.encode(t, normalize_embeddings=True, show_progress_bar=False, batch_size=256)


def main() -> None:
    conn = sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
    trials = conn.execute("SELECT nct_id, brief_summary, detailed_description FROM trials "
                          "WHERE brief_summary IS NOT NULL AND detailed_description IS NOT NULL "
                          "AND length(detailed_description) > 200 ORDER BY nct_id").fetchall()
    recs, stats = build_pairs(trials, default_encode(), eval_ncts())
    stats.update(sim_threshold=config.CTGOV_PAIR_SIM, min_fk_drop=config.CTGOV_MIN_FK_DROP, trials_considered=len(trials))
    with config.CTGOV_PAIRS_PATH.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config.CTGOV_PAIRS_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
