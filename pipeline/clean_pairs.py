"""Clean the PDQ training targets of generic boilerplate and unsupported sentences.

Why: the first fine-tuned models learned PDQ habits ("X is a disease in which...", "There are different
types of treatment for...") and wrote them even when the trial text never said so. Here, every TARGET
sentence is removed when
  1. it matches a generic PDQ intro/definition pattern (config.PDQ_GENERIC_TARGET_PATTERNS) that the
     SOURCE does not contain, or
  2. backend.support says it is not supported by the source (same logic as at run time, but with the looser
     PDQ_CLEAN_* thresholds: patient wording differs from professional wording, so word overlap is not required).
A pair is dropped if too little target is left; the cleaned set is then deduplicated again.
Inputs are the already-deduplicated paragraph and sentence files; nothing existing is overwritten.

Run (CPU, needs sentence-transformers):  python -m pipeline.clean_pairs
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from backend import config
from backend import support
from pipeline.pdq_pairs import dedupe_records

_GENERIC_RES = [re.compile(p, re.I) for p in config.PDQ_GENERIC_TARGET_PATTERNS]


def is_generic(sentence: str, source: str) -> bool:
    """True if the sentence is a generic PDQ intro/definition that the source does not itself contain."""
    src = source.lower()
    return any(r.search(sentence) and not r.search(src) for r in _GENERIC_RES)


def clean_records(records: list[dict], encode: Callable | None = None) -> tuple[list[dict], dict]:
    """Return (cleaned records, stats). Records are copied, not changed in place."""
    stats = {"input": len(records), "sentences_removed_generic": 0, "sentences_removed_unsupported": 0,
             "pairs_changed": 0, "dropped_too_short": 0, "dropped_ratio": 0}
    prepared = []
    for r in records:
        sents = support.split_sentences(r["target"]) or [r["target"]]
        prepared.append((r, sents, [is_generic(s, r["source"]) for s in sents]))
    # support is judged per pair on the sentences that survived the generic filter
    pairs = [(r["source"], " ".join(s for s, g in zip(sents, gen) if not g)) for r, sents, gen in prepared]
    old = (config.SUPPORT_SIM_THRESHOLD, config.SUPPORT_WORD_FRACTION)
    config.SUPPORT_SIM_THRESHOLD, config.SUPPORT_WORD_FRACTION = config.PDQ_CLEAN_SUPPORT_SIM, config.PDQ_CLEAN_WORD_FRACTION
    try:
        res = support.sentence_support_many(pairs, encode)
    finally:
        config.SUPPORT_SIM_THRESHOLD, config.SUPPORT_WORD_FRACTION = old
    out: list[dict] = []
    for (r, sents, gen), (_, bad, _n) in zip(prepared, res):
        bad_set = set(bad)
        keep = [s for s, g in zip(sents, gen) if not g and s not in bad_set]
        stats["sentences_removed_generic"] += sum(gen)
        stats["sentences_removed_unsupported"] += sum(1 for s, g in zip(sents, gen) if not g and s in bad_set)
        if len(keep) != len(sents):
            stats["pairs_changed"] += 1
        new = " ".join(keep)
        floor = config.PDQ_SENT_MIN_WORDS if r.get("granularity") == "sentence" else config.PDQ_MIN_WORDS
        if len(new.split()) < floor:
            stats["dropped_too_short"] += 1
            continue
        if len(new.split()) > config.PDQ_MAX_LEN_RATIO * len(r["source"].split()):
            stats["dropped_ratio"] += 1
            continue
        out.append({**r, "target": new, "granularity": r.get("granularity", "paragraph")})
    stats["after_clean"] = len(out)
    stats["dedupe_dropped"] = dedupe_records(out)
    stats["final"] = len(out)
    stats["per_split"] = {s: sum(1 for r in out if r["split"] == s) for s in ("train", "val", "test")}
    return out, stats


def main() -> None:
    para = [json.loads(l) for l in config.PDQ_PAIRS_PATH.open(encoding="utf-8") if l.strip()]
    sent = [json.loads(l) for l in config.PDQ_SENT_PAIRS_PATH.open(encoding="utf-8") if l.strip()]
    cleaned, stats = clean_records(para + sent)
    all_path = config.DATA_DIR / "pdq_pairs_clean.jsonl"
    para_path = config.DATA_DIR / "pdq_pairs_clean_para.jsonl"
    for path, rows in ((all_path, cleaned), (para_path, [r for r in cleaned if r["granularity"] == "paragraph"])):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["paragraph_only_per_split"] = {s: sum(1 for r in cleaned if r["granularity"] == "paragraph" and r["split"] == s)
                                         for s in ("train", "val", "test")}
    (config.DATA_DIR / "pdq_clean_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
