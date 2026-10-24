"""Score one or more checkpoints on the FIXED validation set (data/pdq_val_fixed.jsonl).

Why: val loss from runs trained on different data / thresholds is NOT comparable
(each run's own val split differs). This tool freezes one set of 125 val pairs
and reports the same four numbers for every checkpoint:
  * val loss on the fixed set
  * ROUGE-L of generated text vs the NCI patient text (target)
  * mean Flesch-Kincaid grade of outputs vs sources (lower = easier)
  * number faithfulness: % of outputs whose every number / stage numeral also
    appears in the source (hard rule 3, numbers only; the full check lives in
    backend/summarize.py)
Only the "val" split file is read; the "test" split stays reserved for evaluate.py.

Run:  python -m pipeline.compare_checkpoints checkpoints/flan-t5-pdq-baseline checkpoints/...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from backend import config
from backend.readability import count_syllables, fk_grade  # noqa: F401  (shared with baselines.py)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_STAGE_RE = re.compile(r"\bstages?\b((?:\s*(?:,|and\b|or\b|to\b|through\b|-|–)?\s*(?:0|iv|i{1,3}|[1-4])[abc]?\b)+)", re.I)
_ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def _stages(text: str) -> set[int]:
    out: set[int] = set()
    for m in _STAGE_RE.finditer(text):
        for tok in re.findall(r"\b(0|iv|i{1,3}|[1-4])[abc]?\b", m.group(1), re.I):
            out.add(_ROMAN[tok.lower()])
    return out


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(_STAGE_RE.sub(" ", text))}


def numbers_faithful(source: str, output: str) -> bool:
    """True if every number and stage numeral in output also appears in source."""
    return _numbers(output) <= _numbers(source) and _stages(output) <= _stages(source)


def load_fixed(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert all(r["split"] == "val" for r in rows), "fixed set must contain val rows only"
    return rows


def generate(model, tok, prompts: list[str], device, use_bf16: bool, bs: int = 16) -> list[str]:
    """Beam search with the same settings the app uses (config). `prompts` are full model inputs."""
    import torch
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [""] * len(prompts)
    model.eval()
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([prompts[i] for i in idx], max_length=config.MAX_INPUT_TOKENS,
                  truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            ids = model.generate(**enc, num_beams=config.NUM_BEAMS, no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                                 max_new_tokens=config.MAX_OUTPUT_TOKENS)
        for i, text in zip(idx, tok.batch_decode(ids, skip_special_tokens=True)):
            outs[i] = text
    return outs


def evaluate_checkpoint(ckpt: Path, rows: list[dict], cpu: bool = False) -> dict:
    """Return the metrics dict for one checkpoint (also keeps outputs for inspection)."""
    from rouge_score import rouge_scorer
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from pipeline.train import evaluate_loss, make_collate, pick_device

    device, use_bf16 = pick_device(cpu)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device)
    loss = evaluate_loss(model, DataLoader(rows, batch_size=8, collate_fn=make_collate(tok)), device, use_bf16)
    outs = generate(model, tok, [config.TASK_PREFIX + r["source"] for r in rows], device, use_bf16)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = sum(scorer.score(r["target"], o)["rougeL"].fmeasure for r, o in zip(rows, outs)) / len(rows)
    n = len(rows)
    return {
        "checkpoint": str(ckpt), "n": n, "val_loss": round(loss, 4), "rougeL": round(rl, 4),
        "fk_out": round(sum(fk_grade(o) for o in outs) / n, 2),
        "fk_src": round(sum(fk_grade(r["source"]) for r in rows) / n, 2),
        "fk_target": round(sum(fk_grade(r["target"]) for r in rows) / n, 2),
        "number_faithful_pct": round(100 * sum(numbers_faithful(r["source"], o) for r, o in zip(rows, outs)) / n, 1),
        "outputs": outs,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score checkpoints on the fixed val set.")
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--fixed", type=Path, default=config.PDQ_VAL_FIXED_PATH)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    rows = load_fixed(args.fixed)
    for ckpt in args.checkpoints:
        res = evaluate_checkpoint(ckpt, rows, args.cpu)
        (ckpt / "fixed_val_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "outputs"}), flush=True)


if __name__ == "__main__":
    main()
