"""Offline batch summarizer: generate once on the GPU, check as often as needed on the CPU.

    python -m pipeline.batch_summarize generate [--status RECRUITING ...] [--limit N] [--model-dir D]
    python -m pipeline.batch_summarize check    [--all | --limit N | --nct NCT...] [--model-dir D]

Why two stages: model generation is the slow, expensive part, but the safety
checks (faithfulness_check + the support / boilerplate / direction guards) are
quick and will change once the team writes faithfulness_check. So `generate`
stores the raw model text per piece in the `model_outputs` table, and `check`
turns those stored outputs into the `summaries` rows the live API reads. Both
stages split and judge text with the exact functions backend/summarize.py uses
for live requests, so batch and live results cannot diverge.

`generate` needs only torch + transformers + sentencepiece (it never imports
faiss or sentence-transformers). Rule 4: no network, models load from disk.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend import config, db, summarize

logger = logging.getLogger("batch_summarize")


def open_db(path):
    """Writable connection that waits (up to BATCH_DB_WAIT_SECONDS) for other readers/writers instead of failing with 'database is locked'."""
    conn = db.connect(path)
    conn.execute(f"PRAGMA busy_timeout={int(config.BATCH_DB_WAIT_SECONDS * 1000)}")
    return conn


# ---------------------------------------------------------------------------
# Stage 1: generate (GPU)
# ---------------------------------------------------------------------------
def _status_rank_sql() -> str:
    """SQL CASE that sorts trials by config.BATCH_STATUS_ORDER (unlisted statuses last)."""
    whens = " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(config.BATCH_STATUS_ORDER))
    return f"CASE status {whens} ELSE {len(config.BATCH_STATUS_ORDER)} END"


def pending_trials(conn, version: str, statuses: list[str] | None = None) -> list[dict]:
    """Trials still to generate (no cached piece for this model version), RECRUITING first.

    Why trial-level resume: a slab of trials is written in one transaction, so a
    trial either has all its pieces stored or none.
    """
    where, args = "", []
    if statuses:
        where = f"WHERE status IN ({','.join('?' * len(statuses))})"
        args = list(statuses)
    rows = conn.execute(f"SELECT nct_id, brief_summary, eligibility_text FROM trials {where} "
                        f"ORDER BY {_status_rank_sql()}, nct_id", args).fetchall()
    done = db.trials_with_outputs(conn, version)
    return [dict(r) for r in rows if r["nct_id"] not in done]


def _pick_device(torch):
    """The CUDA GPU with the most memory (the 5090, not the 4070 Ti), else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return torch.device(f"cuda:{best}")


def load_generator(model_dir: Path | str | None):
    """Load tokenizer + model once (local files only, bf16 on the GPU). Returns (tok, model, device, resolved_dir)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    d = summarize.resolve_model_dir(model_dir)
    device = _pick_device(torch)
    dtype = getattr(torch, config.BATCH_DTYPE) if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(d), local_files_only=True, dtype=dtype).to(device).eval()
    return tok, model, device, d


def _generate_ids(prompts: list[str], tok, model, device):
    """One generate() call over a list of prompts; on a GPU out-of-memory error, split the list in half and retry."""
    import torch
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=config.MAX_INPUT_TOKENS).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, **summarize.generation_kwargs())
        return tok.batch_decode(ids, skip_special_tokens=True)
    except torch.OutOfMemoryError:
        if len(prompts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(prompts) // 2
        return _generate_ids(prompts[:half], tok, model, device) + _generate_ids(prompts[half:], tok, model, device)


def run_model(texts: list[str], tok, model, device, batch: int = config.BATCH_GEN_PIECES) -> list[str]:
    """Rewrite each source chunk. Sorted longest-first so padding stays small; results come back in input order."""
    prompts = [config.TASK_PREFIX + t for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outs: list[str] = [""] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        for i, text in zip(idx, _generate_ids([prompts[i] for i in idx], tok, model, device)):
            outs[i] = text.strip()
    return outs


def generate_slab(conn, trials: list[dict], version: str, generate_fn, count) -> tuple[int, int]:
    """Plan, generate and store one slab of trials. Returns (pieces stored, trials stored)."""
    planned = [(t["nct_id"], p) for t in trials for p in summarize.plan_pieces(t, count)]
    outs = generate_fn([p["source_text"] for _, p in planned]) if planned else []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [(nct, p["piece_id"], p["kind"], p["source_text"], o, version, now)
            for (nct, p), o in zip(planned, outs)]
    db.save_model_outputs(conn, rows)
    conn.commit()
    return len(rows), len({r[0] for r in rows})


def _fmt_eta(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def cmd_generate(args) -> int:
    conn = open_db(args.db)
    db.init_schema(conn)
    tok, model, device, mdir = load_generator(args.model_dir)
    version = summarize.model_version(mdir)
    todo = pending_trials(conn, version, args.status)
    if args.limit:
        todo = todo[:args.limit]
    print(f"model={version} device={device} pending_trials={len(todo)}", flush=True)
    count = lambda s: len(tok(s, add_special_tokens=True)["input_ids"])
    gen = lambda texts: run_model(texts, tok, model, device)
    t0, pieces, trials_done = time.time(), 0, 0
    step = config.BATCH_TRIALS_PER_COMMIT
    for i in range(0, len(todo), step):
        n_p, n_t = generate_slab(conn, todo[i:i + step], version, gen, count)
        pieces += n_p
        trials_done = min(i + step, len(todo))
        el = time.time() - t0
        rate = trials_done / el
        print(f"[{trials_done}/{len(todo)}] pieces={pieces} {pieces / el:.1f} pieces/s {rate * 60:.0f} trials/min "
              f"elapsed={_fmt_eta(el)} eta={_fmt_eta((len(todo) - trials_done) / rate)}", flush=True)
    print(f"done: {trials_done} trials, {pieces} pieces in {_fmt_eta(time.time() - t0)}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Stage 2: check (CPU or GPU) -> summaries table
# ---------------------------------------------------------------------------
def faithfulness_implemented() -> bool:
    """True once the team has replaced the HUMAN-WRITTEN stub (it raises NotImplementedError until then)."""
    try:
        summarize.faithfulness_check("x", "x")
    except NotImplementedError:
        return False
    except Exception:  # a crash is still an implementation attempt
        return True
    return True


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _stale(trial: dict, cached: dict[str, tuple[str, str]]) -> bool:
    """True if a cached piece's source text is no longer part of the trial's current text (trial changed after generation)."""
    lay = summarize.trial_layout(trial)
    for pid, (src, _out) in cached.items():
        parts = pid.split(":")
        if parts[0] == "S":
            whole = lay["summary_src"]
        else:
            side = "include" if parts[0] == "I" else "exclude"
            todo = lay["sides"][side]["todo"]
            i = int(parts[1])
            if i not in lay["sides"][side]["long_idx"]:
                return True
            whole = todo[i]
        if _normalize(src) not in _normalize(whole):
            return True
    return False


def trial_items(trial: dict, rows) -> tuple[list[tuple[str, str, str]], bool]:
    """(items for judge_chunks, usable): usable is False if the cache is incomplete or stale for this trial."""
    cached = {r["piece_id"]: (r["source_text"], r["output_text"]) for r in rows}
    have = {pid.rsplit(":", 1)[0] for pid in cached}
    if not set(summarize.required_groups(trial)) <= have or _stale(trial, cached):
        return [], False
    return [(pid, src, out) for pid, (src, out) in cached.items()], True


def _in_guard_sample(nct: str, stride: int) -> bool:
    return zlib.crc32(nct.encode()) % stride == 0


def cmd_check(args) -> int:
    conn = open_db(args.db)
    db.init_schema(conn)
    version = args.model_version or summarize.model_version(args.model_dir)
    ncts = [r[0] for r in conn.execute("SELECT DISTINCT nct_id FROM model_outputs WHERE model_version=? ORDER BY nct_id",
                                       (version,))]
    if args.nct:
        ncts = [n for n in ncts if n in set(args.nct)]
    if args.limit:
        ncts = ncts[:args.limit]
    if not ncts:
        print(f"no cached model outputs for {version}; run `generate` first", file=sys.stderr)
        return 1
    implemented = faithfulness_implemented()
    stride = max(1, len(ncts) // config.BATCH_GUARD_SAMPLE_TRIALS)
    st, guard, trials = Counter(), Counter(), Counter()
    guard_trials = 0
    t0 = time.time()
    for k, nct in enumerate(ncts, 1):
        trow = conn.execute("SELECT nct_id, brief_summary, eligibility_text FROM trials WHERE nct_id=?", (nct,)).fetchone()
        if trow is None:
            trials["missing_trial"] += 1
            continue
        trial = dict(trow)
        items, usable = trial_items(trial, db.get_model_outputs(conn, nct, version))
        if not usable:
            trials["skipped_incomplete_or_stale"] += 1
            continue
        shown, fallback = summarize.judge_chunks(items, nct, stats=st)
        summary = summarize.assemble_summary(trial, shown, fallback, version)
        summarize.save_summary(conn, summary)
        trials["checked"] += 1
        trials["fallback_used"] += int(summary["fallback_used"])
        if not args.no_guard_stat and _in_guard_sample(nct, stride):
            summarize.judge_chunks(items, nct, check=summarize.always_ok, stats=guard)   # measured only, never stored
            guard_trials += 1
        if k % 500 == 0:
            conn.commit()
            print(f"[{k}/{len(ncts)}] {k / (time.time() - t0):.0f} trials/s", flush=True)
    conn.commit()
    stats = build_stats(version, implemented, trials, st, guard, guard_trials)
    config.BATCH_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


def build_stats(version: str, implemented: bool, trials: Counter, st: Counter, guard: Counter, guard_trials: int) -> dict:
    """Pass / fallback counts per check, plus the guard-only pass rate (what faithfulness_check passing everything would allow)."""
    def pieces_block(c: Counter) -> dict:
        total = c["pieces"]
        return {"pieces_total": total, "shown_as_model_text": c["shown_model_text"],
                "shown_model_text_trimmed": c["shown_trimmed"],
                "fell_back": {"empty_output": c["fell_back_empty"], "faithfulness_check": c["fell_back_faithfulness"],
                              "guards_whole_piece": c["fell_back_guards"]},
                "guard_sentences_dropped": {"support": c["guard_support_sentences"],
                                            "boilerplate_risk_definition": c["guard_boilerplate_sentences"],
                                            "direction": c["guard_direction_sentences"]},
                "model_text_rate": round(c["shown_model_text"] / total, 4) if total else None}
    return {"model_version": version, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "faithfulness_check_implemented": implemented, "trials": dict(trials),
            "real_run": pieces_block(st),
            "guard_only": {"note": "what would be shown if faithfulness_check passed everything; sampled, never stored",
                           "sample_trials": guard_trials, **pieces_block(guard)}}


def effective_reading_grade(conn, sample: int = 1000, seed: int = 42) -> dict:
    """Mean Flesch-Kincaid grade of the text patients actually see, on a seeded random sample of trials.

    Why reading_after: assemble_summary computes it over the displayed text only
    (model text where shown, glossary-swapped original where a piece fell back),
    so it is the honest number, unlike the grade of the model's raw output.
    """
    import random
    rows = conn.execute("SELECT nct_id, reading_before, reading_after FROM summaries "
                        "WHERE reading_after IS NOT NULL ORDER BY nct_id").fetchall()
    pick = random.Random(seed).sample(rows, min(sample, len(rows)))
    n = len(pick)
    return {"sample_trials": n, "seed": seed,
            "source_grade_mean": round(sum(r[1] for r in pick if r[1] is not None) / n, 2),
            "effective_grade_mean": round(sum(r[2] for r in pick) / n, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=str(config.DB_PATH))
    common.add_argument("--model-dir", default=None, help="checkpoint folder (default: flan-t5-pdq-best)")
    g = sub.add_parser("generate", parents=[common], help="GPU: store raw model outputs")
    g.add_argument("--status", nargs="+", default=None, help="only these statuses (default: all, RECRUITING first)")
    g.add_argument("--limit", type=int, default=0, help="stop after N pending trials")
    g.set_defaults(fn=cmd_generate)
    c = sub.add_parser("check", parents=[common], help="CPU/GPU: judge stored outputs, write the summaries table")
    c.add_argument("--all", action="store_true", help="every trial with cached outputs (the default)")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--nct", nargs="+", default=None)
    c.add_argument("--model-version", default=None, help="override the cached model_version to read")
    c.add_argument("--no-guard-stat", action="store_true", help="skip the guard-only sample (it embeds sentences)")
    c.add_argument("--stats", default=str(config.BATCH_STATS_PATH))
    c.set_defaults(fn=cmd_check)
    r = sub.add_parser("reading", parents=[common], help="effective reading grade of the stored summaries (seeded sample)")
    r.add_argument("--sample", type=int, default=1000)
    r.set_defaults(fn=lambda a: print(json.dumps(effective_reading_grade(open_db(a.db), a.sample))) or 0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")   # per-piece warnings would flood the console
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
