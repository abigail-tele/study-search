"""Compare baselines and fine-tuned checkpoints with Precision / Recall / F1.

Three steps (each is a subcommand):
  build-trials   data/trial_eval_set.jsonl: 40 real trials (one per cancer type) -> brief summary + 5 criteria each
  generate       run every system on the chosen split(s); write results/raw_predictions_<system>.jsonl   (GPU, .venv-train)
  score          read the predictions; write results/baselines_results.csv + error_analysis_<system>.txt (CPU, .venv)

Systems (all see exactly the same examples):
  copy-source            output = source unchanged (lower bound; trivially faithful)
  zero-shot-simplify     google/flan-t5-base, prompt "simplify: {source}"
  zero-shot-instruct     google/flan-t5-base, prompt "Rewrite this for a patient in plain language: {source}"
  few-shot-k3            google/flan-t5-base + up to 3 TRAIN-split example pairs (seed 42; shots are dropped until the prompt fits 512 tokens)
  <checkpoint folder>    every fine-tuned model under checkpoints/ (folders ending in "-best" are copies and skipped)

Splits: "val" = data/pdq_val_fixed.jsonl (use this to PICK a winner), "test" = PDQ test split (run ONCE at the end,
needs --allow-test), "trials" = the trial-domain set (no reference text; this is the setting the app lives in).

Metrics
  ROUGE-1/2/L P/R/F1      vs the NCI patient text (rouge-score)
  token P/R/F1            SQuAD-style bag of lower-cased word tokens, micro-averaged over the split
  key-term P/R/F1         key terms = numbers/percentages, units that follow a number (years, mg, ...), stage numerals
                          (roman and arabic are the same: "stage III" == "stage 3"), gene/marker names (HER2, EGFR, PD-L1,
                          lexicon + "capital letters with a digit" pattern), drug names (lexicon + suffixes like -mab, -nib,
                          -platin). Precision = output key terms that are also in the SOURCE (low = invented facts, the
                          hard-rule-3 risk); recall = source key terms kept in the output. Micro-averaged.
  support precision       share of output sentences that backend.support.sentence_support accepts (meaning close to a source
                          sentence AND words found in source/glossary); plus % outputs fully supported
  boilerplate rate        % outputs with a generic PDQ opener ("There are different types of ...") absent from the source
  Flesch-Kincaid grade    backend.readability (our own, no textstat)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from backend import config
from backend.readability import fk_grade

SPLITS = ("val", "test", "trials")
INSTRUCT_PREFIX = "Rewrite this for a patient in plain language: "
FEWSHOT_HEADER = "Rewrite professional cancer text in plain language.\n\n"

# --- key terms -----------------------------------------------------------------
_ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
_STAGE_RE = re.compile(r"\bstages?\b((?:\s*(?:,|and\b|or\b|to\b|through\b|-|–)?\s*(?:0|iv|i{1,3}|[1-4])[abc]?\b)+)", re.I)
_NUM_RE = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")
_UNIT_RE = re.compile(r"\d\s*-?\s*(years?|yrs?|months?|weeks?|days?|mg|mcg|ml|gy|cm|mm|kg)\b|\d\s*(%)|(\bpercent\b)", re.I)
_UNIT_NORM = {"year": "years", "years": "years", "yr": "years", "yrs": "years", "month": "months", "months": "months",
              "week": "weeks", "weeks": "weeks", "day": "days", "days": "days", "percent": "%"}
_GENE_LEXICON = {"HER2", "HER-2", "EGFR", "BRCA1", "BRCA2", "BRCA", "KRAS", "NRAS", "BRAF", "ALK", "ROS1", "MET", "NTRK",
                 "RET", "FLT3", "TP53", "PIK3CA", "PSA", "HPV", "MSI", "ER", "PR", "PD-L1", "PD-1", "CTLA-4", "BCR-ABL",
                 "JAK2", "IDH1", "IDH2", "CD20", "CD19", "CD33", "AR", "FGFR", "MYC", "NPM1", "ECOG"}
_GENE_PATTERN_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z]?\d+)?\b")
_DRUG_SUFFIXES = ("mab", "nib", "ciclib", "platin", "taxel", "rubicin", "zomib", "parib", "tecan", "trexate",
                  "citabine", "fosfamide", "mustine", "lutamide", "mycin")
_DRUG_LEXICON = {"tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant", "methotrexate", "cyclophosphamide",
                 "doxorubicin", "paclitaxel", "docetaxel", "cisplatin", "carboplatin", "oxaliplatin", "gemcitabine",
                 "capecitabine", "fluorouracil", "temozolomide", "bevacizumab", "imatinib", "rituximab", "pembrolizumab",
                 "nivolumab", "atezolizumab", "durvalumab", "olaparib", "enzalutamide", "abiraterone", "leuprolide",
                 "goserelin", "etoposide", "vincristine", "bleomycin", "irinotecan", "lenalidomide", "bortezomib",
                 "thalidomide", "dexamethasone", "prednisone", "cytarabine", "azacitidine", "decitabine", "mitomycin",
                 "cetuximab", "trastuzumab", "hydroxyurea", "tretinoin"}


def _stage_terms(text: str) -> set[str]:
    out: set[str] = set()
    for m in _STAGE_RE.finditer(text):
        for tok in re.findall(r"\b(0|iv|i{1,3}|[1-4])[abc]?\b", m.group(1), re.I):
            out.add(f"stage:{_ROMAN[tok.lower()]}")
    return out


def _gene_terms(text: str) -> set[str]:
    out = set()
    for tok in _GENE_PATTERN_RE.findall(text):
        if tok in _GENE_LEXICON or (re.search(r"\d", tok) and len(tok) >= 3 and re.search(r"[A-Z]{2}", tok)):
            out.add("gene:" + tok.replace("-", "").lower())
    for g in re.findall(r"\bPD-L1\b|\bPD-1\b|\bHER-2\b|\bCTLA-4\b|\bBCR-ABL\b", text):
        out.add("gene:" + g.replace("-", "").lower())
    return out


def _drug_terms(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]{6,}", text.lower()):
        if w in _DRUG_LEXICON or (len(w) >= 7 and w.endswith(_DRUG_SUFFIXES)):
            out.add("drug:" + w)
    return out


def key_terms(text: str) -> set[str]:
    """Set of normalised key terms (see module docstring) found in text."""
    plain = _STAGE_RE.sub(" ", text)  # stage numerals are handled separately, not as plain numbers
    terms = {"n:" + n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(plain)}
    for m in _UNIT_RE.finditer(text):
        u = (m.group(1) or m.group(2) or m.group(3)).lower()
        terms.add("u:" + _UNIT_NORM.get(u, u))
    return terms | _stage_terms(text) | _gene_terms(text) | _drug_terms(text)


# --- metric functions (pure, tested without models) ---------------------------------
def prf(tp: float, n_out: float, n_ref: float) -> tuple[float, float, float]:
    """Precision, recall, F1. An empty side counts as 1.0 (nothing wrong / nothing missed)."""
    p = tp / n_out if n_out else 1.0
    r = tp / n_ref if n_ref else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_prf(refs: list[str], outs: list[str]) -> tuple[float, float, float]:
    """SQuAD-style bag-of-tokens overlap, micro-averaged (counts summed over all examples first)."""
    tp = n_out = n_ref = 0
    for ref, out in zip(refs, outs):
        a, b = Counter(tokens(out)), Counter(tokens(ref))
        tp += sum((a & b).values())
        n_out += sum(a.values())
        n_ref += sum(b.values())
    return prf(tp, n_out, n_ref)


def keyterm_prf(sources: list[str], outputs: list[str]) -> dict:
    """Key-term P/R/F1 of outputs vs their SOURCES, plus % outputs with no invented key term."""
    tp = n_out = n_src = clean = 0
    hallucinated: list[list[str]] = []
    for src, out in zip(sources, outputs):
        s, o = key_terms(src), key_terms(out)
        tp += len(o & s)
        n_out += len(o)
        n_src += len(s)
        bad = sorted(o - s)
        hallucinated.append(bad)
        clean += not bad
    p, r, f = prf(tp, n_out, n_src)
    return {"p": p, "r": r, "f": f, "zero_halluc_pct": 100 * clean / max(len(outputs), 1), "hallucinated": hallucinated}


def rouge_prf(refs: list[str], outs: list[str]) -> dict[str, tuple[float, float, float]]:
    """Mean ROUGE-1/2/L precision, recall, F1 over examples."""
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    acc = {k: [0.0, 0.0, 0.0] for k in ("rouge1", "rouge2", "rougeL")}
    for ref, out in zip(refs, outs):
        for k, v in sc.score(ref, out).items():
            for i, x in enumerate((v.precision, v.recall, v.fmeasure)):
                acc[k][i] += x
    n = max(len(refs), 1)
    return {k: tuple(x / n for x in v) for k, v in acc.items()}


# --- data ------------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_split(name: str, min_similarity: float = 0.70) -> list[dict]:
    """Rows for a split: {id, source, target (None for trials), kind}. Test needs the caller to opt in."""
    if name == "val":
        rows = _jsonl(config.PDQ_VAL_FIXED_PATH)
    elif name == "test":
        rows = [r for r in _jsonl(config.PDQ_PAIRS_PATH) if r["split"] == "test" and r["similarity"] >= min_similarity]
    elif name == "trials":
        rows = _jsonl(config.TRIAL_EVAL_PATH)
        return [{"id": r["id"], "source": r["source"], "target": None, "kind": r["kind"]} for r in rows]
    else:
        raise ValueError(name)
    return [{"id": r["id"], "source": r["source"], "target": r["target"], "kind": "pair"} for r in rows]


def build_trial_set(db_path: Path = config.DB_PATH, out: Path = config.TRIAL_EVAL_PATH,
                    n: int | None = None, seed: int = config.RANDOM_SEED) -> list[dict]:
    """Pick real trials, one per cancer type (so >= TRIAL_EVAL_MIN_TYPES types), seed 42, read-only DB.

    Each trial gives its brief summary (cut near TRIAL_EVAL_MAX_SUMMARY_WORDS at a sentence end) and its first
    TRIAL_EVAL_CRITERIA inclusion criteria of 5+ words. No reference text is needed: these sources are scored
    for faithfulness / support only.
    """
    import sqlite3

    from backend.summarize import clean_text, split_eligibility, split_sentences

    n = n or config.TRIAL_EVAL_COUNT
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    rng = random.Random(seed)
    types = [r[0] for r in conn.execute("SELECT DISTINCT cancer_type_id FROM trial_cancer_types ORDER BY 1")]
    rng.shuffle(types)
    rows: list[dict] = []
    seen: set[str] = set()
    pools = {}
    for ctype in types:
        ids = [r[0] for r in conn.execute("SELECT nct_id FROM trial_cancer_types WHERE cancer_type_id=? ORDER BY 1", (ctype,))]
        rng.shuffle(ids)
        pools[ctype] = ids[:80]

    def take(ctype: str) -> bool:
        """Add the next usable trial of this type; False if the type has none left."""
        while pools[ctype]:
            nct = pools[ctype].pop()
            if nct in seen:
                continue
            row = conn.execute("SELECT brief_summary, eligibility_text FROM trials WHERE nct_id=?", (nct,)).fetchone()
            summary = clean_text(row[0])
            crit = [c for c in split_eligibility(row[1])[0] if len(c.split()) >= 5][: config.TRIAL_EVAL_CRITERIA]
            if len(summary.split()) < 40 or len(crit) < config.TRIAL_EVAL_CRITERIA:
                continue
            seen.add(nct)
            cut, words = [], 0
            for sent in split_sentences(summary):
                cut.append(sent)
                words += len(sent.split())
                if words >= config.TRIAL_EVAL_MAX_SUMMARY_WORDS:
                    break
            rows.append({"id": f"{nct}-summary", "nct_id": nct, "cancer_type": ctype, "kind": "summary", "source": " ".join(cut)})
            rows.extend({"id": f"{nct}-c{i}", "nct_id": nct, "cancer_type": ctype, "kind": "criterion", "source": c}
                        for i, c in enumerate(crit))
            return True
        return False

    while len(seen) < n and any(pools.values()):   # round-robin: one trial per type per pass
        for ctype in types:
            if len(seen) < n:
                take(ctype)
    if len({r["cancer_type"] for r in rows}) < config.TRIAL_EVAL_MIN_TYPES:
        raise RuntimeError("trial eval set spans too few cancer types")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# --- generation --------------------------------------------------------------------------
def pick_shots(train_rows: list[dict], k: int, seed: int = config.RANDOM_SEED) -> list[dict]:
    """k short TRAIN pairs, chosen with a fixed seed so every run uses the same shots."""
    pool = [r for r in train_rows if len(r["source"].split()) <= config.FEWSHOT_MAX_SOURCE_WORDS
            and len(r["target"].split()) <= config.FEWSHOT_MAX_SOURCE_WORDS]
    return random.Random(seed).sample(pool, k)


def fewshot_prompt(source: str, shots: list[dict], count_tokens, limit: int = config.MAX_INPUT_TOKENS) -> tuple[str, int]:
    """Prompt with as many shots as fit in `limit` tokens (drops from the end). Returns (prompt, shots used)."""
    for n in range(len(shots), -1, -1):
        body = "".join(f"Professional: {s['source']}\nPlain: {s['target']}\n\n" for s in shots[:n])
        prompt = f"{FEWSHOT_HEADER}{body}Professional: {source}\nPlain:"
        if count_tokens(prompt) <= limit or n == 0:
            return prompt, n
    raise AssertionError("unreachable")


def discover_checkpoints(root: Path = config.CHECKPOINT_DIR) -> list[Path]:
    """Direct sub-folders of checkpoints/ holding a model, minus '*-best' copies."""
    return sorted(p for p in root.iterdir() if (p / "model.safetensors").exists() and not p.name.endswith("-best"))


def _run_model(model_dir: str, prompts_by_key: dict[str, list[str]], cpu: bool) -> dict[str, list[str]]:
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from pipeline.compare_checkpoints import generate
    from pipeline.train import pick_device

    device, bf16 = pick_device(cpu)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    return {k: generate(model, tok, p, device, bf16) for k, p in prompts_by_key.items()}


def is_causal(ckpt: Path) -> bool:
    """True for LoRA/chat checkpoints (train_lora.py writes model_type=causal into train_metadata.json)."""
    meta = Path(ckpt) / "train_metadata.json"
    return meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("model_type") == "causal"


def _run_causal(model_dir: str, sources_by_key: dict[str, list[str]], cpu: bool) -> dict[str, list[str]]:
    """Generate with a decoder-only chat model: chat prompt in, new tokens out (same beams / n-gram rule as T5)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline.train import pick_device
    from pipeline.train_lora import chat_prompt

    device, bf16 = pick_device(cpu)
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16 if bf16 else torch.float32).to(device).eval()
    result = {}
    for key, sources in sources_by_key.items():
        prompts = [chat_prompt(tok, s) for s in sources]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [""] * len(prompts)
        for k in range(0, len(order), 8):
            idx = order[k:k + 8]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                ids = model.generate(**enc, num_beams=config.NUM_BEAMS, no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                                     max_new_tokens=config.MAX_OUTPUT_TOKENS, do_sample=False)
            for i, t in zip(idx, tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                outs[i] = t.strip()
        result[key] = outs
    return result


def _store_predictions(system: str, split: str, rows: list[dict], outputs: list[str], shots: list[int] | None = None) -> None:
    path = config.RESULTS_DIR / f"raw_predictions_{system}.jsonl"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keep = [r for r in _jsonl(path) if r["split"] != split] if path.exists() else []
    new = [{"split": split, "system": system, "id": r["id"], "kind": r["kind"], "source": r["source"],
            "target": r["target"], "output": o, **({"shots_used": shots[i]} if shots else {})}
           for i, (r, o) in enumerate(zip(rows, outputs))]
    with path.open("w", encoding="utf-8") as f:
        for r in keep + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_generate(args: argparse.Namespace) -> None:
    splits = args.splits.split(",")
    if "test" in splits and not args.allow_test:
        sys.exit("refusing to touch the test split without --allow-test (run it ONCE, after choosing on val)")
    data = {s: load_split(s) for s in splits}
    for s, rows in data.items():
        print(f"split {s}: {len(rows)} examples", flush=True)
    wanted = None if args.systems == "all" else set(args.systems.split(","))
    use = lambda name: wanted is None or name in wanted  # noqa: E731

    if use("copy-source"):
        for s, rows in data.items():
            _store_predictions("copy-source", s, rows, [r["source"] for r in rows])
    if any(use(n) for n in ("zero-shot-simplify", "zero-shot-instruct", "few-shot-k3")):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        count = lambda t: len(tok(t)["input_ids"])  # noqa: E731
        train = [r for r in _jsonl(config.PDQ_PAIRS_PATH) if r["split"] == "train" and r["similarity"] >= 0.70]
        shots = pick_shots(train, config.FEWSHOT_K)
        keyed: dict[str, list[str]] = {}
        meta: dict[str, list[int]] = {}
        for s, rows in data.items():
            if use("zero-shot-simplify"):
                keyed[f"zero-shot-simplify|{s}"] = [config.TASK_PREFIX + r["source"] for r in rows]
            if use("zero-shot-instruct"):
                keyed[f"zero-shot-instruct|{s}"] = [INSTRUCT_PREFIX + r["source"] for r in rows]
            if use("few-shot-k3"):
                built = [fewshot_prompt(r["source"], shots, count) for r in rows]
                keyed[f"few-shot-k3|{s}"] = [b[0] for b in built]
                meta[f"few-shot-k3|{s}"] = [b[1] for b in built]
        outs = _run_model(config.BASE_MODEL_NAME, keyed, args.cpu)
        for key, o in outs.items():
            system, s = key.split("|")
            _store_predictions(system, s, data[s], o, meta.get(key))
            if key in meta:
                print(f"few-shot {s}: mean shots used {sum(meta[key]) / len(meta[key]):.2f} of {config.FEWSHOT_K}")
    for ckpt in discover_checkpoints():
        if not use(ckpt.name):
            continue
        print(f"== {ckpt.name}", flush=True)
        if is_causal(ckpt):
            res = _run_causal(str(ckpt), {s: [r["source"] for r in rows] for s, rows in data.items()}, args.cpu)
        else:
            keyed = {s: [config.TASK_PREFIX + r["source"] for r in rows] for s, rows in data.items()}
            res = _run_model(str(ckpt), keyed, args.cpu)
        for s, o in res.items():
            _store_predictions(ckpt.name, s, data[s], o)


# --- scoring ------------------------------------------------------------------------------
COLUMNS = ["system", "split", "n", "rouge1_p", "rouge1_r", "rouge1_f", "rouge2_p", "rouge2_r", "rouge2_f",
           "rougeL_p", "rougeL_r", "rougeL_f", "token_p", "token_r", "token_f", "keyterm_p", "keyterm_r", "keyterm_f",
           "keyterm_zero_halluc_pct", "support_precision", "fully_supported_pct", "boilerplate_pct", "guard_pass_pct",
           "fk_out", "fk_src", "fk_target", "avg_out_words"]


def score_system(rows: list[dict], encode=None) -> tuple[dict, dict]:
    """All metrics for one system on one split. Returns (metrics row, per-example details for error analysis)."""
    from backend import support

    srcs, outs = [r["source"] for r in rows], [r["output"] for r in rows]
    n = len(rows)
    m: dict = {"n": n}
    has_ref = all(r["target"] for r in rows)
    if has_ref:
        refs = [r["target"] for r in rows]
        for k, (p, r_, f) in rouge_prf(refs, outs).items():
            m.update({f"{k}_p": p, f"{k}_r": r_, f"{k}_f": f})
        m.update(zip(("token_p", "token_r", "token_f"), token_prf(refs, outs)))
        m["fk_target"] = sum(fk_grade(t) for t in refs) / n
    kt = keyterm_prf(srcs, outs)
    m.update(keyterm_p=kt["p"], keyterm_r=kt["r"], keyterm_f=kt["f"], keyterm_zero_halluc_pct=kt["zero_halluc_pct"])
    m["boilerplate_pct"] = 100 * sum(support.has_boilerplate(s, o) for s, o in zip(srcs, outs)) / n
    m.update(fk_out=sum(fk_grade(o) for o in outs) / n, fk_src=sum(fk_grade(s) for s in srcs) / n,
             avg_out_words=sum(len(o.split()) for o in outs) / n)
    unsupported: list[list[str]] = [[] for _ in rows]
    try:
        res = support.sentence_support_many(list(zip(srcs, outs)), encode)
        total = sum(r[2] for r in res)
        bad = sum(len(r[1]) for r in res)
        m["support_precision"] = (total - bad) / total if total else 1.0
        m["fully_supported_pct"] = 100 * sum(r[0] for r in res) / n
        unsupported = [r[1] for r in res]
        # what actually reaches patients: no sentence flagged by support / boilerplate / risk / definition / direction
        m["guard_pass_pct"] = 100 * sum(
            not (set(bad) | set(support.boilerplate_sentences(s, o)) | set(support.direction_flipped_sentences(s, o)))
            for (s, o), (_, bad, _n) in zip(zip(srcs, outs), res)) / n
    except ImportError:
        print("sentence-transformers missing: support metrics skipped (run score in .venv)", file=sys.stderr)
    return m, {"hallucinated": kt["hallucinated"], "unsupported": unsupported}


def _replace_sections(path: Path, split: str, text: str) -> None:
    """Keep other splits' sections in the error-analysis file; replace this split's."""
    parts: dict[str, str] = {}
    if path.exists():
        for chunk in re.split(r"(?m)^(?==== split: )", path.read_text(encoding="utf-8")):
            m = re.match(r"=== split: (\w+) ===", chunk)
            if m:
                parts[m.group(1)] = chunk
    parts[split] = f"=== split: {split} ===\n{text}\n"
    path.write_text("".join(parts[k] for k in sorted(parts)), encoding="utf-8")


def write_error_analysis(system: str, split: str, rows: list[dict], det: dict, k: int = 15) -> None:
    """Worst k outputs by key-term precision: source / target / output / invented terms / unsupported sentences."""
    def prec(i: int) -> float:
        o = key_terms(rows[i]["output"])
        return 1.0 if not o else 1 - len(det["hallucinated"][i]) / len(o)
    worst = sorted(range(len(rows)), key=lambda i: (prec(i), -len(det["hallucinated"][i])))[:k]
    blocks = []
    for i in worst:
        r = rows[i]
        blocks.append(f"[{r['id']}] key-term precision {prec(i):.2f}\nSOURCE: {r['source']}\nTARGET: {r['target']}\n"
                      f"OUTPUT: {r['output']}\nINVENTED KEY TERMS: {det['hallucinated'][i]}\n"
                      f"UNSUPPORTED SENTENCES: {det['unsupported'][i]}\n")
    _replace_sections(config.RESULTS_DIR / f"error_analysis_{system}.txt", split, "\n".join(blocks))


def _update_csv(new_rows: list[dict]) -> None:
    path = config.RESULTS_DIR / "baselines_results.csv"
    old = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    replaced = {(r["system"], r["split"]) for r in new_rows}
    merged = [r for r in old if (r["system"], r["split"]) not in replaced] + new_rows
    merged.sort(key=lambda r: (r["split"], r["system"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        w.writeheader()
        for r in merged:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def cmd_score(args: argparse.Namespace) -> None:
    encode = None
    try:
        from backend.embedding import get_encoder
        encode = get_encoder()
    except ImportError:
        pass
    splits = args.splits.split(",")
    out_rows = []
    for path in sorted(config.RESULTS_DIR.glob("raw_predictions_*.jsonl")):
        allrows = _jsonl(path)
        for split in splits:
            rows = [r for r in allrows if r["split"] == split]
            if not rows:
                continue
            system = rows[0]["system"]
            m, det = score_system(rows, encode)
            out_rows.append({"system": system, "split": split, **m})
            write_error_analysis(system, split, rows, det)
            print(f"scored {system} / {split}", flush=True)
    _update_csv(out_rows)
    print_table(out_rows)


def print_table(rows: list[dict]) -> None:
    cols = ["rougeL_f", "token_f", "keyterm_p", "keyterm_r", "keyterm_f", "support_precision", "boilerplate_pct", "fk_out"]
    print("system".ljust(34), "split".ljust(7), *[c[:10].rjust(10) for c in cols])
    for r in rows:
        vals = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else "-" for c in cols]
        print(r["system"].ljust(34), r["split"].ljust(7), *[v.rjust(10) for v in vals])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-trials")
    g = sub.add_parser("generate")
    g.add_argument("--splits", default="val,trials")
    g.add_argument("--systems", default="all", help="comma list or 'all'")
    g.add_argument("--allow-test", action="store_true")
    g.add_argument("--cpu", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--splits", default="val,trials")
    args = ap.parse_args(argv)
    if args.cmd == "build-trials":
        rows = build_trial_set()
        print(f"{len(rows)} rows, {len({r['nct_id'] for r in rows})} trials, {len({r['cancer_type'] for r in rows})} cancer types")
    elif args.cmd == "generate":
        cmd_generate(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
