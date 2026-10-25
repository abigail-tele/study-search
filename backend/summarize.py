"""Plain-language summaries of trials (CLAUDE.md rules 2, 3 and 7).

Flow: split the eligibility text into include / exclude criteria, break long
text into sentence windows that fit the model, let the fine-tuned flan-t5
rewrite each piece, then run faithfulness_check on EVERY piece. A piece that
fails the check is replaced by the original wording with jargon explained from
the NCI glossary, so the model can never put a new number or drug in front of
a patient. Only brief_summary and eligibility are rewritten (rule 2); every
other field is read straight from stored API data elsewhere.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

from backend import config, support
from backend.readability import reading_grade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Eligibility splitting
# ---------------------------------------------------------------------------
_HEADER_WORDS = r"(?:key|main|major|general|specific|study|patient|participant|subject|additional|other|basic)"
_HEADER_RE = re.compile(
    rf"^[\s#*_>\-]*(?:{_HEADER_WORDS}\s+){{0,2}}(inclusion|exclusion|eligibility)\s+criteria\b[^\n:]{{0,40}}?:?[ \t]*(.*)$",
    re.IGNORECASE)
# CT.gov often flattens everything onto one line: "...therapy). Exclusion Criteria Prior surgery."
_INLINE_HEADER_RE = re.compile(r"(?<=[.;:\)])[ \t]+(?=(?:inclusion|exclusion)\s+criteria\b)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(\s*)(?:[*\-\u2022\u2013\u25cf]|\d{1,2}[.)]|\(?[a-z][.)]|\([ivx]+\)|[ivx]+[.)])\s+(.*)$")
_MD_ESCAPE_RE = re.compile(r"\\([<>_*\[\]()#+\-.!~=&|`])")
_SENT_SPLIT_RE = re.compile(r"(?<=[a-z0-9\)]{2}[.;])\s+(?=[A-Z0-9(])")


def clean_text(text: str | None) -> str:
    """Undo CT.gov markdown escapes (`\\>` -> `>`) and tidy whitespace.

    Why: the API stores markdown; the model and the faithfulness check should
    see the symbols a human would read (">=", "<50%").
    """
    if not text:
        return ""
    t = _MD_ESCAPE_RE.sub(r"\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r"[ \t]+", " ", t).strip()


def _strip_item(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(";,").strip()


def _segment_items(lines: list[str]) -> list[str]:
    """Turn one include/exclude block into criteria, handling bullets, nesting and plain paragraphs."""
    items: list[str] = []
    parent_of: list[int] = []           # parallel to items: index of the parent bullet or -1
    stack: list[tuple[int, int]] = []   # (indent, index in items) of open parent bullets
    parents_with_children: set[int] = set()
    saw_bullet = False
    prev_blank = True
    for raw in lines:
        if not raw.strip():
            prev_blank = True
            continue
        m = _BULLET_RE.match(raw.expandtabs(4))
        if m:
            saw_bullet = True
            indent, body = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent_idx = stack[-1][1] if stack else None
            if parent_idx is not None:
                parents_with_children.add(parent_idx)
            items.append(body)
            parent_of.append(parent_idx if parent_idx is not None else -1)
            stack.append((indent, len(items) - 1))
        elif items and saw_bullet and not prev_blank:
            items[-1] += " " + raw.strip()          # wrapped continuation line
        else:
            items.append(raw.strip())               # plain paragraph / lead-in
            parent_of.append(-1)
            stack = []
        prev_blank = False
    return _flatten(items, parent_of, parents_with_children, saw_bullet)


def _flatten(items: list[str], parent_of: list[int], parents: set[int], saw_bullet: bool) -> list[str]:
    """Drop 'header:' parents and prefix their children with the parent text for context."""
    out: list[str] = []
    for i, text in enumerate(items):
        if i in parents and text.rstrip().endswith(":"):
            continue
        p = parent_of[i]
        if p >= 0 and items[p].rstrip().endswith(":"):
            text = f"{items[p].rstrip().rstrip(':').strip()}: {text}"
        out.append(text)
    final: list[str] = []
    for t in out:
        t = _strip_item(t)
        if not t:
            continue
        if not saw_bullet and len(t.split()) > 25:   # a pasted paragraph: one criterion per sentence
            final.extend(_strip_item(s) for s in _SENT_SPLIT_RE.split(t) if _strip_item(s))
        else:
            final.append(t)
    # a lone non-bullet line ending in ':' is a lead-in ("Patients must meet all of:"), not a criterion
    return [t for t in final if not (t.endswith(":") and len(t.split()) < 25)]


def split_eligibility(text: str | None) -> tuple[list[str], list[str]]:
    """Split CT.gov eligibility text into (include, exclude) criteria lists.

    Why: inclusion and exclusion must be explained separately - mixing them
    would tell a patient the opposite of the truth. Handles missing headers
    (everything counts as inclusion), odd casing, headers glued to text,
    bullets of several styles, numbered lists and nested bullets.
    """
    t = clean_text(text)
    if not t:
        return [], []
    t = _INLINE_HEADER_RE.sub("\n", t)
    blocks: dict[str, list[str]] = {"include": [], "exclude": []}
    current = "include"
    for line in t.split("\n"):
        m = _HEADER_RE.match(line)
        if m and len(line) - len(m.group(2)) < 90:
            kind = m.group(1).lower()
            current = "exclude" if kind == "exclusion" else "include"
            if m.group(2).strip():
                blocks[current].append(m.group(2))
        else:
            blocks[current].append(line)
    inc = _segment_items(blocks["include"])
    exc = _segment_items(blocks["exclude"])
    return inc, exc


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def _word_count_tokens(text: str) -> int:
    """Token estimate used only when no tokenizer is available (unit tests)."""
    return math.ceil(len(text.split()) * config.SUMMARY_FALLBACK_TOKENS_PER_WORD)


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps abbreviations like 'e.g.' and decimals like '2.5' intact."""
    parts = re.split(r"(?<=[a-z0-9\)\]][.!?])\s+(?=[A-Z(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(text: str, count: Callable[[str], int] = _word_count_tokens,
                    max_tokens: int = config.MAX_INPUT_TOKENS,
                    prefix: str = config.TASK_PREFIX) -> list[str]:
    """Group whole sentences into windows so prefix + chunk fits max_tokens.

    Why sentence windows: cutting mid-sentence confuses the model, and one
    oversized input would be silently truncated (losing criteria). A single
    sentence longer than the limit is cut by words as a last resort.
    """
    budget = max_tokens - count(prefix)
    chunks: list[str] = []
    cur: list[str] = []
    for sent in split_sentences(text):
        if count(sent) > budget:                      # pathological sentence: word-level cut
            if cur:
                chunks.append(" ".join(cur)); cur = []
            words, piece = sent.split(), []
            for w in words:
                if piece and count(" ".join(piece + [w])) > budget:
                    chunks.append(" ".join(piece)); piece = []
                piece.append(w)
            if piece:
                chunks.append(" ".join(piece))
            continue
        if cur and count(" ".join(cur + [sent])) > budget:
            chunks.append(" ".join(cur)); cur = []
        cur.append(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------------------
# 3. Model loading + generation
# ---------------------------------------------------------------------------
_cache: dict[str, object] = {}


def resolve_model_dir(path: Path | str | None = None) -> Path:
    """The requested checkpoint, or the plain 'flan-t5-pdq' one if the default 'best' is not trained yet."""
    p = Path(path) if path else config.SUMMARY_MODEL_DIR
    if p.exists():
        return p
    if p == config.SUMMARY_MODEL_DIR and config.SUMMARY_FALLBACK_MODEL_DIR.exists():
        return config.SUMMARY_FALLBACK_MODEL_DIR
    raise FileNotFoundError(f"No summarizer checkpoint found at {p}")


def load_model(path: Path | str = config.SUMMARY_MODEL_DIR):
    """Load (once) and return (tokenizer, model, device). CPU unless CUDA is available.

    Why lazy + cached: loading takes seconds and the web server should start
    instantly; tests and search never pay for it.
    """
    d = resolve_model_dir(path)
    key = str(d)
    if key not in _cache:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # local_files_only: rule 4, never contact the Hugging Face Hub at request time.
        tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(d), local_files_only=True).to(device).eval()
        _cache[key] = (tok, model, device)
    return _cache[key]


def model_version(path: Path | str = config.SUMMARY_MODEL_DIR) -> str:
    """'<checkpoint folder>@<git commit of the training code>' so a summary can be traced to its model."""
    d = resolve_model_dir(path)
    commit = ""
    meta = d / "train_metadata.json"
    if meta.exists():
        try:
            commit = json.loads(meta.read_text(encoding="utf-8")).get("git_commit", "")[:8]
        except (ValueError, OSError):
            pass
    return f"{d.name}@{commit}" if commit else d.name


def _token_counter() -> Callable[[str], int]:
    """Exact token counts if a tokenizer is loaded/loadable, else the word-based estimate."""
    try:
        tok = load_model()[0]
        return lambda s: len(tok(s, add_special_tokens=True)["input_ids"])
    except Exception:  # no checkpoint on this machine
        return _word_count_tokens


def generation_kwargs() -> dict:
    """The decoding settings, in one place so the live path and the GPU batch pipeline cannot differ."""
    return {"num_beams": config.NUM_BEAMS, "no_repeat_ngram_size": config.NO_REPEAT_NGRAM_SIZE,
            "max_new_tokens": config.MAX_OUTPUT_TOKENS}


def _generate_batch(prompts: list[str]) -> list[str]:
    """Run the model on prompts (already prefixed) with the beam settings from config."""
    import torch
    tok, model, device = load_model()
    outs: list[str] = []
    for i in range(0, len(prompts), config.SUMMARY_BATCH_SIZE):
        enc = tok(prompts[i:i + config.SUMMARY_BATCH_SIZE], return_tensors="pt", padding=True,
                  truncation=True, max_length=config.MAX_INPUT_TOKENS).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, **generation_kwargs())
        outs.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return [o.strip() for o in outs]


def simplify_chunks(chunks: list[str]) -> list[str]:
    """Model rewrite of each chunk (adds the task prefix, batches the calls)."""
    if not chunks:
        return []
    return _generate_batch([config.TASK_PREFIX + c for c in chunks])


def simplify(text: str) -> str:
    """Plain-language rewrite of free text: chunk -> generate -> join. NOT faithfulness-checked."""
    return " ".join(simplify_chunks(chunk_sentences(text, _token_counter())))


# ---------------------------------------------------------------------------
# 4. Faithfulness check  (AI-written at the team's request; team must review)
# ---------------------------------------------------------------------------
# How it works, in plain words: pull every "protected term" out of the SOURCE
# and out of the MODEL OUTPUT (numbers, number+unit pairs, genes/markers,
# mutation names, stage terms, drug names), then require every protected term in
# the output to also be in the source. Ordinary words are never compared, so the
# model is free to simplify language; it just cannot introduce a new fact.
# Pieces are first "masked" out of the text (stage and marker text is removed
# before digits are read) so the "3" of "stage 3B" or the "2" of "HER2" is not
# mistaken for a free-standing number.
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = (r"%|percent|per\s?cent|years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?|hrs?|minutes?|mins?|"
         r"(?:mg|mcg|[uµμ]g|ng|pg|g|kg|ml|dl|l|iu|units?|mmol|gy|cgy|cm|mm)"
         r"(?:/(?:m2|m²|kg|mg|ml|l|dl|min|day|d|h|hr|mm3|[uµμ]l))?|/(?:mm3|[uµμ]l|l|dl)")
_UNIT_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9.])((?:{_NUM})(?:\s*(?:-|–|—|to|and|or|,)\s*(?:{_NUM}))*)[\s-]*({_UNIT})(?![A-Za-z])", re.IGNORECASE)
_NUM_RE = re.compile(rf"(?<![A-Za-z0-9.])({_NUM})")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
_STAGE_TOK = r"(?:iv|iii|ii|i|v|[0-4])(?:[abce][12]?)?(?![a-z0-9])"
_STAGE_RE = re.compile(rf"\bstages?\s+({_STAGE_TOK}(?:\s*(?:,|/|-|–|to|or|and)\s*{_STAGE_TOK})*)", re.IGNORECASE)
_MUTATION_RE = re.compile(r"\b[A-Z]\d{2,4}[A-Z*](?![A-Za-z0-9])", re.IGNORECASE)
# genes / markers that are distinctive in any case (contain a digit or are long), and
# short or word-like ones that only count when written in capitals (ER, PR, ALK, MET, ...)
_MARKER_ANY_RE = re.compile(
    r"\b(?:her-?2(?:/neu)?|erbb2|egfr|ros1|braf|kras|nras|hras|brca[12]?|pik3ca|pd-?l1|pd-?1|ctla-?4|msi-?h|dmmr|pmmr|"
    r"ntrk[123]?|fgfr[1-4]?|idh[12]|flt3|pten|tp53|palb2|cd\d+[a-z]?|bcr-?abl|jak2|npm1|cebpa|bcl-?2|esr1|mgmt|"
    r"cdk4/6|ca-?125|ca-?19-?9|hpv|ebv|hiv|hbv|hcv|psa|afp)(?![a-z0-9])", re.IGNORECASE)
_MARKER_CAPS_RE = re.compile(r"\b(?:ALK|ER|PR|AR|RET|MET|KIT|ATM|MYC|TMB|MSI|CEA|TERT)\b")
_MARKER_GENERIC_RE = re.compile(r"\b[A-Z]{2,}[A-Z]*\d[A-Z0-9]*\b")     # BCL6, MLH1, FOXP3 ... any unseen gene
_BRCA_PAIR_RE = re.compile(r"brca([12])\s*(?:/|and|or|,)\s*(?:brca)?([12])", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_DRUG_STEMS = ("mab", "nib", "ciclib", "parib", "lib", "platin", "taxel", "rubicin", "mycin", "stat", "vir",
               "zomib", "limus", "tinib", "ximab", "zumab", "umab")
_BRAND_RE = re.compile(r"\(\s*([A-Z][A-Za-z]{3,})\s*\)")
_UNIT_CANON = (("%", "%"), ("percent", "%"), ("per cent", "%"), ("percentage", "%"))


def _canon_value(num: str) -> str:
    """'1,500' -> '1500', '9.50' -> '9.5', '007' -> '7' so equal numbers compare equal."""
    n = num.replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n.lstrip("0") or "0"


def _canon_unit(unit: str) -> str:
    """Spelling variants of one unit become one key ('Years', 'yrs' -> 'year'; 'µg' -> 'mcg'; 'm²' -> 'm2')."""
    u = re.sub(r"\s+", "", unit.lower()).replace("m²", "m2").replace("µ", "u").replace("μ", "u")
    if u in ("%", "percent", "percent"):
        return "%"
    u = re.sub(r"^ug", "mcg", u).replace("/ul", "/ul")
    for stem, canon in (("year", "year"), ("yr", "year"), ("month", "month"), ("mo", "month"), ("week", "week"),
                        ("wk", "week"), ("day", "day"), ("hour", "hour"), ("hr", "hour"), ("minute", "minute")):
        if u == stem or u == stem + "s":
            return canon
    if u in ("min", "mins"):
        return "minute"
    return u.rstrip("s") if u.startswith("unit") else u


def _stage_tokens(span: str) -> set[str]:
    """'IIIB or IIIC' -> {'3b','3c'}; 'I-III' -> {'1','2','3'} (a range includes the stages in between)."""
    toks = re.findall(_STAGE_TOK, span, flags=re.IGNORECASE)
    out = set()
    for t in toks:
        t = t.lower()
        m = re.match(r"(iv|iii|ii|i|v|[0-4])(.*)", t)
        out.add(_ROMAN.get(m.group(1), m.group(1)) + m.group(2))
    plain = sorted(int(t) for t in out if t.isdigit())
    if plain and re.search(r"-|–|\bto\b", span) and len(plain) >= 2:
        out |= {str(i) for i in range(plain[0], plain[-1] + 1)}
    return out


@lru_cache(maxsize=1)
def _drug_lexicon() -> frozenset[str]:
    """Drug names the suffix rules miss (data/drug_lexicon.txt, one lower-case name per line, '#' = comment)."""
    try:
        lines = Path(config.DRUG_LEXICON_PATH).read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(x.strip().lower() for x in lines if x.strip() and not x.startswith("#"))


def _is_drug_word(w: str) -> bool:
    w = w.lower()
    return w.endswith(_DRUG_STEMS) or w in _drug_lexicon()


def protected_terms(text: str) -> dict[str, set[str]]:
    """Every protected term in `text`, grouped: number, unit (value|unit pairs), marker, stage, drug.

    Why this shape: each group is compared separately so the report can say
    exactly what the model invented ("number 21", "drug pembrolizumab").
    """
    terms: dict[str, set[str]] = {k: set() for k in ("number", "unit", "marker", "stage", "drug")}
    work = text
    for m in _STAGE_RE.finditer(work):
        terms["stage"] |= {f"stage {t}" for t in _stage_tokens(m.group(1))}
    work = _STAGE_RE.sub(" ", work)
    work = _BRCA_PAIR_RE.sub(lambda m: f"BRCA{m.group(1)} BRCA{m.group(2)}", work)
    for rx in (_MARKER_ANY_RE, _MARKER_GENERIC_RE, _MARKER_CAPS_RE, _MUTATION_RE):
        for m in rx.finditer(work):
            terms["marker"].add(re.sub(r"[-\s]", "", m.group(0).lower()))
        work = rx.sub(" ", work)
    for m in _UNIT_PAIR_RE.finditer(work):
        unit = _canon_unit(m.group(2))
        for n in re.findall(_NUM, m.group(1)):
            terms["unit"].add(f"{_canon_value(n)} {unit}")
    terms["number"] = {_canon_value(n) for n in _NUM_RE.findall(work)}
    for w in _WORD_RE.findall(text):
        if _is_drug_word(w):
            terms["drug"].add(w.lower())
    for m in _BRAND_RE.finditer(text):                         # "pembrolizumab (Keytruda)"
        before = re.findall(r"[A-Za-z]+", text[:m.start()])[-1:]
        if before and _is_drug_word(before[0]):
            terms["drug"].add(m.group(1).lower())
    return terms


# HUMAN-WRITTEN -> AI-written (Claude Code, 2026-10-24) at the team's request - team must review
def faithfulness_check(source: str, output: str) -> tuple[bool, list[str]]:
    """Return (ok, missing_terms): did the model invent anything? (CLAUDE.md rule 3)

    Every one of these found in `output` must also appear in `source`:
      * numbers (integers, decimals like 2.5, percentages, ranges like 18-75)
      * ages and durations with their units ("18 years or older", "6 months")
      * dose / lab units (mg/m2, mg/kg, mL/min, x10^9/L)
      * drug names (pembrolizumab, trastuzumab, ...)
      * genes / markers (HER2, EGFR exon 19, BRCA1, PD-L1, KRAS G12C)
      * stage terms ("stage IIIB", "stage 3"): roman and arabic forms of the
        same stage are equivalent; the letter suffix (A/B/C) must match
    Comparison is case-insensitive. Words the model adds that are not in these
    categories are fine (that is the point of simplifying). Return
    (True, []) when nothing is missing, else (False, [the missing terms]).
    An empty output is not faithful (nothing to show): return (False, [...]).

    How to recognise a drug name (a spec, not an implementation). Treat a word
    as a drug if ANY of these holds:
      * it is lower-case (or capitalised) and ends in a common drug stem:
        -mab, -zumab, -ximab, -umab, -nib, -tinib, -ciclib, -parib, -lib,
        -platin, -taxel, -rubicin, -mycin, -stat, -vir;
      * it is a capitalised brand name in parentheses after a generic name,
        e.g. "pembrolizumab (Keytruda)";
      * it is on a small lexicon file the team adds (e.g. data/drug_lexicon.txt)
        for drugs the stems miss ("methotrexate", "tamoxifen", "letrozole").
        The endings -ide and -ine are too common in ordinary words ("outside",
        "decline"), so use them ONLY together with the lexicon, never alone.

    Direction and negation must also survive (the number check alone misses
    these). The output must NOT reverse a comparison or a negation the source
    states, even when every number and drug is present:
      * "at least 18" -> "under 18"; "18 years or older" -> "younger than 18";
      * ">= 2 lines" -> "<= 2 lines";
      * "no prior chemotherapy" -> "prior chemotherapy";
      * "must not be pregnant" -> "must be pregnant";
      * "excluded" <-> "included".
    Safe paraphrases that MUST still pass: "18 years or older" <-> "at least 18
    years old"; "no prior chemo" <-> "have not had chemotherapy before".
    (support.direction_check already screens this automatically as a second
    layer; this function rejects it too, so rule 3 does not depend on
    the optional screen.)

    Implementation: protected_terms() lists the protected terms of both texts;
    any output term absent from the source is reported as "<kind> <term>".
    Units are compared as (value, unit) pairs, so "18 months" does not match a
    source that only says "18 years"; ranges give the unit to both ends.
    A plain number in the output only needs the same value somewhere in the source.
    """
    if not output.strip():
        return False, ["empty output"]
    src, out = protected_terms(source), protected_terms(output)
    missing = [f"{kind} {t}" for kind in ("number", "unit", "marker", "stage", "drug")
               for t in sorted(out[kind] - src[kind])]
    ok_dir, problems = support.direction_check(source, output)
    if not ok_dir:
        missing += [f"direction {p}" for p in problems]
    return not missing, missing


_warned_stub = False


def _check(source: str, output: str, nct: str = "") -> tuple[bool, list[str]]:
    """faithfulness_check, but a missing implementation OR any crash counts as FAILED.

    Why: one bug in the human-written check (say, a regex error on odd text)
    must cost that piece its model rewrite, not abort the whole trial. Only the
    NCT id and the exception type are logged, never the text.

    LOUD NOTE: until the team writes faithfulness_check, EVERY model output is
    rejected and patients see only the glossary-swapped original. That is the
    safe default for a medical-adjacent app - never show unchecked model text.
    """
    global _warned_stub
    try:
        return faithfulness_check(source, output)
    except NotImplementedError:
        if not _warned_stub:
            logger.error("faithfulness_check is not implemented: ALL model output is rejected "
                         "and original wording (glossary-swapped) is shown instead.")
            _warned_stub = True
        return False, ["faithfulness_check not implemented"]
    except Exception as exc:
        logger.error("faithfulness_check crashed nct=%s error=%s; piece falls back to original wording",
                     nct, type(exc).__name__)
        return False, [f"faithfulness_check error: {type(exc).__name__}"]


# ---------------------------------------------------------------------------
# 5. Glossary fallback
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_glossary(path: Path | str = config.GLOSSARY_PATH) -> dict[str, str]:
    """term -> plain meaning, from data/glossary.json (NCI Dictionary of Cancer Terms paraphrases)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t: v["plain"] for t, v in data["terms"].items()}


@lru_cache(maxsize=1)
def _glossary_regex() -> tuple[re.Pattern, dict[str, str]]:
    g = load_glossary()
    terms = sorted(g, key=len, reverse=True)   # longest first: "radiation therapy" before "therapy"
    pat = r"(?<![\w-])(" + "|".join(re.escape(t) + (r"(?:s)?" if not t.endswith("s") else "") for t in terms) + r")(?![\w-])"
    return re.compile(pat, re.IGNORECASE), {t.lower(): p for t, p in g.items()}


def glossary_swap(text: str) -> str:
    """Explain jargon in the ORIGINAL wording: 'term (plain meaning)', first occurrence of each term only.

    Why: this is what a patient sees when the model's rewrite fails the
    faithfulness check. It only inserts words after a term; it never edits or
    removes existing text, so numbers and units stay exactly as written.
    """
    rx, g = _glossary_regex()
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        word = m.group(1)
        key = word.lower()
        base = key if key in g else (key[:-1] if key.endswith("s") and key[:-1] in g else None)
        if base is None or base in seen:
            return word
        seen.add(base)
        return f"{word} ({g[base]})"

    return rx.sub(repl, text)


# ---------------------------------------------------------------------------
# 6. Whole-trial summary  (split -> generate -> judge -> assemble)
# ---------------------------------------------------------------------------
# Why four small steps: pipeline/batch_summarize.py runs the SAME split and the
# SAME judge as the live path, but generates on a GPU ahead of time and judges
# later. Sharing these functions means the two paths cannot drift apart.
def _encode(texts: list[str]):
    """MiniLM sentence vectors (the shared encoder from backend.embedding, loaded once). Tests replace this."""
    from backend.embedding import get_encoder
    return get_encoder()(texts)


def _screen_detail(src: str, out: str, nct: str, bad: list[str]) -> tuple[str | None, dict[str, int]]:
    """Apply the sentence-level guards to one output that already passed faithfulness_check.

    `bad` = sentences the semantic-support check rejected. Boilerplate sentences
    (generic PDQ openers, tautologies, unbacked risk claims and definitions) and
    sentences that flip a comparison or negation (support.direction_check) are
    added. Rule: if nothing is bad, keep the whole output; if some are bad, drop
    only those sentences provided at least SUMMARY_MIN_KEPT_FRACTION of the
    sentences remain (so the piece still covers the source); otherwise return None
    = fall back for the whole piece. Dropping can only remove words, so nothing
    new can appear. Also returns how many sentences each guard caught (for stats).
    """
    sents = support.split_sentences(out)
    boiler = set(support.boilerplate_sentences(src, out))
    flipped = set(support.direction_flipped_sentences(src, out))
    drop = set(bad) | boiler | flipped
    hits = {"support": len(set(bad)), "boilerplate": len(boiler), "direction": len(flipped)}
    if not drop:
        return out, hits
    kept = [x for x in sents if x not in drop]
    logger.warning("support check nct=%s dropped_sentences=%s kept=%d/%d", nct, sorted(drop), len(kept), len(sents))
    if kept and len(kept) / len(sents) >= config.SUMMARY_MIN_KEPT_FRACTION:
        return " ".join(kept), hits
    return None, hits


def _screen(src: str, out: str, nct: str, bad: list[str]) -> str | None:
    """_screen_detail without the per-guard counts."""
    return _screen_detail(src, out, nct, bad)[0]


def always_ok(source: str, output: str, nct: str = "") -> tuple[bool, list[str]]:
    """A check that passes everything: used ONLY to measure what the guards alone would let through (never stored)."""
    return True, []


def judge_chunks(items: list[tuple[str, str, str]], nct: str, check: Callable = _check,
                 stats: Counter | None = None) -> tuple[dict[str, str], bool]:
    """Decide what a patient sees for each (piece_id, source chunk, model output).

    Check 1: faithfulness_check (numbers/drugs/units, rule 3). Check 2:
    support.sentence_support (every sentence must be semantically and lexically
    backed by the source) plus the boilerplate / direction guards, see
    _screen_detail. A chunk that fails, or whose output is empty, becomes the
    glossary-swapped original; details are logged (nct, which check, offending
    text). Returns ({piece_id: shown text}, any_fallback). `stats`, when given,
    counts pieces, shown model text, and fallbacks by reason.
    """
    st = stats if stats is not None else Counter()
    results: dict[str, str] = {}
    fallback = False
    passed: list[tuple[str, str, str]] = []
    for pid, src, out in items:
        st["pieces"] += 1
        if not out.strip():
            ok, missing, reason = False, ["empty output"], "fell_back_empty"
        else:
            ok, missing = check(src, out, nct)
            reason = "fell_back_faithfulness"
        if ok:
            passed.append((pid, src, out))
        else:
            fallback = True
            st[reason] += 1
            logger.warning("faithfulness fallback nct=%s missing=%s", nct, missing)
            results[pid] = glossary_swap(src)
    if passed:
        verdicts = support.sentence_support_many([(s, o) for _, s, o in passed], _encode)
        for (pid, src, out), (_, bad, _n) in zip(passed, verdicts):
            shown, hits = _screen_detail(src, out, nct, bad)
            for g, n in hits.items():
                st[f"guard_{g}_sentences"] += n
            if shown is None:
                fallback = True
                st["fell_back_guards"] += 1
                logger.warning("support fallback nct=%s unsupported=%s", nct, bad)
                results[pid] = glossary_swap(src)
            else:
                st["shown_model_text"] += 1
                st["shown_trimmed"] += int(shown != out)
                results[pid] = shown
    return results, fallback


def _rewrite_pieces(pieces: list[str], nct: str) -> tuple[list[str], bool]:
    """Rewrite each piece (as sentence-window chunks) and judge it; kept as the one-call form used by tests."""
    count = _token_counter()
    per_piece = [chunk_sentences(p, count) for p in pieces]
    flat = [c for chunks in per_piece for c in chunks]
    raw = simplify_chunks(flat)
    shown, fallback = judge_chunks([(str(i), s, o) for i, (s, o) in enumerate(zip(flat, raw))], nct)
    out_pieces, k = [], 0
    for chunks in per_piece:
        out_pieces.append(" ".join(shown[str(k + j)] for j in range(len(chunks))))
        k += len(chunks)
    return out_pieces, fallback


def _bounded(criteria: list[str]) -> tuple[list[str], list[str]]:
    """Split criteria into (to rewrite, shown as original) by the per-side cap and minimum length."""
    cap = config.SUMMARY_MAX_CRITERIA_PER_SIDE
    return criteria[:cap], criteria[cap:]


def trial_layout(trial: dict) -> dict:
    """Split one trial into the texts we work with (no model, no tokenizer needed).

    Returns summary_src plus, per side ('include'/'exclude'): `todo` (criteria
    within the cap), `extra` (overflow, shown as original) and `long_idx`
    (indexes in `todo` long enough to be worth a model call).
    """
    include, exclude = split_eligibility(trial.get("eligibility_text") or "")
    sides = {}
    for side, crit in (("include", include), ("exclude", exclude)):
        todo, extra = _bounded(crit)
        long_idx = [i for i, c in enumerate(todo) if len(c.split()) >= config.SUMMARY_MIN_CRITERION_WORDS]
        sides[side] = {"todo": todo, "extra": extra, "long_idx": long_idx}
    return {"summary_src": clean_text(trial.get("brief_summary")), "sides": sides}


_SIDE_TAG = {"include": "I", "exclude": "E"}


def plan_pieces(trial: dict, count: Callable[[str], int] = _word_count_tokens) -> list[dict]:
    """Every model input for one trial as {piece_id, kind, source_text}, in a stable order.

    piece_id 'S:c' = chunk c of the brief summary; 'I:i:c' / 'E:i:c' = chunk c of
    include / exclude criterion i. Why ids: the batch pipeline caches model
    output per piece, so judging can be redone later without regenerating.
    """
    lay = trial_layout(trial)
    pieces = [{"piece_id": f"S:{c}", "kind": "summary", "source_text": t}
              for c, t in enumerate(chunk_sentences(lay["summary_src"], count))] if lay["summary_src"] else []
    for side, info in lay["sides"].items():
        for i in info["long_idx"]:
            pieces += [{"piece_id": f"{_SIDE_TAG[side]}:{i}:{c}", "kind": side, "source_text": t}
                       for c, t in enumerate(chunk_sentences(info["todo"][i], count))]
    return pieces


def _join_group(shown: dict[str, str], prefix: str) -> str:
    """Re-join the shown chunks of one piece group ('S' or 'I:3') in chunk order."""
    keys = sorted((k for k in shown if k.rsplit(":", 1)[0] == prefix), key=lambda k: int(k.rsplit(":", 1)[1]))
    return " ".join(shown[k] for k in keys)


def required_groups(trial: dict) -> list[str]:
    """Piece-group prefixes ('S', 'I:0', 'E:2') a complete set of outputs must cover for this trial."""
    lay = trial_layout(trial)
    groups = ["S"] if lay["summary_src"] else []
    for side, info in lay["sides"].items():
        groups += [f"{_SIDE_TAG[side]}:{i}" for i in info["long_idx"]]
    return groups


def assemble_summary(trial: dict, shown: dict[str, str], fallback: bool, version: str | None = None) -> dict:
    """Build the PLAN section 2 object from judged chunk texts (`shown`, keyed by piece_id).

    Short fragments and overflow criteria never go to the model: they are
    glossary-swapped originals, and overflow sets fallback_used.
    """
    nct = trial["nct_id"]
    lay = trial_layout(trial)
    summary_src, elig_src = lay["summary_src"], trial.get("eligibility_text") or ""
    what = _join_group(shown, "S") if summary_src else ""
    sides: dict[str, list[str]] = {}
    for side, info in lay["sides"].items():
        res = [glossary_swap(c) for c in info["todo"]]
        for i in info["long_idx"]:
            res[i] = _join_group(shown, f"{_SIDE_TAG[side]}:{i}")
        sides[side] = res + [glossary_swap(c) for c in info["extra"]]
        fallback = fallback or bool(info["extra"])
    before = reading_grade(" ".join([summary_src] + [clean_text(elig_src)]))
    after = reading_grade(". ".join([what] + sides["include"] + sides["exclude"]))
    return {
        "nct_id": nct,
        "what_is_tested": what,
        "who_can_join": {"include": sides["include"], "exclude": sides["exclude"]},
        "original": {"summary": trial.get("brief_summary"), "eligibility": trial.get("eligibility_text")},
        "faithful": not fallback,   # True = every shown piece is checked model text; False = some piece is original wording
        "fallback_used": fallback,
        "reading_level": {"before": before, "after": after},
        "model_version": version or model_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def summarize_trial(trial: dict) -> dict:
    """Build the PLAN section 2 summary object for one trial row (dict with nct_id, brief_summary, eligibility_text).

    Rule 3 in code: every model piece goes through faithfulness_check; a miss
    swaps in the glossary-wrapped original for that piece and sets
    fallback_used. `faithful` is True only when NO piece fell back, i.e. every
    shown rewrite is checked model text; `fallback_used` is True when any piece
    (or any overflow criterion) was shown as original wording instead. Either
    way the patient only sees checked or original text.
    """
    pieces = plan_pieces(trial, _token_counter())
    raw = simplify_chunks([p["source_text"] for p in pieces])
    shown, fallback = judge_chunks([(p["piece_id"], p["source_text"], o) for p, o in zip(pieces, raw)],
                                   trial["nct_id"])
    return assemble_summary(trial, shown, fallback)


# ---------------------------------------------------------------------------
# 7. Storage
# ---------------------------------------------------------------------------
def save_summary(conn, summary: dict) -> None:
    """Upsert a summary object into the summaries table (columns defined in db.SCHEMA). Caller commits."""
    conn.execute(
        "INSERT OR REPLACE INTO summaries (nct_id, what_is_tested, include_json, exclude_json, faithful,"
        " fallback_used, reading_before, reading_after, model_version, generated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (summary["nct_id"], summary["what_is_tested"],
         json.dumps(summary["who_can_join"]["include"], ensure_ascii=False),
         json.dumps(summary["who_can_join"]["exclude"], ensure_ascii=False),
         int(summary["faithful"]), int(summary["fallback_used"]),
         summary["reading_level"]["before"], summary["reading_level"]["after"],
         summary["model_version"], summary["generated_at"]))
