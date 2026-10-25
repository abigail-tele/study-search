"""Semantic support check: does each output sentence actually say something the source says?

Why: the number/term faithfulness check cannot see a fluent sentence that is
simply NOT in the source ("Women age 18 or older are more likely to get breast
cancer" from a bare age criterion). A sentence counts as supported only if BOTH
  1. it is close in meaning (MiniLM cosine >= SUPPORT_SIM_THRESHOLD) to some
     source sentence or window of up to SUPPORT_WINDOW consecutive sentences, AND
  2. at least SUPPORT_WORD_FRACTION of its content words appear in the source or
     in the NCI glossary definitions (words the glossary legitimately adds).
This is an extra, automatic fallback trigger; the human-written
summarize.faithfulness_check is separate and untouched.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Callable

from backend import config

_STOP = frozenset("""a an and are as at be been being but by can could did do does for from had has have he her his how i if in into
is it its may might more most no not of on one or other our out over she so some such than that the their them then there these
they this those to too under up was we were what when which who will with would you your also about after all any both each few
many much only own same should very""".split())
_SENT_RE = re.compile(r"(?<=[a-z0-9\)\]][.!?])\s+(?=[A-Z(\[])")
_WORD_RE = re.compile(r"[a-z][a-z-]{2,}")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]


def _stem(w: str) -> str:
    """Very light stemming so 'tumors' matches 'tumor' and 'treated' matches 'treat'."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    """Lower-case stemmed non-stopwords of 3+ letters."""
    return {_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


@lru_cache(maxsize=1)
def glossary_words() -> frozenset[str]:
    """Words used in glossary terms and their plain definitions (allowed even if absent from the source)."""
    try:
        terms = json.loads(config.GLOSSARY_PATH.read_text(encoding="utf-8")).get("terms", {})
    except (OSError, ValueError):
        return frozenset()
    out: set[str] = set()
    for term, info in terms.items():
        out |= content_words(term) | content_words(info.get("plain", ""))
    return frozenset(out)


def _windows(sents: list[str], size: int) -> list[str]:
    return [" ".join(sents[i : i + n]) for i in range(len(sents)) for n in range(1, size + 1) if i + n <= len(sents)]


def sentence_support(source: str, output: str, encode: Callable | None = None) -> tuple[bool, list[str]]:
    """Return (all output sentences supported, list of unsupported output sentences)."""
    ok, bad, _ = sentence_support_many([(source, output)], encode)[0]
    return ok, bad


def sentence_support_many(
    pairs: list[tuple[str, str]], encode: Callable | None = None
) -> list[tuple[bool, list[str], int]]:
    """Batch version: one embedding call for everything. Returns (ok, unsupported, n_sentences) per pair."""
    import numpy as np

    if encode is None:
        from backend.embedding import get_encoder
        encode = get_encoder()
    prepared = []
    texts: list[str] = []
    for source, output in pairs:
        out_s = split_sentences(output)
        win = _windows(split_sentences(source) or [source], config.SUPPORT_WINDOW)
        prepared.append((out_s, win))
        texts += out_s + win
    uniq = list(dict.fromkeys(t for t in texts if t))
    vec = dict(zip(uniq, np.asarray(encode(uniq)))) if uniq else {}
    gloss = glossary_words()
    results = []
    for (source, _), (out_s, win) in zip(pairs, prepared):
        src_words = content_words(source) | gloss
        bad = []
        for s in out_s:
            cw = content_words(s)
            sim = max((float(vec[s] @ vec[w]) for w in win), default=0.0)
            frac = (sum(1 for w in cw if w in src_words) / len(cw)) if len(cw) >= config.SUPPORT_MIN_CONTENT_WORDS else 1.0
            if sim < config.SUPPORT_SIM_THRESHOLD or frac < config.SUPPORT_WORD_FRACTION:
                bad.append(s)
        results.append((not bad, bad, len(out_s)))
    return results


def has_boilerplate(source: str, output: str) -> bool:
    """True if the output contains a generic PDQ opener that the source does not contain."""
    o, s = output.lower(), source.lower()
    return any(p in o and p not in s for p in config.BOILERPLATE_OPENERS)


_IS_A_RE = re.compile(r"^(.{2,80}?)\s+(?:is|are)\s+(?:an?\s+)?(?:type|kind|form|group)s?\s+of\s+(.+)$", re.IGNORECASE)


def _ordered_content_words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


def is_tautology(sentence: str) -> bool:
    """'Radiation therapy is a type of radiation therapy ...': the subject's first words reappear after 'is a type of'."""
    m = _IS_A_RE.match(sentence.strip())
    if not m:
        return False
    subj = _ordered_content_words(m.group(1))[: config.TAUTOLOGY_CONTENT_WORDS]
    return bool(subj) and all(w in set(_ordered_content_words(m.group(2))) for w in subj)


def adds_risk_claim(source: str, sentence: str) -> bool:
    """True if the sentence uses a risk-lexicon word (likely, risk, cure, ...) that the source never uses."""
    src = content_words(source)
    return any(_stem(w) in content_words(sentence) and _stem(w) not in src for w in config.RISK_WORDS)


_DEF_RE = re.compile(config.DEFINITION_RE, re.IGNORECASE)
_DEF_SPLIT_RE = re.compile(r"\b(?:is|are)\s+(?:a|an|the)\s+", re.IGNORECASE)


def is_unbacked_definition(source: str, sentence: str) -> bool:
    """General rule: drop a definitional sentence whose predicate the source does not back.

    Why: trial summaries and eligibility rules state requirements; they never
    need us to add definitions the source didn't give. A sentence is definitional
    if it looks like "X is a <noun> that/which/who ..." or says "may include /
    can include / includes" (and the source lacks that verb phrase). It is dropped
    when more than DEFINITION_ABSENT_FRACTION of the predicate's content words are
    missing from the source. Source-backed definitions therefore survive.
    """
    low, src_low = sentence.lower(), source.lower()
    if _DEF_RE.match(sentence):
        pred = _DEF_SPLIT_RE.split(sentence, maxsplit=1)[-1]
    else:
        verb = next((v for v in config.DEFINITION_VERBS if re.search(rf"\b{v}\b", low) and not re.search(rf"\b{v}\b", src_low)), None)
        if verb is None:
            return False
        pred = sentence
    words = content_words(pred)
    if not words:
        return False
    src = content_words(source)
    return sum(1 for w in words if w not in src) / len(words) > config.DEFINITION_ABSENT_FRACTION


_CLAIM_RES = [re.compile(p, re.IGNORECASE) for p in config.CLAIM_PATTERNS]
_ACRONYM_DEF_RE = re.compile(config.ACRONYM_DEFINITION_RE)   # case-sensitive on purpose
_COPULA_RE = re.compile(r"\b(?:is|are)\s+(?:a|an|the)\s+([\w\-]+(?:\s+[\w\-]+)*)", re.IGNORECASE)


def adds_invented_claim(source: str, sentence: str) -> bool:
    """True if the sentence defines or claims something the source never says.

    Why: the model learned PDQ-style textbook sentences ("X is a rare type of
    cancer", "Y is used to treat Z"). Trial text states requirements, so any
    such statement we add is invented, and for a drug it would be a medical
    claim (rule 7). Two parts: (1) a documented phrase list (config.CLAIM_PATTERNS),
    dropped unless the source has the same phrase; (2) general rule: any
    "X is/are a/an/the <predicate>" is dropped unless the predicate's first
    COPULA_BACKED_WORDS words appear verbatim in the source.
    """
    src = " ".join(source.lower().split())
    for rx in _CLAIM_RES:
        m = rx.search(sentence)
        if m and not rx.search(src):
            return True
    am = _ACRONYM_DEF_RE.match(sentence)
    if am and not re.search(re.escape(am.group(0).strip().lower()), src):
        return True
    m = _COPULA_RE.search(sentence)
    if m:
        pred = " ".join(m.group(1).lower().split()[: config.COPULA_BACKED_WORDS])
        return pred not in src
    return False


def boilerplate_sentences(source: str, output: str) -> list[str]:
    """Output sentences to hard-drop at inference time (generic claims the model learned from PDQ, not from the trial).

    Four rules: (1) contains a generic-claim phrase from config.BOILERPLATE_OPENERS that the source lacks;
    (2) 'X is a type of X' tautology; (3) introduces a risk-lexicon word absent from the source;
    (4) unbacked definition (is_unbacked_definition).
    """
    return [x for x in split_sentences(output)
            if has_boilerplate(source, x) or is_tautology(x) or adds_risk_claim(source, x) or is_unbacked_definition(source, x)
                    or adds_invented_claim(source, x)]


# ---------------------------------------------------------------------------
# Direction / negation check (automatic, NOT human-owned)
# ---------------------------------------------------------------------------
# Why: the number check only asks "is 18 in the source?". It cannot see that
# "at least 18" became "under 18", or that "no prior chemotherapy" became
# "prior chemotherapy". Both turn a rule into its opposite. Two simple rules:
#   1. Number direction: for every number that appears in both texts, if the
#      source bounds it one way (lower: at least / or older / >= ...) and the
#      output bounds it the other way (upper: under / up to / <= ...), flag it.
#      If the source gives no bound for that number nothing is flagged (the
#      number check still guards the digits themselves).
#   2. Negation: split both texts into short clauses. For each output clause,
#      find the source clause sharing the most content words (at least
#      DIRECTION_MIN_SHARED_WORDS). If one is negated (no/not/without/never/
#      cannot/excluded/ineligible...) and the other is not, flag the clause.
# When unsure (no overlapping clause, different wording) it stays silent: it is
# a cheap tripwire for flips, not a proof of equivalence.
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_LOWER, _UPPER = "lower", "upper"


def _alt(phrases) -> str:
    return "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))


_PREFIX_DIR = {p: _LOWER for p in config.BOUND_LOWER_PREFIX} | {p: _UPPER for p in config.BOUND_UPPER_PREFIX}
_SUFFIX_DIR = {p: _LOWER for p in config.BOUND_LOWER_SUFFIX} | {p: _UPPER for p in config.BOUND_UPPER_SUFFIX}
_PREFIX_RE = re.compile(rf"(?<!\w)({_alt(_PREFIX_DIR)})\s*(?:of\s+)?{_NUM}", re.IGNORECASE)
_SUFFIX_RE = re.compile(rf"{_NUM}\s*(?:[A-Za-z%/]+\s+){{0,3}}({_alt(_SUFFIX_DIR)})(?!\w)", re.IGNORECASE)
_PLUS_RE = re.compile(rf"{_NUM}\+")


def _norm_num(s: str) -> str:
    return "%g" % float(s.replace(",", ""))


def number_bounds(text: str) -> list[tuple[str, str, str]]:
    """(number, 'lower'|'upper', matched phrase) for every bounded number in the text."""
    out = []
    for m in _PREFIX_RE.finditer(text):
        out.append((_norm_num(m.group(2)), _PREFIX_DIR[m.group(1).lower()], m.group(0)))
    for m in _SUFFIX_RE.finditer(text):
        out.append((_norm_num(m.group(1)), _SUFFIX_DIR[m.group(2).lower()], m.group(0)))
    out += [(_norm_num(m.group(1)), _LOWER, m.group(0)) for m in _PLUS_RE.finditer(text)]
    return out


def _in_range(num: str, text: str) -> bool:
    """True if `num` is an end of a range in the text ('18-75', '18 to 75', 'between 18 and 75'): both ends kept means no comparator is needed."""
    for m in re.finditer(rf"{_NUM}\s*(?:-|–|—|to|and)\s*{_NUM}", text, re.IGNORECASE):
        if num in (_norm_num(m.group(1)), _norm_num(m.group(2))):
            return True
    return False


def _dropped_comparators(source: str, output: str) -> list[str]:
    """Numbers the source bounds (>= 18, 3 months or less) that the output repeats with NO bound of that direction.

    Why: the T5 tokenizer cannot write >= <= < >, so '>= 18 years' came out as
    '18 years' and the requirement silently lost its direction. A bare number is
    only accepted when it is part of a range whose ends are both kept.
    """
    out_bounds: dict[str, set[str]] = {}
    for n, d, _ in number_bounds(output):
        out_bounds.setdefault(n, set()).add(d)
    out_nums = {_norm_num(m.group(1)) for m in re.finditer(rf"(?<![\w.]){_NUM}(?![\w])", output)}
    problems = []
    for n, d, phrase in number_bounds(source):
        if n in out_nums and d not in out_bounds.get(n, set()) and not _in_range(n, output):
            problems.append(f"comparator dropped for {n}: source '{phrase}', output has the number without a {d} bound")
    return problems


def _flipped_numbers(source: str, output: str) -> list[str]:
    src: dict[str, dict[str, str]] = {}
    for n, d, phrase in number_bounds(source):
        src.setdefault(n, {})[d] = phrase
    problems = _dropped_comparators(source, output)
    for n, d, phrase in number_bounds(output):
        if n in src and d not in src[n]:
            problems.append(f"direction flipped for {n}: output '{phrase}' vs source '{next(iter(src[n].values()))}'")
    return problems


_NEG_RE = re.compile(rf"(?<!\w)(?:{_alt(config.NEGATION_CUES)})(?!\w)|n't", re.IGNORECASE)
_NEUTRAL_RE = re.compile(rf"(?<!\w)(?:{_alt(config.NEGATION_NEUTRAL_PHRASES)})(?!\w)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"[.;:\n]|\s+(?:and|but|or)\s+(?=(?:must|should|are|is|have|has|do|does|be|can|cannot|may|will)\b)", re.IGNORECASE)
# words that carry no topic: ignoring them stops "must"/"patients" alone from matching two clauses
_GENERIC = frozenset("must should need patient participant subject person people trial study includ include eligible eligibility criteria".split())


def _clauses(text: str) -> list[tuple[bool, frozenset[str]]]:
    """(is_negated, topic words) per clause. Neutral phrases ('no more than') are blanked first."""
    out = []
    for c in _CLAUSE_RE.split(text):
        c = _NEUTRAL_RE.sub(" ", c)
        neg = bool(_NEG_RE.search(c))
        words = frozenset(w for w in content_words(_NEG_RE.sub(" ", c)) if w not in _GENERIC)
        if words:
            out.append((neg, words))
    return out


def _flipped_negations(source: str, output: str) -> list[str]:
    src = _clauses(source)
    problems = []
    for o_neg, o_words in _clauses(output):
        need = min(config.DIRECTION_MIN_SHARED_WORDS, len(o_words))
        scored = [(len(o_words & s_words), s_neg) for s_neg, s_words in src
                  if len(o_words & s_words) / len(s_words) >= config.DIRECTION_MIN_SOURCE_FRACTION]
        best = max((n for n, _ in scored), default=0)
        if best < need or best / len(o_words) < config.DIRECTION_MIN_SHARED_FRACTION:
            continue                      # no source clause says "the same thing": stay silent
        if all(s_neg != o_neg for n, s_neg in scored if n == best):
            problems.append(f"negation flipped on {sorted(o_words)}: output {'negates' if o_neg else 'asserts'}, source {'asserts' if o_neg else 'negates'}")
    return problems


def direction_check(source: str, output: str) -> tuple[bool, list[str]]:
    """(ok, problems): did the output reverse a comparison or a negation that the source states?"""
    problems = _flipped_numbers(source, output) + _flipped_negations(source, output)
    return not problems, problems


def direction_flipped_sentences(source: str, output: str) -> list[str]:
    """Output sentences that direction_check rejects (checked one by one against the whole source)."""
    return [s for s in split_sentences(output) if not direction_check(source, s)[0]]
