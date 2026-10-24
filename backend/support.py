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


def boilerplate_sentences(source: str, output: str) -> list[str]:
    """Output sentences to hard-drop at inference time (generic claims the model learned from PDQ, not from the trial).

    Four rules: (1) contains a generic-claim phrase from config.BOILERPLATE_OPENERS that the source lacks;
    (2) 'X is a type of X' tautology; (3) introduces a risk-lexicon word absent from the source;
    (4) unbacked definition (is_unbacked_definition).
    """
    return [x for x in split_sentences(output)
            if has_boilerplate(source, x) or is_tautology(x) or adds_risk_claim(source, x) or is_unbacked_definition(source, x)]
