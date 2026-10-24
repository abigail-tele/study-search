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
