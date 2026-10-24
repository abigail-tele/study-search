"""Reading-level estimate (Flesch-Kincaid grade) with no extra dependency.

Why our own: textstat's license is still undecided, and the formula is short.
The syllable counter is a heuristic (vowel groups), good enough to compare
"before" and "after" text, not to publish as an exact grade.
"""
from __future__ import annotations

import re

from backend import config


def count_syllables(word: str) -> int:
    """Rough syllable count: groups of vowels, minus a silent trailing 'e' (min 1)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def fk_grade(text: str) -> float:
    """Flesch-Kincaid grade = 0.39*words/sentence + 11.8*syllables/word - 15.59 (0.0 for no words)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words:
        return 0.0
    sents = max(len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]), 1)
    syl = sum(count_syllables(w) for w in words)
    return 0.39 * len(words) / sents + 11.8 * syl / len(words) - 15.59


def reading_grade(text: str) -> float:
    """fk_grade rounded for display and storage, never below 0."""
    return round(max(fk_grade(text), 0.0), config.READING_LEVEL_DECIMALS)
