"""Cancer-type taxonomy: loading the lists and detecting types in a query.

Why: rule 1 (cancer only) hinges on deciding which cancer type(s) a patient
means in their own words, or deciding the query is not about cancer at all.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from backend import config


@lru_cache(maxsize=1)
def load_cancer_types() -> list[dict]:
    """All cancer types from data/cancer_types.json (id, label, lay_terms, ...)."""
    return json.loads(config.CANCER_TYPES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_umbrella_terms() -> dict[str, list[str]]:
    """Broad phrases ("blood cancer") mapped to several type ids."""
    return json.loads(config.UMBRELLA_TERMS_PATH.read_text(encoding="utf-8"))


def cancer_type_chips() -> list[dict]:
    """[{id, label}] for the clickable chips shown on the landing/off-topic page."""
    return [{"id": t["id"], "label": t["label"]} for t in load_cancer_types()]



# ---------------------------------------------------------------- detection helpers
_CANCER_NOUN = re.compile(r"^" + config.CANCER_NOUN_REGEX + r"$")
_ORGAN_GAP_WORDS = 3          # organ word and "cancer" may be this many words apart ("breast and lung cancer")
_FUZZY_NGRAM_MAX = 4          # longest word n-gram of the query compared with the terms
_FUZZY_MIN_TERM_LEN = 4       # terms of 3 letters or fewer are never fuzzy-matched
_FUZZY_TIE_MARGIN = 5         # keep fuzzy types scoring within this of the best one
_EMBED_TIE_MARGIN = 0.08      # same idea for embedding similarity
_ALL_NOT_FOLLOWED_BY = r"(?!\s+(?:of|the|my|our|your|his|her|their|day|night|and|or|is|are|was)\b)"


def _norm(text: str) -> str:
    """Lowercase, hyphens/punctuation to spaces, collapse spaces (so "Non-Small" == "non small")."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _phrase_spans(norm_query: str, phrase: str) -> list[tuple[int, int]]:
    """Spans where `phrase` appears as whole words (optional plural s) in the normalized query."""
    pat = r"(?<![a-z0-9])" + re.escape(phrase) + r"s?(?![a-z0-9])"
    return [m.span() for m in re.finditer(pat, norm_query)]


@lru_cache(maxsize=1)
def _term_index() -> list[tuple[str, str]]:
    """[(normalized term, type id)] over every lay and clinical term."""
    return [(_norm(term), t["id"]) for t in load_cancer_types()
            for term in t["lay_terms"] + t["clinical_terms"] if _norm(term)]


def _is_excluded(raw_lower: str, t: dict) -> bool:
    """True if the type's exclude_patterns hit the query ("cervical spine" is not cervical cancer)."""
    return any(re.search(p, raw_lower) for p in t.get("exclude_patterns") or [])


def _specific_matches(raw: str, norm_q: str) -> dict[str, list[tuple[int, int]]]:
    """Step 1: type id -> spans (in norm_q) found by terms, match_patterns, organ+cancer words, or "ALL".

    A span of (-1, -1) means "matched, but no text span to protect" (it cannot suppress an umbrella).
    """
    raw_lower = raw.lower()
    found: dict[str, list[tuple[int, int]]] = {}
    types = load_cancer_types()
    by_id = {t["id"]: t for t in types}
    for term, tid in _term_index():
        found.setdefault(tid, []).extend(_phrase_spans(norm_q, term))
    for t in types:
        for pat in t.get("match_patterns") or []:
            if re.search(pat, raw_lower):   # loose trial-text patterns must not hide an umbrella phrase
                found.setdefault(t["id"], []).append((-1, -1))
    words = norm_q.split()
    nouns = [i for i, w in enumerate(words) if _CANCER_NOUN.match(w)]
    for t in types:
        organs = {_norm(w) for w in t.get("organ_words") or []}
        hits = [i for i, w in enumerate(words) if w in organs]
        if any(abs(i - j) <= _ORGAN_GAP_WORDS + 1 for i in hits for j in nouns):
            found.setdefault(t["id"], []).append((-1, -1))
    if re.search(r"(?<![A-Za-z])ALL(?![A-Za-z])" + _ALL_NOT_FOLLOWED_BY, raw) and (
            raw == "ALL" or not raw.isupper()):
        found.setdefault("leukemia_all", []).append((-1, -1))
    return {tid: sp for tid, sp in found.items() if sp and not _is_excluded(raw_lower, by_id[tid])}


def _umbrella_matches(norm_q: str, specific_spans: list[tuple[int, int]]) -> list[str]:
    """Step 2: ids of every umbrella phrase in the query, unless a more specific term already covers it.

    Why the span check: "chronic lymphocytic leukemia" contains the umbrella word
    "leukemia", but the patient named one type; returning all four would be noise.
    """
    out: list[str] = []
    for phrase, ids in load_umbrella_terms().items():
        for lo, hi in _phrase_spans(norm_q, _norm(phrase)):
            if not any(slo <= lo and hi <= shi for slo, shi in specific_spans if slo >= 0):
                out.extend(ids)
    return out


def _fuzzy_targets() -> list[tuple[str, list[str]]]:
    """[(normalized text, ids)] for fuzzy matching: terms longer than 3 letters plus umbrella phrases."""
    targets = [(term, [tid]) for term, tid in _term_index() if len(term) >= _FUZZY_MIN_TERM_LEN]
    targets += [(_norm(p), ids) for p, ids in load_umbrella_terms().items()
                if len(_norm(p)) >= _FUZZY_MIN_TERM_LEN]
    return targets


def _fuzzy_match(norm_q: str) -> list[str]:
    """Step 3: typo tolerance. Compare word n-grams of the query with each term (fuzz.ratio)."""
    words = norm_q.split()
    grams = {" ".join(words[i:i + n]) for n in range(1, _FUZZY_NGRAM_MAX + 1)
             for i in range(len(words) - n + 1)}
    grams = {g for g in grams if len(g) >= _FUZZY_MIN_TERM_LEN}
    scored: dict[str, float] = {}
    for text, ids in _fuzzy_targets():
        best = max((fuzz.ratio(g, text) for g in grams), default=0.0)
        if best >= config.TYPE_DETECT_FUZZY_THRESHOLD:
            for tid in ids:
                scored[tid] = max(scored.get(tid, 0.0), best)
    if not scored:
        return []
    top = max(scored.values())
    return [tid for tid, sc in scored.items() if sc >= top - _FUZZY_TIE_MARGIN]


@lru_cache(maxsize=1)
def _type_embeddings() -> tuple[np.ndarray, list[str]]:
    """One MiniLM vector per label/term, plus the type id each belongs to (computed once, local model only)."""
    from backend.embedding import get_encoder
    texts, owners = [], []
    for t in load_cancer_types():
        organ_phrases = [f"{w} cancer" for w in t.get("organ_words") or []]
        for text in [t["label"], *t["lay_terms"], *t["clinical_terms"], *organ_phrases]:
            texts.append(text)
            owners.append(t["id"])
    for phrase, ids in load_umbrella_terms().items():     # "lung cancer" belongs to every lung type
        for tid in ids:
            texts.append(phrase)
            owners.append(tid)
    return get_encoder()(texts), owners


def _embedding_match(norm_q: str) -> list[str]:
    """Step 4: meaning-based fallback ("tummy cancer"). Best type(s) whose closest term is similar enough.

    Uses cancer_type_term_matrix() (PR semantic fallback) with threshold, margin and cap from config.
    The semantic step only runs when the query contains a cancer cue word, so a
    symptom like "stomach ache" is not mistaken for stomach cancer (rule 1).
    """
    cue = getattr(config, "TYPE_DETECT_CUE_REGEX", "")
    if cue and not re.search(cue, norm_q, re.IGNORECASE):
        return []
    from backend.embedding import get_encoder
    try:
        terms, owners, matrix = cancer_type_term_matrix()
        q = get_encoder()([norm_q])[0]
    except RuntimeError:          # model folder missing: behave as "no match" rather than crash
        return []
    if not terms:
        return []
    sims = matrix @ q               # vectors are normalized, so dot product == cosine
    best: dict[str, float] = {}
    for sim, tid in zip(sims, owners):
        best[tid] = max(best.get(tid, -1.0), float(sim))
    if not best:
        return []
    top = max(best.values())
    if top < config.TYPE_DETECT_EMBED_THRESHOLD:
        return []
    margin = getattr(config, "TYPE_DETECT_EMBED_MARGIN", 0.05)
    cap = getattr(config, "TYPE_DETECT_EMBED_MAX_TYPES", 3)
    chosen = [tid for tid, sc in best.items() if sc >= top - margin]
    chosen.sort(key=lambda tid: (-best[tid],))
    chosen = chosen[:cap]
    chosen_set = set(chosen)
    return [t["id"] for t in load_cancer_types() if t["id"] in chosen_set]


# AI-written (Claude Code, 2026-10-24) at the team's request - team must review


@lru_cache(maxsize=1)
def cancer_type_term_matrix() -> tuple[list[str], list[str], Any]:
    """(terms, matching type ids, normalized MiniLM vectors) for the semantic fallback.

    Why cache: the human-written detector only needs this after exact, umbrella
    and fuzzy matching fail, but embedding every label/lay/clinical term on each
    such query would be slow. We embed them once and reuse the matrix; the
    detector scores the query against it and applies the config threshold,
    margin and cap. The matrix has one row per term, so `matrix @ query_vector`
    gives one cosine score per term. Terms are de-duplicated within each type,
    and each term keeps the id of the type it belongs to, so one term can still
    map to several types.
    """
    from backend.embedding import get_encoder

    terms: list[str] = []
    type_ids: list[str] = []
    for ctype in load_cancer_types():
        raw = [ctype.get("label", ""), *ctype.get("lay_terms", []),
               *ctype.get("clinical_terms", [])]
        for term in dict.fromkeys(t.strip() for t in raw if t and t.strip()):
            terms.append(term)
            type_ids.append(ctype["id"])
    if not terms:
        import numpy as np
        return [], [], np.zeros((0, 0), dtype="float32")
    return terms, type_ids, get_encoder()(terms)




def detect_cancer_types(query: str) -> list[str]:
    """Return the cancer type ids a plain-words query refers to; [] if off-topic.

    Why layered: cheap exact matching is precise, so it goes first; typo
    tolerance and the embedding model only run when nothing exact was found, so
    they cannot add noise to queries we already understood. Anything that still
    matches nothing is off-topic: we return [] so search never touches the
    index (rule 1).

      1. Exact: every lay/clinical term or match_pattern found as whole words
         (case-insensitive, hyphens ignored), skipping types whose
         exclude_patterns hit ("cervical spine"). An organ word within a few
         words of "cancer/tumor" also counts, so "breast and lung cancer" finds
         both. UNIONED with step 2.
      2. Umbrella phrases ("blood cancer", "lung cancer") add ALL their ids,
         unless the phrase sits inside a more specific term the user typed.
      3. Fuzzy (only if 1+2 found nothing): rapidfuzz ratio of query n-grams vs
         terms longer than 3 letters ("brest cancer", "prostrate", "leukemea").
         Short acronyms are excluded because partial matching would find "aml"
         inside "camel".
      4. Embedding (only if 1-3 found nothing): MiniLM similarity of the query
         to each type's label/terms must reach TYPE_DETECT_EMBED_THRESHOLD, and
         the query must contain a cancer cue word (so "stomach ache" is not
         mistaken for stomach cancer).
    The bare word "all" never matches; only the phrase "all leukemia" or the
    upper-case acronym "ALL" does. Ids are returned without duplicates in
    cancer_types.json order.
    """
    raw = (query or "").strip()
    norm_q = _norm(raw)
    if not norm_q:
        return []
    specific = _specific_matches(raw, norm_q)
    spans = [sp for lst in specific.values() for sp in lst]
    ids = set(specific) | set(_umbrella_matches(norm_q, spans))
    if not ids:
        ids = set(_fuzzy_match(norm_q))
    if not ids:
        ids = set(_embedding_match(norm_q))
    raw_lower = raw.lower()
    return [t["id"] for t in load_cancer_types()
            if t["id"] in ids and not _is_excluded(raw_lower, t)]
