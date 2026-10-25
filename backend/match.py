"""'About you' matching: compare a person's age, sex and typed conditions with ONE trial's rules.

Why this exists: patients want to know which trials even mention their situation. This is a
READING AID, never a verdict (rule 7): the words are "Possible match" / "Check with study team" /
"Likely not a match", and the study team always decides who can join.

Rules in one line each (a student should be able to explain every one):
  1. Age and sex are compared with the trial's STORED fields (rule 2), never with model text.
  2. Each typed condition is cut into short lowercase terms (parse_conditions).
  3. A term matches a criterion if the criterion contains the term as whole words (after tidying
     "stage III" -> "stage 3", "HER2+" -> "her2 positive"), or one of its synonyms (SYNONYMS below).
  4. If no phrase matches, a term WITHOUT digits can still match by meaning: MiniLM cosine between
     the term and the criterion >= config.MATCH_SIM_THRESHOLD. Digit terms (stages, HER2 status)
     need an exact phrase because meaning-search confuses "stage 3" with "stage 4".
  5. A match in an INCLUSION criterion is a requirement the person seems to meet, UNLESS the
     criterion is negated ("no prior chemotherapy") - then it is a possible conflict.
  6. A match in an EXCLUSION criterion is a concern.
  7. Label: age outside / sex mismatch -> "Likely not a match"; any exclusion mention or conflict
     -> "Check with study team"; otherwise "Possible match".
Terms the person wrote as an absence ("no chemo", "never smoked") are ignored, not matched.
Rule 8: nothing here logs or stores the person's input; caches hold only public trial text.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable, Iterable

import numpy as np

from backend import config

LABEL_POSSIBLE = "Possible match"
LABEL_CHECK = "Check with study team"
LABEL_NOT = "Likely not a match"
LABEL_ORDER = {LABEL_POSSIBLE: 0, LABEL_CHECK: 1, LABEL_NOT: 2}
LABEL_TEXT = {  # what the UI shows; always says the study team decides (rule 7)
    LABEL_POSSIBLE: "Possible match — the study team decides who can join",
    LABEL_CHECK: "Check with study team — they decide who can join",
    LABEL_NOT: "Likely not a match — the study team decides who can join",
}

# Groups of words that mean the same thing in a criterion. Normalized (lowercase, no punctuation).
# Plain data on purpose: add a line to teach the matcher a new synonym.
SYNONYMS: list[set[str]] = [
    {"chemotherapy", "chemo", "chemotherapeutic", "cytotoxic therapy"},
    {"radiation", "radiotherapy", "radiation therapy", "irradiation"},
    {"surgery", "surgical", "resection"},
    {"immunotherapy", "checkpoint inhibitor", "pd 1", "pd l1"},
    {"hormone therapy", "endocrine therapy", "hormonal therapy"},
    {"pregnant", "pregnancy", "breastfeeding", "nursing", "lactating"},
    {"diabetes", "diabetic", "diabetes mellitus"},
    {"her2 positive", "her2 overexpression", "her2 amplified", "erbb2 positive"},
    {"her2 negative", "her2 low"},
    {"heart disease", "cardiac disease", "cardiovascular disease", "heart failure",
     "coronary artery disease", "myocardial infarction"},
    {"high blood pressure", "hypertension"},
    {"kidney disease", "renal disease", "renal insufficiency", "kidney failure"},
    {"liver disease", "hepatic impairment", "cirrhosis"},
    {"hiv", "human immunodeficiency virus"},
    {"brain metastases", "brain mets", "cns metastases", "central nervous system metastases"},
    {"smoker", "smoking", "tobacco use", "tobacco"},
    {"autoimmune disease", "autoimmune"},
]

_NEG_CUES = re.compile(
    r"\b(no|not|without|never|none|absence of|free of|cannot|unable|must not|"
    r"no history of|no prior|no evidence of)\b")
_USER_ABSENCE = re.compile(r"^(no|not|never|without|none|denies|negative for)\b")
_SPLIT = re.compile(r"[,;\n]+|\band\b")
_LEAD_IN = re.compile(r"^(?:i |im |am |have |has |had |having |history of |a history of |prior |previous |previously |"
                      r"currently |taking |on |with |diagnosed with |treated with |received |was |were )+")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}
_STAGE = re.compile(r"\bstage\s+(iv|iii|ii|i)([abc])?\b")
_ENCODER_DEFAULT = object()   # "use the real local MiniLM"; tests pass a fake, None means no embeddings


# ------------------------------------------------------------------ text helpers
def parse_conditions(text: str | None) -> list[str]:
    """Free text -> up to MATCH_MAX_TERMS short lowercase terms.

    Splits on commas, semicolons, new lines and the word "and" (so "heart and lung disease"
    becomes two terms - a known, accepted limit). Each term is trimmed, cut to
    MATCH_MAX_TERM_CHARS, and duplicates are dropped.
    """
    out: list[str] = []
    for piece in _SPLIT.split((text or "").lower()):
        term = re.sub(r"\s+", " ", piece).strip(" .-\t")[:config.MATCH_MAX_TERM_CHARS].strip()
        if term and term not in out:
            out.append(term)
        if len(out) >= config.MATCH_MAX_TERMS:
            break
    return out


def normalize(text: str) -> str:
    """Lowercase and tidy so spelling variants compare equal: 'HER2+' = 'her2 positive', 'Stage III' = 'stage 3'."""
    t = (text or "").lower()
    t = re.sub(r"her[- ]?2\s*\+", "her2 positive", t)
    t = re.sub(r"her[- ]?2\b", "her2", t)
    t = re.sub(r"[‐-―\-/]", " ", t)
    t = re.sub(r"[^a-z0-9+ ]", " ", t).replace("+", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return _STAGE.sub(lambda m: f"stage {_ROMAN[m.group(1)]}{m.group(2) or ''}", t)


def core_term(term: str) -> str:
    """Drop lead-in words so 'had chemotherapy' and 'history of diabetes' match criteria that say 'chemotherapy' and 'diabetes'."""
    n = normalize(term)
    return _LEAD_IN.sub("", n).strip() or n


def variants(term: str) -> set[str]:
    """The term plus every synonym of it (all normalized)."""
    base = core_term(term)
    out = {base}
    for group in SYNONYMS:
        if base in group:
            out |= group
    return {v for v in out if v}


def _phrase_regex(phrase: str) -> re.Pattern:
    tail = "[abc]?" if re.fullmatch(r"stage \d", phrase) else "(?:s)?"   # 'stage 3' also finds 'stage 3a'
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + tail + r"(?![a-z0-9])")


def find_phrase(term: str, criterion_norm: str) -> int | None:
    """Start index of the first whole-word hit of the term or a synonym in a normalized criterion, else None."""
    hits = [m.start() for v in variants(term) if (m := _phrase_regex(v).search(criterion_norm))]
    return min(hits) if hits else None


def is_negated(criterion_norm: str, hit_at: int | None) -> bool:
    """True if a negation cue ('no', 'not', 'without', 'never'...) appears before the hit (or anywhere, for a meaning match)."""
    zone = criterion_norm if hit_at is None else criterion_norm[max(0, hit_at - 40):hit_at]
    return bool(_NEG_CUES.search(zone))


# ------------------------------------------------------------------ age / sex
def age_status(min_years: float | None, max_years: float | None, age: float | None) -> str:
    """'ok' | 'outside' | 'unknown' (unknown = the person gave no age). A missing limit means no limit."""
    if age is None:
        return "unknown"
    if (min_years is not None and age < min_years) or (max_years is not None and age > max_years):
        return "outside"
    return "ok"


def sex_status(trial_sex: str | None, sex: str | None) -> str:
    """'ok' | 'no' | 'unknown'. Trial sex is ALL/FEMALE/MALE from the API; the person's is female/male/any."""
    s = (sex or "").lower()
    if s not in ("female", "male"):
        return "unknown"
    t = (trial_sex or "ALL").upper()
    return "ok" if t in ("ALL", "", s.upper()) else "no"


# ------------------------------------------------------------------ embeddings
_vec_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _criteria_vectors(key: str, texts: list[str], encoder: Callable) -> np.ndarray:
    """MiniLM vectors for a trial's criteria, computed once per trial (small LRU cache; public text only)."""
    ck = (key, hash(tuple(texts)))
    if ck in _vec_cache:
        _vec_cache.move_to_end(ck)
        return _vec_cache[ck]
    vecs = np.asarray(encoder(texts), dtype="float32")
    _vec_cache[ck] = vecs
    while len(_vec_cache) > config.MATCH_CACHE_SIZE:
        _vec_cache.popitem(last=False)
    return vecs


def _resolve_encoder(encoder: Any) -> Callable | None:
    if encoder is not _ENCODER_DEFAULT:
        return encoder
    try:
        from backend.embedding import get_encoder
        return get_encoder()
    except Exception:   # model missing: phrase matching still works
        return None


# ------------------------------------------------------------------ main matcher
def _match_side(terms: list[str], crits: list[str], key: str, sim_vecs: np.ndarray | None,
                encoder: Callable | None) -> list[dict]:
    """Every (term, criterion) hit on one side, with a `negated` flag. Order: by term, then criterion."""
    norms = [normalize(c) for c in crits]
    crit_vecs = None
    if encoder is not None and sim_vecs is not None and crits:
        short = [c[:config.MATCH_CRITERION_MAX_CHARS] for c in crits]
        crit_vecs = _criteria_vectors(key, short, encoder)
    hits: list[dict] = []
    for ti, term in enumerate(terms):
        found = 0
        for ci, crit in enumerate(crits):
            at = find_phrase(term, norms[ci])
            by_meaning = False
            if at is None and crit_vecs is not None and sim_vecs is not None and not re.search(r"\d", term) \
                    and len(crit) <= config.MATCH_CRITERION_MAX_CHARS \
                    and float(crit_vecs[ci] @ sim_vecs[ti]) >= config.MATCH_SIM_THRESHOLD:
                by_meaning = True
            if at is None and not by_meaning:
                continue
            hits.append({"term": term, "criterion": crit, "negated": is_negated(norms[ci], at)})
            found += 1
            if found >= config.MATCH_MAX_PER_TERM:
                break
    return hits


def match_trial(trial_fields: dict, eligibility_text: str | None, age: float | None = None,
                sex: str | None = None, conditions: Iterable[str] = (),
                encoder: Any = _ENCODER_DEFAULT) -> dict:
    """Compare one person with one trial. Pure: no database, no logging, nothing stored.

    trial_fields needs min_age_years, max_age_years, sex (and nct_id for the cache key).
    `conditions` is the output of parse_conditions. Returns the API `match` object.
    """
    include, exclude = split_criteria(eligibility_text)
    terms = [t for t in conditions if not _USER_ABSENCE.match(t)]
    ignored = [t for t in conditions if _USER_ABSENCE.match(t)]
    requirements: list[dict] = []
    exclusions: list[dict] = []
    conflicts: list[dict] = []
    if terms and (include or exclude):
        enc = _resolve_encoder(encoder)
        sim_vecs = np.asarray(enc([core_term(t) for t in terms]), dtype="float32") if enc is not None else None
        key = str(trial_fields.get("nct_id") or "")
        for h in _match_side(terms, include, key + ":i", sim_vecs, enc):
            item = {"term": h["term"], "criterion": h["criterion"]}
            (conflicts if h["negated"] else requirements).append(item)
        for h in _match_side(terms, exclude, key + ":e", sim_vecs, enc):
            exclusions.append({"term": h["term"], "criterion": h["criterion"]})
    a = age_status(trial_fields.get("min_age_years"), trial_fields.get("max_age_years"), age)
    s = sex_status(trial_fields.get("sex"), sex)
    if a == "outside" or s == "no":
        label = LABEL_NOT
    elif exclusions or conflicts:
        label = LABEL_CHECK
    else:
        label = LABEL_POSSIBLE
    return {"age": a, "sex": s, "mentioned_in_requirements": requirements,
            "mentioned_in_exclusions": exclusions, "possible_conflicts": conflicts,
            "ignored_terms": ignored, "summary_label": label, "label_text": LABEL_TEXT[label]}


EMPTY_MATCH = {"mentioned_in_requirements": [], "mentioned_in_exclusions": [], "possible_conflicts": []}


def split_criteria(eligibility_text: str | None) -> tuple[list[str], list[str]]:
    """(include, exclude) criteria lists; one place so the app and match_trial split text the same way."""
    from backend.summarize import split_eligibility   # lazy: summarize pulls in heavy libraries
    return split_eligibility(eligibility_text)


def flag_criteria(include: list[str], exclude: list[str], match: dict) -> dict:
    """Mark each original criterion for the study panel: flag 'match' (green check), 'concern' (amber warning) or None.

    `text` is the original wording with plain definitions added after jargon (glossary_swap, no model),
    so it is a safe fallback when no model summary exists. `original` is the untouched criterion.
    """
    from backend.summarize import glossary_swap
    def terms_for(items: list[dict], crit: str) -> list[str]:
        return [i["term"] for i in items if i["criterion"] == crit]
    inc = []
    for c in include:
        concern = terms_for(match["possible_conflicts"], c)
        good = terms_for(match["mentioned_in_requirements"], c)
        inc.append({"text": glossary_swap(c), "original": c, "flag": "concern" if concern else ("match" if good else None),
                    "terms": concern or good})
    exc = []
    for c in exclude:
        t = terms_for(match["mentioned_in_exclusions"], c)
        exc.append({"text": glossary_swap(c), "original": c, "flag": "concern" if t else None, "terms": t})
    return {"include": inc, "exclude": exc}


# ------------------------------------------------------------------ database glue
def load_trial_fields(conn: Any, nct: str) -> tuple[dict, str] | None:
    """(stored fields, eligibility text) for one trial, or None. Read-only."""
    row = conn.execute("SELECT nct_id, min_age_years, max_age_years, sex, eligibility_text "
                       "FROM trials WHERE nct_id=?", (nct,)).fetchone()
    if row is None:
        return None
    return dict(row), row["eligibility_text"]


def match_for_nct(conn: Any, nct: str, age: float | None, sex: str | None,
                  conditions: list[str], encoder: Any = _ENCODER_DEFAULT) -> dict | None:
    """match_trial for a stored trial id; None if the trial is not in the database."""
    loaded = load_trial_fields(conn, nct)
    if loaded is None:
        return None
    fields, elig = loaded
    return match_trial(fields, elig, age=age, sex=sex, conditions=conditions, encoder=encoder)


def sort_by_match(results: list[dict]) -> list[dict]:
    """Possible match first, then Check, then Likely not; the relevance order is kept inside each group (sort is stable)."""
    return sorted(results, key=lambda r: LABEL_ORDER.get((r.get("match") or {}).get("summary_label"), 1))
