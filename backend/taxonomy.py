"""Cancer-type taxonomy: loading the lists and detecting types in a query.

Why: rule 1 (cancer only) hinges on deciding which cancer type(s) a patient
means in their own words, or deciding the query is not about cancer at all.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

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


# HUMAN-WRITTEN
def detect_cancer_types(query: str) -> list[str]:
    """Return the cancer type ids a plain-words query refers to; [] if off-topic.

    Intended algorithm:
      1. Exact: lowercase/trim the query; every type whose lay_term or
         clinical_term appears as a whole word/phrase is a match.
      2. Umbrella: every phrase from umbrella_terms.json ("blood cancer",
         "lung cancer") in the query adds ALL its listed ids.
      Steps 1 and 2 are UNIONED (do not stop at the first hit), so a mixed
      query like "breast and lung cancer" returns breast, lung_nsclc and
      lung_sclc together.
      3. Fuzzy (only if steps 1+2 found nothing): rapidfuzz partial/ratio match
         of the query (and its word n-grams) against lay + clinical terms with
         score >= config.TYPE_DETECT_FUZZY_THRESHOLD ("brest cancer",
         "prostrate"). Skip every term of 3 letters or fewer (aml, cll, cml,
         all, nhl): partial_ratio finds them inside unrelated words such as
         "small", "camel" or "ball".
      4. Embedding fallback (only if steps 1-3 found nothing): embed the query
         with MiniLM (backend.embedding.get_encoder) and compare with the
         embedded terms; accept the best type(s) only if cosine similarity >=
         config.TYPE_DETECT_EMBED_THRESHOLD.
    Short acronyms (3 letters or fewer) only count as whole words in step 1,
    and "ALL" (leukemia) only counts case-sensitively, so "AML" matches
    leukemia_aml but "small" does not.
    Rules: never match the bare word "all" (the lay term "all leukemia" only
    counts as that whole phrase, so "all of my family" is off-topic). Anything
    that matches nothing is off-topic: return [] so search never touches the
    index (rule 1). Return ids without duplicates, in a stable order.
    """
    raise NotImplementedError("HUMAN-WRITTEN: detect_cancer_types")
