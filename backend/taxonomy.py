"""Cancer-type taxonomy: loading the lists and detecting types in a query.

Why: rule 1 (cancer only) hinges on deciding which cancer type(s) a patient
means in their own words, or deciding the query is not about cancer at all.
"""
from __future__ import annotations

import json
from functools import lru_cache

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


# HUMAN-WRITTEN
def detect_cancer_types(query: str) -> list[str]:
    """Return the cancer type ids a plain-words query refers to; [] if off-topic.

    Intended algorithm (try each step in order; stop at the first that matches):
      1. Exact match: lowercase/trim the query; if it contains a lay_term or
         clinical_term of a type (whole-word match), return that type's id.
      2. Umbrella terms: if it contains a phrase from umbrella_terms.json
         ("blood cancer", "lung cancer"), return ALL the listed ids.
      3. Fuzzy: rapidfuzz partial/ratio match of the query (and its word
         n-grams) against all lay + clinical terms with score >=
         config.TYPE_DETECT_FUZZY_THRESHOLD ("brest cancer", "prostrate").
      4. Embedding fallback: embed the query with MiniLM (backend.embedding.
         get_encoder) and compare with the embedded terms; accept the best
         type(s) only if cosine similarity >= config.TYPE_DETECT_EMBED_THRESHOLD.
    Rules: never match the bare word "all" (the lay term "all leukemia" only
    counts as that whole phrase, so "all of my family" is off-topic). Anything
    that matches nothing is off-topic: return [] so search never touches the
    index (rule 1). Return ids without duplicates, in a stable order.
    """
    raise NotImplementedError("HUMAN-WRITTEN: detect_cancer_types")
