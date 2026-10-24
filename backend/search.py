"""Hybrid cancer-trial search: tag candidates -> keyword + meaning ranks -> RRF -> filters.

Why this shape: rule 1 says we only ever search inside trials tagged with the
cancer type(s) the patient meant (never the whole index for off-topic text);
rule 4 says no network at request time (SQLite + FAISS only); rule 8 says the
query and ZIP are used in memory and never logged or stored.
"""
from __future__ import annotations

import json
import re
import sqlite3
from math import cos, radians
from typing import Any, Iterable

import faiss
import numpy as np

from backend import config, geo, taxonomy
from backend.embedding import get_encoder

_index_cache: tuple[Any, list[str]] | None = None
_STOPWORDS = {"a", "an", "the", "of", "in", "my", "me", "i", "for", "and", "or", "to", "near",
              "is", "with", "on", "have", "has", "do", "how", "can", "what", "about", "trial",
              "trials", "clinical", "study", "studies", "cancer"}


# ---------------------------------------------------------------- pure helpers
def build_citation(trial: dict) -> str:
    """Exact citation string from stored fields (never from the model).

    Format: {Sponsor}. {Official title}. ClinicalTrials.gov ID: {NCT}. Last
    updated {YYYY-MM-DD}. Available from: https://clinicaltrials.gov/study/{NCT}
    Falls back to the brief title when there is no official title.
    """
    title = trial.get("official_title") or trial.get("brief_title") or trial.get("title") or ""
    nct = trial["nct_id"]
    return (f"{trial.get('sponsor') or 'Unknown sponsor'}. {title}. ClinicalTrials.gov ID: {nct}. "
            f"Last updated {trial.get('last_updated') or 'unknown'}. "
            f"Available from: {config.STUDY_URL_TEMPLATE.format(nct=nct)}")


def normalize_phase(phases_json: str | None) -> str:
    """'["PHASE1","PHASE2"]' -> 'PHASE1|PHASE2'; none -> 'NA'."""
    try:
        phases = json.loads(phases_json or "[]")
    except (TypeError, ValueError):
        phases = []
    return "|".join(phases) if phases else "NA"


def allowed_statuses(status: str | None) -> tuple[str, ...]:
    """None/'ANY' -> every indexed status; otherwise that one (must be known)."""
    if status is None or status.upper() == "ANY":
        return config.INDEX_STATUSES
    if status.upper() not in config.INDEX_STATUSES:
        raise ValueError(f"unknown status: {status}")
    return (status.upper(),)


def age_ok(min_age: float | None, max_age: float | None, age: float | None) -> bool:
    """True if age fits the trial's limits; a NULL limit means no limit."""
    if age is None:
        return True
    return (min_age is None or min_age <= age) and (max_age is None or age <= max_age)


def bounding_box(center: geo.LatLon, radius_mi: float) -> tuple[float, float, float, float]:
    """(lat_lo, lat_hi, lon_lo, lon_hi) box around a point, a cheap SQL pre-filter."""
    dlat = radius_mi / 69.0
    dlon = radius_mi / (69.0 * max(cos(radians(center[0])), 0.01))
    return center[0] - dlat, center[0] + dlat, center[1] - dlon, center[1] + dlon


def fts_query(text: str) -> str:
    """Make user text safe for FTS5: quoted word tokens joined by OR, stopwords dropped."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


# HUMAN-WRITTEN
def reciprocal_rank_fusion(rankings: list[list[str]],
                           k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Merge several ranked lists of ids into one list of (id, score), best first.

    Formula: score(id) = sum over every list containing id of 1 / (k + rank),
    where rank starts at 1 for the first item. An id found in only one list
    just gets that one term. Sort by score descending; break ties
    deterministically (by id ascending) so results are stable. Empty input -> [].
    """
    raise NotImplementedError("HUMAN-WRITTEN: reciprocal_rank_fusion")


# ---------------------------------------------------------------- resources
def _get_index() -> tuple[Any, list[str]]:
    """Load the FAISS index and row map once per process."""
    global _index_cache
    if _index_cache is None:
        idx = faiss.read_index(str(config.FAISS_INDEX_PATH))
        rowmap = json.loads(config.FAISS_ROWMAP_PATH.read_text(encoding="utf-8"))
        _index_cache = (idx, rowmap)
    return _index_cache


# ---------------------------------------------------------------- retrieval
def candidate_ids(conn: sqlite3.Connection, types: list[str], statuses: Iterable[str],
                  age: float | None, origin: geo.LatLon | None,
                  radius_mi: float | None) -> list[str]:
    """Trials tagged with the types that also pass status, age and radius filters.

    Why filter BEFORE ranking: ranking inside the eligible set means the top 10
    are all usable, instead of filtering a top-50 down to almost nothing.
    """
    ss = tuple(statuses)
    rows = conn.execute(
        f"SELECT DISTINCT t.nct_id, t.min_age_years, t.max_age_years FROM trials t "
        f"JOIN trial_cancer_types c ON c.nct_id = t.nct_id "
        f"WHERE c.cancer_type_id IN ({','.join('?' * len(types))}) "
        f"AND t.status IN ({','.join('?' * len(ss))})",
        (*types, *ss)).fetchall()
    ids = [r["nct_id"] for r in rows if age_ok(r["min_age_years"], r["max_age_years"], age)]
    if origin is not None and radius_mi is not None:
        lo_lat, hi_lat, lo_lon, hi_lon = bounding_box(origin, radius_mi)
        near = {r[0] for r in conn.execute(
            "SELECT DISTINCT nct_id FROM locations "
            "WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            (lo_lat, hi_lat, lo_lon, hi_lon))}
        ids = [i for i in ids if i in near]
    return ids


def fts_ranking(conn: sqlite3.Connection, text: str, cands: list[str], k: int) -> list[str]:
    """Top-k candidate ids by FTS5 bm25 (best first)."""
    match = fts_query(text)
    if not match or not cands:
        return []
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS cand (nct_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM cand")
    conn.executemany("INSERT OR IGNORE INTO cand VALUES (?)", [(c,) for c in cands])
    rows = conn.execute(
        "SELECT nct_id FROM fts WHERE fts MATCH ? AND nct_id IN (SELECT nct_id FROM cand) "
        "ORDER BY rank LIMIT ?", (match, k)).fetchall()
    conn.execute("DELETE FROM cand")
    return [r[0] for r in rows]


def faiss_ranking(text: str, cands: list[str], k: int) -> list[str]:
    """Top-k candidate ids by embedding similarity, searching only inside cands."""
    if not cands:
        return []
    index, rowmap = _get_index()
    row_of = {n: i for i, n in enumerate(rowmap)}
    rows = np.array([row_of[c] for c in cands if c in row_of], dtype="int64")
    if rows.size == 0:
        return []
    vec = get_encoder()([text])
    params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(rows))
    _, idx = index.search(vec, min(k, int(rows.size)), params=params)
    return [rowmap[i] for i in idx[0] if i >= 0]


# ---------------------------------------------------------------- results
def build_results(conn: sqlite3.Connection, fused: list[tuple[str, float]],
                  zip_code: str | None, radius_mi: float | None) -> list[dict]:
    """Turn fused (nct, score) pairs into result items; apply the exact radius check."""
    out: list[dict] = []
    for nct, score in fused:
        t = conn.execute("SELECT * FROM trials WHERE nct_id=?", (nct,)).fetchone()
        if t is None:
            continue
        site = None
        if zip_code:
            locs = conn.execute("SELECT city, state, lat, lon FROM locations WHERE nct_id=?",
                                (nct,)).fetchall()
            site = geo.nearest_site(locs, zip_code)
            if radius_mi is not None and (site is None or site["distance_mi"] > radius_mi):
                continue
        tags = [r[0] for r in conn.execute(
            "SELECT cancer_type_id FROM trial_cancer_types WHERE nct_id=? ORDER BY cancer_type_id",
            (nct,))][:config.MAX_CANCER_TAGS_ON_CARD]
        out.append({
            "nct_id": nct, "title": t["brief_title"], "status": t["status"],
            "phase": normalize_phase(t["phases"]), "cancer_tags": tags,
            "sponsor": t["sponsor"], "last_updated": t["last_updated"], "nearest_site": site,
            "url": config.STUDY_URL_TEMPLATE.format(nct=nct),
            "citation": build_citation(dict(t)), "score": round(score, 4),
        })
        if len(out) >= config.RESULT_LIMIT:
            break
    return out


def search(q: str, cancer: str | list[str] | None = None, zip: str | None = None,
           radius_mi: float | None = None, status: str | None = None,
           age: float | None = None, conn: sqlite3.Connection | None = None) -> dict:
    """Run a search and return exactly the API contract shape (PLAN section 2).

    `cancer` (a chip click: one id or a list) overrides detection. If no cancer
    type is found the query is off-topic: we return the chips and NO results
    and never touch the index (rule 1). q and zip are never logged or stored.
    An unknown/invalid ZIP is ignored (no distance filter); validate in the API.
    """
    if isinstance(cancer, str):
        cancer = [cancer]
    known = {t["id"] for t in taxonomy.load_cancer_types()}
    types = [c for c in (cancer or []) if c in known] or taxonomy.detect_cancer_types(q or "")
    if not types:
        return {"query": q, "detected_types": [], "off_topic": True,
                "suggested_types": taxonomy.cancer_type_chips(), "results": []}
    statuses = allowed_statuses(status)
    if conn is None:
        from backend import db
        conn = db.connect()
    origin = geo.lookup(zip) if zip else None
    cands = candidate_ids(conn, types, statuses, age, origin, radius_mi if origin else None)
    text = (q or "").strip() or " ".join(
        t["label"] for t in taxonomy.load_cancer_types() if t["id"] in types)
    fused = reciprocal_rank_fusion(
        [fts_ranking(conn, text, cands, config.FTS_TOP_K),
         faiss_ranking(text, cands, config.FAISS_TOP_K)], config.RRF_K)
    results = build_results(conn, fused, zip if origin else None,
                            radius_mi if origin else None)
    return {"query": q, "detected_types": types, "off_topic": False,
            "suggested_types": [], "results": results}
