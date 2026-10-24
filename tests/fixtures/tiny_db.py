"""Tiny in-memory fixture: 6 trials, a fake encoder and a matching FAISS index.

Why fake: tests must run offline and fast, so no MiniLM download; a hashed
bag-of-words vector is enough to prove the pipeline wiring end to end.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from backend import db
from pipeline import embed

DIM = 64

TRIALS = [
    # nct, title, status, phases, conditions, tags, min, max, sites[(city,state,lat,lon)]
    ("NCT00000001", "HER2 breast cancer targeted drug", "RECRUITING", ["PHASE2"],
     ["Breast Cancer"], ["breast"], 18.0, None, [("New York", "NY", 40.75, -73.99)]),
    ("NCT00000002", "Breast cancer hormone therapy", "RECRUITING", ["PHASE1", "PHASE2"],
     ["Breast Cancer"], ["breast"], 18.0, 65.0, [("Los Angeles", "CA", 34.05, -118.24)]),
    ("NCT00000003", "Breast cancer in young people", "NOT_YET_RECRUITING", [],
     ["Breast Cancer"], ["breast"], 0.5, 17.0, [("Newark", "NJ", 40.73, -74.17)]),
    ("NCT00000004", "Non small cell lung cancer immunotherapy", "RECRUITING", ["PHASE3"],
     ["NSCLC"], ["lung_nsclc"], 18.0, None, [("New York", "NY", 40.76, -73.98)]),
    ("NCT00000005", "Small cell lung cancer chemotherapy", "ACTIVE_NOT_RECRUITING", ["PHASE2"],
     ["SCLC"], ["lung_sclc"], None, None, []),
    ("NCT00000006", "Prostate cancer radiation", "RECRUITING", ["PHASE3"],
     ["Prostate Cancer"], ["prostate"], 40.0, None, [("Boston", "MA", 42.36, -71.06)]),
]


def fake_encode(texts: list[str]) -> np.ndarray:
    """Hashed bag-of-words, L2-normalized (stands in for MiniLM)."""
    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def build_conn():
    """In-memory DB loaded with TRIALS."""
    conn = db.connect(":memory:")
    db.init_schema(conn)
    for nct, title, status, phases, conds, tags, mn, mx, sites in TRIALS:
        db.upsert_trial(conn, {
            "nct_id": nct, "brief_title": title, "official_title": f"Official: {title}",
            "status": status, "phases": phases, "conditions": conds, "keywords": [],
            "brief_summary": f"A study of {title}.", "eligibility_text": "Inclusion: adults",
            "min_age_years": mn, "max_age_years": mx, "sex": "ALL", "sponsor": "Test Sponsor",
            "start_date": "2026-01-01", "last_updated": "2026-09-01", "enrollment": 50})
        db.add_cancer_tags(conn, nct, tags)
        db.replace_locations(conn, nct, [
            {"facility": f"{c} Hospital", "city": c, "state": s, "zip": "00000",
             "country": "United States", "status": "RECRUITING", "lat": la, "lon": lo}
            for c, s, la, lo in sites])
    conn.commit()
    return conn


def build_index(conn):
    """(faiss index, row->nct list) built with the fake encoder."""
    return embed.build_index(conn, fake_encode)
