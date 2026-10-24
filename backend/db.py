"""SQLite schema and small helpers.

Why SQLite: one file, no server, ships with Python, and has FTS5 full-text
search built in. The live app only READS this file (rule 4); the offline
indexer is the only writer of trials/locations/tags.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    brief_title TEXT, official_title TEXT, status TEXT,
    phases TEXT, conditions TEXT, keywords TEXT,
    brief_summary TEXT, detailed_description TEXT, eligibility_text TEXT,
    min_age_years REAL, max_age_years REAL,
    sex TEXT, healthy_volunteers TEXT, sponsor TEXT,
    start_date TEXT, last_updated TEXT, enrollment INTEGER
);
CREATE TABLE IF NOT EXISTS trial_cancer_types (
    nct_id TEXT NOT NULL, cancer_type_id TEXT NOT NULL,
    PRIMARY KEY (nct_id, cancer_type_id)
);
CREATE INDEX IF NOT EXISTS idx_tct_type ON trial_cancer_types(cancer_type_id);
CREATE TABLE IF NOT EXISTS locations (
    nct_id TEXT NOT NULL, facility TEXT, city TEXT, state TEXT, zip TEXT,
    country TEXT, status TEXT, lat REAL, lon REAL
);
CREATE INDEX IF NOT EXISTS idx_loc_nct ON locations(nct_id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    nct_id UNINDEXED, title, conditions, keywords, brief_summary, eligibility
);
CREATE TABLE IF NOT EXISTS summaries (
    nct_id TEXT PRIMARY KEY,
    what_is_tested TEXT, include_json TEXT, exclude_json TEXT,
    faithful INTEGER, fallback_used INTEGER,
    reading_before REAL, reading_after REAL,
    model_version TEXT, generated_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_UNIT_YEARS = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.1775,
               "day": 1 / 365.25, "hour": 1 / 8766, "minute": 1 / 525960}
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+?)s?\s*$")


def parse_age_years(text: str | None) -> float | None:
    """Turn '18 Years' / '6 Months' / 'N/A' into years; None if unknown.

    Why: age filtering needs one comparable number, but the API gives text.
    """
    if not text:
        return None
    m = _AGE_RE.match(text)
    if not m:
        return None
    factor = _UNIT_YEARS.get(m.group(2).lower())
    return None if factor is None else round(float(m.group(1)) * factor, 4)


def connect(path: str | Path = config.DB_PATH) -> sqlite3.Connection:
    """Open the database with row access by column name."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _j(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def upsert_trial(conn: sqlite3.Connection, rec: dict[str, Any]) -> bool:
    """Insert or replace a trial row and its FTS row. Returns True if it was new.

    Replacing (not merging) keeps the row identical to the latest API record.
    Locations are replaced separately by replace_locations.
    """
    nct = rec["nct_id"]
    existed = conn.execute("SELECT 1 FROM trials WHERE nct_id=?", (nct,)).fetchone() is not None
    conn.execute(
        """INSERT OR REPLACE INTO trials (nct_id, brief_title, official_title, status,
            phases, conditions, keywords, brief_summary, detailed_description,
            eligibility_text, min_age_years, max_age_years, sex, healthy_volunteers,
            sponsor, start_date, last_updated, enrollment)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (nct, rec.get("brief_title"), rec.get("official_title"), rec.get("status"),
         _j(rec.get("phases")), _j(rec.get("conditions")), _j(rec.get("keywords")),
         rec.get("brief_summary"), rec.get("detailed_description"),
         rec.get("eligibility_text"), rec.get("min_age_years"), rec.get("max_age_years"),
         rec.get("sex"), rec.get("healthy_volunteers"), rec.get("sponsor"),
         rec.get("start_date"), rec.get("last_updated"), rec.get("enrollment")),
    )
    conn.execute("DELETE FROM fts WHERE nct_id=?", (nct,))
    conn.execute(
        "INSERT INTO fts (nct_id, title, conditions, keywords, brief_summary, eligibility)"
        " VALUES (?,?,?,?,?,?)",
        (nct, rec.get("brief_title") or "",
         " ; ".join(rec.get("conditions") or []), " ; ".join(rec.get("keywords") or []),
         rec.get("brief_summary") or "", rec.get("eligibility_text") or ""),
    )
    return not existed


def add_cancer_tags(conn: sqlite3.Connection, nct_id: str, type_ids: Iterable[str]) -> int:
    """Tag a trial with cancer types (idempotent). Returns number of new tags."""
    n = 0
    for t in type_ids:
        n += conn.execute(
            "INSERT OR IGNORE INTO trial_cancer_types VALUES (?,?)", (nct_id, t)).rowcount
    return n


def replace_locations(conn: sqlite3.Connection, nct_id: str, locs: list[dict[str, Any]]) -> int:
    """Replace all site rows for a trial (sites change as trials update)."""
    conn.execute("DELETE FROM locations WHERE nct_id=?", (nct_id,))
    conn.executemany(
        "INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,?)",
        [(nct_id, l.get("facility"), l.get("city"), l.get("state"), l.get("zip"),
          l.get("country"), l.get("status"), l.get("lat"), l.get("lon")) for l in locs])
    return len(locs)


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))


def mark_index_run(conn: sqlite3.Connection) -> str:
    """Record the UTC date of this index run (used for --since auto mode)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    set_meta(conn, "last_index_run", ts)
    return ts


def delete_trial(conn: sqlite3.Connection, nct_id: str) -> None:
    """Remove a trial and every row that belongs to it.

    Why one helper: a half-deleted trial (say, tags left behind) would show up
    in search results with no page to open, so all tables go together.
    """
    for table in ("trials", "trial_cancer_types", "locations", "fts", "summaries"):
        conn.execute(f"DELETE FROM {table} WHERE nct_id=?", (nct_id,))
