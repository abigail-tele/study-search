"""Build data/trials_ship.db: the slim, read-only database the deployed app ships with.

Why: the full trials.db carries things only the offline pipeline needs (long detailed
descriptions, cache tables such as model_outputs). The live app (backend/search.py,
backend/db.py get_study/get_summary) reads a fixed set of tables and columns, so we copy
just those and VACUUM. The source is opened read-only through SQLite's backup API, so it
is safe while another process (batch_summarize) is writing to it; if the source is
locked we retry.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from backend import config

# Tables the live app reads (fts_* shadow tables belong to the fts virtual table).
KEEP_TABLES = {"trials", "trial_cancer_types", "locations", "fts", "summaries", "meta", "trial_interventions"}
# Columns of `trials` the app never reads (get_study / search / summarize use the rest).
DROP_TRIAL_COLUMNS = ("detailed_description",)
RETRIES = 6
RETRY_SECONDS = 5.0


def _copy_with_retry(src: Path, dst: Path) -> None:
    """Snapshot src (opened mode=ro) into dst with the SQLite backup API, retrying if locked."""
    last: Exception | None = None
    for _ in range(RETRIES):
        source = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=60)
        try:
            dest = sqlite3.connect(str(dst))
            try:
                source.backup(dest)
                return
            finally:
                dest.close()
        except sqlite3.OperationalError as exc:  # locked / busy
            last = exc
            time.sleep(RETRY_SECONDS)
        finally:
            source.close()
    raise RuntimeError(f"could not snapshot {src}: {last}")


def _trim(path: Path) -> None:
    """Drop everything the live app does not read, then VACUUM."""
    conn = sqlite3.connect(str(path))
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for name in names:
            if name in KEEP_TABLES or name.startswith("fts_"):
                continue
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)")}
        for col in DROP_TRIAL_COLUMNS:
            if col in cols:
                conn.execute(f'ALTER TABLE trials DROP COLUMN "{col}"')
        conn.execute("DELETE FROM meta WHERE key LIKE 'interv_checked:%'")  # backfill bookkeeping only
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def make_ship_db(src: Path = config.DEFAULT_DB_PATH, dst: Path = config.SHIP_DB_PATH) -> tuple[int, int]:
    """Build dst from src; return (source bytes, ship bytes)."""
    if dst.exists():
        dst.unlink()
    tmp = dst.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    _copy_with_retry(src, tmp)
    _trim(tmp)
    tmp.replace(dst)
    return src.stat().st_size, dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=config.DEFAULT_DB_PATH)
    ap.add_argument("--dst", type=Path, default=config.SHIP_DB_PATH)
    a = ap.parse_args()
    before, after = make_ship_db(a.src, a.dst)
    print(f"{a.src}: {before / 1e6:.1f} MB -> {a.dst}: {after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
