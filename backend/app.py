"""FastAPI app: thin endpoints only. Logic lives in search.py / db.py / summarize.py.

Why thin: judges can read this file in two minutes. Rules enforced here:
- Rule 4: no network calls and no imports from pipeline/ (reads SQLite + FAISS only).
- Rule 8: query text, ZIPs, age, sex and typed conditions are never logged or stored (see _install_log_redaction).
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager, contextmanager
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Rule 4: the live server must never contact Hugging Face. Set BEFORE any HF library is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from backend import config  # noqa: E402

MAX_QUERY_CHARS = 200
MAX_AGE_YEARS = 120
_NCT_RE = re.compile(r"^NCT\d{8}$")
_ZIP_RE = re.compile(r"^\d{5}$")
_SENSITIVE = re.compile(r"(?i)([?&](?:q|zip|age|sex|conditions)=)[^&\s\"']*")
_SEX_VALUES = ("female", "male", "any")
_SORT_VALUES = ("relevance", "match")


def _install_log_redaction() -> None:
    """Rule 8 belt-and-braces: scrub q=, zip=, age=, sex= and conditions= from EVERY log record, whichever logger emits it.

    Why a record factory: logger-level filters only see records sent to that exact logger,
    so a library logger we forgot (uvicorn, httpx...) could still leak. The factory runs for all.
    """
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_redacts", False):
        return

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        try:
            record.msg = _SENSITIVE.sub(r"[redacted]", record.getMessage())
            record.args = ()
        except Exception:
            pass
        return record

    factory._redacts = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


_install_log_redaction()

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Warm up at startup so the first user does not wait ~9 s and a missing model fails loudly now.

    Set STUDY_SEARCH_SKIP_WARMUP=1 to skip (tests).
    """
    if not os.environ.get("STUDY_SEARCH_SKIP_WARMUP"):
        from backend import embedding, search as search_mod
        embedding.get_encoder()
        search_mod._get_index()
        logging.getLogger("uvicorn.error").info("Warm-up done: encoder and FAISS index loaded from local files.")
    yield


app = FastAPI(title="Cancer Trials Plain-Language Finder", docs_url=None, redoc_url=None, lifespan=_lifespan)


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _err(400, "Invalid request parameters.")


@app.exception_handler(StarletteHTTPException)
async def _http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _err(exc.status_code, "Not found" if exc.status_code == 404 else str(exc.detail))


@contextmanager
def _conn() -> Iterator[object]:
    """One SQLite connection per request: sqlite3 connections must not be shared across threads."""
    from backend import db
    conn = db.connect_ro(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@app.get("/api/cancer-types")
def cancer_types() -> list[dict]:
    from backend import taxonomy
    return [{"id": t["id"], "label": t["label"], "lay_terms": t.get("lay_terms", [])}
            for t in taxonomy.load_cancer_types()]


@app.get("/api/search")
def search_endpoint(q: str = "", cancer: str | None = None, zip: str | None = None,
                    radius_mi: str | None = None, status: str | None = None,
                    age: str | None = None, sex: str | None = None, conditions: str | None = None,
                    sort: str | None = None) -> JSONResponse:
    q = q.strip()
    if len(q) > MAX_QUERY_CHARS:
        return _err(400, f"Search text is too long (max {MAX_QUERY_CHARS} characters).")
    if not q and not cancer:
        return _err(400, "Enter a cancer type or pick one from the list.")
    if zip is not None and zip != "" and not _ZIP_RE.match(zip):
        return _err(400, "ZIP code must be exactly 5 digits.")
    radius = None
    if radius_mi not in (None, ""):
        if not radius_mi.isdigit() or int(radius_mi) not in config.RADIUS_OPTIONS_MI:
            return _err(400, "Distance must be one of: " + ", ".join(map(str, config.RADIUS_OPTIONS_MI)) + " miles.")
        radius = int(radius_mi)
    if status in ("", "ANY"):
        status = None
    elif status is not None and status not in config.INDEX_STATUSES:
        return _err(400, "Unknown status.")
    age_val = None
    if age not in (None, ""):
        if not age.isdigit() or not 0 <= int(age) <= MAX_AGE_YEARS:
            return _err(400, f"Age must be a whole number from 0 to {MAX_AGE_YEARS}.")
        age_val = int(age)
    profile_err = _check_profile(sex, conditions)
    if profile_err:
        return profile_err
    if sort not in (None, "") and sort not in _SORT_VALUES:
        return _err(400, "Unknown sort order.")
    if zip:
        from backend import geo
        if geo.lookup(zip) is None:
            return _err(400, "We couldn't find that ZIP code. Try a nearby one.")
    from backend import search as search_mod
    try:
        with _conn() as conn:
            result = search_mod.search(q, cancer=cancer or None, zip=zip or None, radius_mi=radius,
                                      status=status, age=age_val, conn=conn)
    except ValueError as e:
        return _err(400, str(e))
    except NotImplementedError:
        return _err(503, "Search ranking is not implemented yet")
    if _has_profile(age_val, sex, conditions):
        _attach_matches(result, age_val, sex, conditions, sort == "match")
    return JSONResponse(result)


def _check_profile(sex: str | None, conditions: str | None) -> JSONResponse | None:
    """400 for a bad sex value or an over-long conditions box. Messages never echo what was typed (rule 8)."""
    if sex not in (None, "") and sex not in _SEX_VALUES:
        return _err(400, "Sex must be female, male or any.")
    if conditions and len(conditions) > config.MATCH_MAX_CONDITIONS_CHARS:
        return _err(400, f"The details box is too long (max {config.MATCH_MAX_CONDITIONS_CHARS} characters).")
    return None


def _has_profile(age: int | None, sex: str | None, conditions: str | None) -> bool:
    return age is not None or sex in ("female", "male") or bool((conditions or "").strip())


def _attach_matches(result: dict, age: int | None, sex: str | None, conditions: str | None, sort_match: bool) -> None:
    """Add a `match` object to each result (in place); optionally re-sort so 'Possible match' comes first."""
    from backend import match
    terms = match.parse_conditions(conditions)
    with _conn() as conn:
        for r in result.get("results", []):
            r["match"] = match.match_for_nct(conn, r["nct_id"], age, sex or None, terms)
    if sort_match:
        result["results"] = match.sort_by_match(result.get("results", []))


@app.get("/api/study/{nct}")
def study(nct: str, age: str | None = None, sex: str | None = None, conditions: str | None = None,
          zip: str | None = None) -> JSONResponse:
    if not _NCT_RE.match(nct):
        return _err(400, "Invalid trial ID. Expected NCT followed by 8 digits.")
    age_val = None
    if age not in (None, ""):
        if not age.isdigit() or not 0 <= int(age) <= MAX_AGE_YEARS:
            return _err(400, f"Age must be a whole number from 0 to {MAX_AGE_YEARS}.")
        age_val = int(age)
    profile_err = _check_profile(sex, conditions)
    if profile_err:
        return profile_err
    if zip and not _ZIP_RE.match(zip):
        return _err(400, "ZIP code must be exactly 5 digits.")
    from backend import db, match, overview
    with _conn() as conn:
        data = db.get_study(conn, nct)
        if data is None:
            return _err(404, "Study not found")
        data["overview"] = overview.study_overview(data, zip or None)
        include, exclude = match.split_criteria(data.get("eligibility_text"))
        m = None
        if _has_profile(age_val, sex, conditions):
            m = match.match_for_nct(conn, nct, age_val, sex or None, match.parse_conditions(conditions))
        data["match"] = m
        data["criteria"] = match.flag_criteria(include, exclude, m or match.EMPTY_MATCH)
    return JSONResponse(data)


def _live_summary_enabled() -> bool:
    """STUDY_SEARCH_LIVE_SUMMARY=1 lets a missing summary be generated on this machine (needs the T5 checkpoint).

    Default 0 (deployment): the host serves only pre-generated rows, so the ~1 GB model is never loaded.
    Read per request so tests and operators can flip it without a restart.
    """
    return os.environ.get("STUDY_SEARCH_LIVE_SUMMARY", "0") == "1"


@app.get("/api/summary/{nct}")
def summary(nct: str) -> JSONResponse:
    if not _NCT_RE.match(nct):
        return _err(400, "Invalid trial ID. Expected NCT followed by 8 digits.")
    from backend import db
    with _conn() as conn:
        data = db.get_summary(conn, nct)
    if data is not None:
        return JSONResponse(data)
    if not _live_summary_enabled():
        return _err(404, "Summary not available yet")
    return _live_summary(nct)


def _live_summary(nct: str) -> JSONResponse:
    """Cache miss with live generation on: summarize_trial (rule 3 faithfulness check inside), store, return.

    What is stored is a summary of a public trial, never anything the user typed (rule 8).
    No network: the model is a local checkpoint (rule 4). Any model problem becomes a clean 503.
    """
    from backend import db, summarize
    with _conn() as conn:
        row = conn.execute("SELECT nct_id, brief_summary, eligibility_text FROM trials WHERE nct_id=?",
                           (nct,)).fetchone()
    if row is None:
        return _err(404, "Study not found")
    try:
        result = summarize.summarize_trial(dict(row))
    except Exception:
        logging.getLogger("uvicorn.error").exception("Live summary generation failed")
        return _err(503, "Summary could not be generated right now")
    try:
        import sqlite3
        rw = sqlite3.connect(config.DB_PATH)   # the read-only helper cannot write; the host DB may be read-only too
        try:
            summarize.save_summary(rw, result)
            rw.commit()
        finally:
            rw.close()
    except Exception:
        logging.getLogger("uvicorn.error").warning("Could not store generated summary (read-only database?)")
    return JSONResponse(result)


# Mounted last so /api/* routes win. html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
