"""FastAPI app: thin endpoints only. Logic lives in search.py / db.py / summarize.py.

Why thin: judges can read this file in two minutes. Rules enforced here:
- Rule 4: no network calls and no imports from pipeline/ (reads SQLite + FAISS only).
- Rule 8: query text and ZIPs are never logged or stored (see _install_log_redaction).
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
_SENSITIVE = re.compile(r"(?i)([?&](?:q|zip)=)[^&\s\"']*")


def _install_log_redaction() -> None:
    """Rule 8 belt-and-braces: scrub q= and zip= from EVERY log record, whichever logger emits it.

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
                    age: str | None = None) -> JSONResponse:
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
    return JSONResponse(result)


@app.get("/api/study/{nct}")
def study(nct: str) -> JSONResponse:
    if not _NCT_RE.match(nct):
        return _err(400, "Invalid trial ID. Expected NCT followed by 8 digits.")
    from backend import db
    with _conn() as conn:
        data = db.get_study(conn, nct)
    return JSONResponse(data) if data is not None else _err(404, "Study not found")


@app.get("/api/summary/{nct}")
def summary(nct: str) -> JSONResponse:
    if not _NCT_RE.match(nct):
        return _err(400, "Invalid trial ID. Expected NCT followed by 8 digits.")
    from backend import db
    with _conn() as conn:
        data = db.get_summary(conn, nct)
    if data is not None:
        return JSONResponse(data)
    # TODO: on a miss, run live CPU generation + faithfulness check, store, return
    # (CLAUDE.md architecture: "summary: read SQLite.summaries; miss -> live CPU generation").
    return _err(404, "Summary not available yet")


# Mounted last so /api/* routes win. html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
