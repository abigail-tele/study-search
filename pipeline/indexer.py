"""Offline indexer: ClinicalTrials.gov v2 API -> SQLite.

Why offline: rule 4 says the live server never calls ClinicalTrials.gov. This
script runs by hand (or nightly), pulls open cancer trials for each cancer
type in data/cancer_types.json, and stores them (plus site coordinates) in
SQLite. A trial found by two types is stored once with two tag rows.

Status scope: config.INDEX_STATUSES (recruiting, not yet recruiting, active
not recruiting, enrolling by invitation). Finished/terminated trials cannot be
joined, so they would only confuse patients.

Rule 1 (cancer only) and tag accuracy. ClinicalTrials.gov matching is loose
("small cell" matches inside "non-small cell"; "cervical" matches the neck), so
we never trust the API's match. FINAL TAG RULE: a trial is kept only if it is
oncology (its conditions/keywords/titles match a type term or the general
cancer regex in config), and it gets a tag for every cancer type whose lay,
clinical or `match_patterns` terms appear in those fields as whole words, after
blanking out that type's `exclude_patterns` (e.g. "non-small cell" for SCLC).
If it is oncology but verifies for no type it is dropped (reason
no_type_match): there is NO fallback to the type CT.gov searched for, because
that produced B/T-cell lymphomas tagged Hodgkin. Non-oncology trials are never
stored. MDS (myelodysplastic syndrome) is tagged leukemia_aml, the type it is
usually grouped with in trials. Patterns live in data/cancer_types.json so a human can edit them.

`--retag` re-applies exactly these rules to the existing database without
refetching, and writes everything it removed to data/retag_dropped.csv.

Incremental runs (`--since`) fetch updated trials with NO status filter, so a
trial that closed is seen and deleted instead of staying "Recruiting" forever.

Usage:
    python -m pipeline.indexer [--types breast,lung_nsclc] [--since YYYY-MM-DD]
                               [--limit-pages N]
    python -m pipeline.indexer --retag
    python -m pipeline.indexer --fields-backfill interventions
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import re
import time
from typing import Any, Iterator

import httpx

from backend import config, db

log = logging.getLogger("indexer")


# --- parsing ---------------------------------------------------------------

def parse_study(study: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flatten one API study into (trial record, location rows)."""
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    cond = p.get("conditionsModule", {})
    desc = p.get("descriptionModule", {})
    elig = p.get("eligibilityModule", {})
    spons = p.get("sponsorCollaboratorsModule", {})
    rec = {
        "nct_id": ident["nctId"],
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "status": status.get("overallStatus"),
        "phases": design.get("phases") or [],
        "conditions": cond.get("conditions") or [],
        "keywords": cond.get("keywords") or [],
        "brief_summary": desc.get("briefSummary"),
        "detailed_description": desc.get("detailedDescription"),
        "eligibility_text": elig.get("eligibilityCriteria"),
        "min_age_years": db.parse_age_years(elig.get("minimumAge")),
        "max_age_years": db.parse_age_years(elig.get("maximumAge")),
        "sex": elig.get("sex"),
        "healthy_volunteers": (None if elig.get("healthyVolunteers") is None
                               else str(elig["healthyVolunteers"])),
        "sponsor": spons.get("leadSponsor", {}).get("name"),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "last_updated": status.get("lastUpdatePostDateStruct", {}).get("date"),
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
    }
    locs = []
    for loc in p.get("contactsLocationsModule", {}).get("locations", []) or []:
        geo = loc.get("geoPoint") or {}
        locs.append({
            "facility": loc.get("facility"), "city": loc.get("city"),
            "state": loc.get("state"), "zip": loc.get("zip"),
            "country": loc.get("country"), "status": loc.get("status"),
            "lat": geo.get("lat"), "lon": geo.get("lon"),
        })
    return rec, locs


def parse_interventions(study: dict[str, Any]) -> list[dict[str, str | None]]:
    """[{type, name}] from armsInterventionsModule (e.g. DRUG / 'Pembrolizumab'); [] if absent."""
    items = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions") or []
    return [{"type": i.get("type"), "name": i.get("name")} for i in items if i.get("name")]


# --- fetching --------------------------------------------------------------

def build_params(query_cond: str, since: str | None, page_token: str | None) -> dict[str, str]:
    """Query-string parameters for one page request."""
    params = {
        "query.cond": query_cond,
        "fields": config.CTGOV_FIELDS,
        "pageSize": str(config.CTGOV_PAGE_SIZE),
    }
    if since:
        # No status filter on incremental runs: we must SEE trials that closed
        # so index_type can delete them (otherwise they stay "Recruiting").
        params["filter.advanced"] = f"AREA[LastUpdatePostDate]RANGE[{since},MAX]"
    else:
        params["filter.overallStatus"] = ",".join(config.INDEX_STATUSES)
    if page_token:
        params["pageToken"] = page_token
    return params


def get_with_retry(client: httpx.Client, params: dict[str, str]) -> dict[str, Any]:
    """GET one page; retry 429/5xx/timeouts with exponential backoff."""
    last: Exception | None = None
    for attempt in range(config.CTGOV_MAX_RETRIES + 1):
        try:
            resp = client.get(config.CTGOV_BASE_URL, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {resp.status_code}",
                                            request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500 \
                    and e.response.status_code != 429:
                raise  # 4xx other than 429: our request is wrong, don't retry
            last = e
            wait = config.CTGOV_BACKOFF_BASE_SECONDS * 2 ** attempt
            log.warning("request failed (%s); retry %d in %.1fs", e, attempt + 1, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up after {config.CTGOV_MAX_RETRIES} retries") from last


def fetch_studies(client: httpx.Client, query_cond: str, since: str | None = None,
                  limit_pages: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield every study for a query, following nextPageToken."""
    token: str | None = None
    pages = 0
    while True:
        data = get_with_retry(client, build_params(query_cond, since, token))
        yield from data.get("studies", [])
        pages += 1
        token = data.get("nextPageToken")
        if not token or (limit_pages and pages >= limit_pages):
            return
        time.sleep(config.CTGOV_SLEEP_SECONDS)


# --- cancer-type verification (rule 1) -------------------------------------

def _term_regex(term: str) -> str:
    """Whole-word regex for a plain term.

    Spaces/hyphens are interchangeable, any word may carry a possessive
    ('s or a curly apostrophe), and the last word may be plural.
    """
    poss = r"(?:['’]?s)?"
    words = re.split(r"[\s-]+", term.strip())
    if len(words[-1]) > 4 and words[-1].endswith("s"):
        words[-1] = words[-1][:-1]  # "neoplasms" also matches "neoplasm"
    body = (poss + r"[\s-]+").join(re.escape(w) for w in words)
    return r"\b" + body + r"(?:['’]?s|es)?\b"


@functools.lru_cache(maxsize=None)
def _organ_patterns(organs: tuple) -> list[str]:
    """Variant wordings from a type's organ words plus a shared cancer noun.

    Why: one short word list per type replaces dozens of one-off patterns.
    Covers "breast cancer", "breast neoplasm female", "Cancer, Breast",
    "cancer of the breast" and "malignant neoplasm of breast". The organ word
    never counts alone ("biliary drainage"), is not part of a longer hyphenated
    word ("Hepato-Pancreato-Biliary") and not negated ("Non-Colorectal").
    """
    if not organs:
        return []
    organ = r"(?<![\w-])(?<!non )(?:" + "|".join(organs) + r")\b"
    noun = config.CANCER_NOUN_REGEX
    return [organ + r"\W+(?:\w+\W+){0,2}" + noun,
            noun + r"\W+(?:(?:of|in)\W+)?(?:the\W+)?" + organ]


@functools.lru_cache(maxsize=None)
def _compile(lay: tuple, clinical: tuple, match: tuple, ex: tuple,
             organs: tuple = ()) -> tuple[re.Pattern | None, re.Pattern | None]:
    """(term pattern, exclusion pattern) for one type, cached by its term lists."""
    parts = [_term_regex(t) for t in lay + clinical] + list(match) + _organ_patterns(organs)
    terms = re.compile("|".join(parts), re.I) if parts else None
    excl = re.compile("|".join(f"(?:{e})" for e in ex), re.I) if ex else None
    return terms, excl


def _compiled(ctype: dict[str, Any]) -> tuple[re.Pattern | None, re.Pattern | None]:
    return _compile(tuple(ctype.get("lay_terms", ())), tuple(ctype.get("clinical_terms", ())),
                    tuple(ctype.get("match_patterns", ())), tuple(ctype.get("exclude_patterns", ())),
                    tuple(ctype.get("organ_words", ())))


_ONC = re.compile(config.ONCOLOGY_REGEX, re.I)
_ONC_ACR = re.compile(config.ONCOLOGY_ACRONYMS_REGEX)  # case-sensitive on purpose


def trial_text(trial: dict[str, Any]) -> str:
    """The only fields we trust for classification: conditions, keywords, titles."""
    parts = list(trial.get("conditions") or []) + list(trial.get("keywords") or [])
    parts += [trial.get("brief_title") or "", trial.get("official_title") or ""]
    return " ; ".join(parts)


def _type_match(text: str, ctype: dict[str, Any]) -> tuple[bool, bool]:
    """(matched a term after exclusions, an exclusion pattern fired)."""
    terms, excl = _compiled(ctype)
    fired = bool(excl and excl.search(text))
    if excl:
        text = excl.sub(" ; ", text)  # blank excluded spans, e.g. "non-small cell"
    return bool(terms and terms.search(text)), fired


def verify_type_match(trial: dict[str, Any], ctype: dict[str, Any]) -> bool:
    """True if the trial's own text names this cancer type (not just CT.gov's match)."""
    return _type_match(trial_text(trial), ctype)[0]


def is_oncology(trial: dict[str, Any], types: list[dict[str, Any]] | None = None) -> bool:
    """General cancer check (rule 1): any type term, or the cancer regex/acronyms."""
    text = trial_text(trial)
    if _ONC.search(text) or _ONC_ACR.search(text):
        return True
    return any(_type_match(text, t)[0] for t in (types or []))


def assign_tags(trial: dict[str, Any], types: list[dict[str, Any]]) -> tuple[list[str], str | None]:
    """Decide a trial's cancer-type tags. Returns (tags, drop_reason).

    There is deliberately NO fallback to the type CT.gov's search matched: it
    mis-tagged B/T-cell lymphomas as Hodgkin and mouth cancers as cervical.
    An oncology trial naming no type is dropped; fix gaps with match_patterns.
    """
    text = trial_text(trial)
    results = {t["id"]: _type_match(text, t) for t in types}
    tags = [tid for tid, (hit, _) in results.items() if hit]
    if tags:
        return tags, None
    if not (_ONC.search(text) or _ONC_ACR.search(text)):
        return [], "non_oncology"
    return [], "no_type_match"


# --- orchestration ---------------------------------------------------------

def index_type(conn, client: httpx.Client, ctype: dict[str, Any], since: str | None = None,
               limit_pages: int | None = None,
               all_types: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Fetch and store all trials for one cancer type. Returns counts."""
    all_types = all_types or [ctype]
    db.ensure_interventions_table(conn)
    counts = {"fetched": 0, "new": 0, "updated": 0, "tags": 0, "locations": 0,
              "dropped": 0, "removed": 0}
    for study in fetch_studies(client, ctype["ctgov_query"], since, limit_pages):
        try:
            rec, locs = parse_study(study)
        except KeyError:
            continue  # study without an NCT id: nothing to cite
        counts["fetched"] += 1
        if rec["status"] not in config.INDEX_STATUSES:
            # only reachable on incremental runs (no status filter): it closed
            db.delete_trial(conn, rec["nct_id"])
            counts["removed"] += 1
            continue
        tags, reason = assign_tags(rec, all_types)
        if reason:
            db.delete_trial(conn, rec["nct_id"])  # may have been stored under older rules
            counts["dropped"] += 1
            continue
        counts["new" if db.upsert_trial(conn, rec) else "updated"] += 1
        counts["tags"] += db.add_cancer_tags(conn, rec["nct_id"], tags)
        counts["locations"] += db.replace_locations(conn, rec["nct_id"], locs)
        db.replace_interventions(conn, rec["nct_id"], parse_interventions(study))
    conn.commit()
    return counts


def _row_to_trial(row) -> dict[str, Any]:
    return {"conditions": json.loads(row["conditions"] or "[]"),
            "keywords": json.loads(row["keywords"] or "[]"),
            "brief_title": row["brief_title"], "official_title": row["official_title"]}


def retag(db_path=config.DB_PATH, types: list[dict[str, Any]] | None = None,
          dropped_csv=config.RETAG_DROPPED_PATH) -> dict[str, Any]:
    """Re-apply the tag + oncology rules to the stored trials, no refetching.

    Why: the rules improve over time (and early runs had none); rerunning the
    whole download takes an hour, rewriting tags takes seconds.
    """
    types = types if types is not None else load_types()
    conn = db.connect(db_path)
    db.init_schema(conn)
    count_sql = "SELECT cancer_type_id, COUNT(*) FROM trial_cancer_types GROUP BY 1"
    before = {r[0]: r[1] for r in conn.execute(count_sql)}
    n_before = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    old: dict[str, list[str]] = {}
    for r in conn.execute("SELECT nct_id, cancer_type_id FROM trial_cancer_types"):
        old.setdefault(r[0], []).append(r[1])
    dropped: list[tuple[str, str, str, str]] = []
    new_tags: list[tuple[str, str]] = []
    for row in conn.execute("SELECT * FROM trials").fetchall():
        trial = _row_to_trial(row)
        tags, reason = assign_tags(trial, types)
        if reason:
            dropped.append((row["nct_id"], reason, " | ".join(trial["conditions"]),
                            row["brief_title"] or ""))
        else:
            new_tags += [(row["nct_id"], t) for t in tags]
    for nct, *_ in dropped:
        db.delete_trial(conn, nct)
    conn.execute("DELETE FROM trial_cancer_types")
    conn.executemany("INSERT INTO trial_cancer_types VALUES (?,?)", new_tags)
    conn.commit()
    after = {r[0]: r[1] for r in conn.execute(count_sql)}
    n_after = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    conn.close()
    with open(dropped_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nct_id", "reason", "conditions", "brief_title"])
        w.writerows(dropped)
    for tid in sorted(set(before) | set(after)):
        log.info("%s: %d -> %d", tid, before.get(tid, 0), after.get(tid, 0))
    log.info("trials: %d -> %d (%d dropped, see %s)", n_before, n_after, len(dropped), dropped_csv)
    return {"before": before, "after": after, "trials_before": n_before,
            "trials_after": n_after, "dropped": len(dropped)}


def backfill_interventions(db_path=config.DB_PATH, client: httpx.Client | None = None,
                           batch: int = config.CTGOV_BACKFILL_BATCH) -> dict[str, int]:
    """Fetch ONLY NCTId + intervention fields for trials already stored (no re-download of everything).

    Why filter.ids: one small request per `batch` NCT ids instead of paging whole cancer queries.
    Each batch is written in its own short transaction so a running batch_summarize is not blocked,
    and trials that already have rows are skipped, so the job can be re-run to resume.
    """
    conn = db.connect(db_path)
    conn.execute("PRAGMA busy_timeout=30000")
    db.ensure_interventions_table(conn)
    done = {r[0] for r in conn.execute("SELECT DISTINCT nct_id FROM trial_interventions")}
    seen = {r[0].split(":", 1)[1] for r in conn.execute("SELECT key FROM meta WHERE key LIKE 'interv_checked:%'")}
    todo = [r[0] for r in conn.execute("SELECT nct_id FROM trials ORDER BY nct_id")
            if r[0] not in done and r[0] not in seen]
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": config.CTGOV_USER_AGENT},
                                    timeout=config.CTGOV_TIMEOUT_SECONDS)
    counts = {"requested": len(todo), "with_interventions": 0, "rows": 0}
    try:
        for i in range(0, len(todo), batch):
            ids = todo[i:i + batch]
            params = {"filter.ids": ",".join(ids), "fields": config.INTERVENTION_FIELDS,
                      "pageSize": str(max(batch, 100))}
            data = get_with_retry(client, params)
            got = set()
            for study in data.get("studies", []):
                nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                items = parse_interventions(study)
                if nct and items:
                    counts["rows"] += db.replace_interventions(conn, nct, items)
                    counts["with_interventions"] += 1
                    got.add(nct)
            # remember trials the API has no interventions for, so a re-run does not ask again
            conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                             [(f"interv_checked:{n}", "1") for n in ids if n not in got])
            conn.commit()
            log.info("interventions: %d/%d trials processed", min(i + batch, len(todo)), len(todo))
            time.sleep(config.CTGOV_SLEEP_SECONDS)
    finally:
        if own:
            client.close()
        conn.close()
    return counts


def load_types(only: list[str] | None = None) -> list[dict[str, Any]]:
    with open(config.CANCER_TYPES_PATH, encoding="utf-8") as f:
        types = json.load(f)
    if only:
        unknown = set(only) - {t["id"] for t in types}
        if unknown:
            raise SystemExit(f"unknown cancer type ids: {sorted(unknown)}")
        types = [t for t in types if t["id"] in only]
    return types


def run(types: list[dict[str, Any]], since: str | None = None, limit_pages: int | None = None,
        db_path=config.DB_PATH, client: httpx.Client | None = None,
        all_types: list[dict[str, Any]] | None = None) -> dict[str, dict[str, int]]:
    """Index the given types into the database; log counts per type."""
    conn = db.connect(db_path)
    db.init_schema(conn)
    own_client = client is None
    client = client or httpx.Client(headers={"User-Agent": config.CTGOV_USER_AGENT},
                                    timeout=config.CTGOV_TIMEOUT_SECONDS)
    results: dict[str, dict[str, int]] = {}
    try:
        for t in types:
            t0 = time.time()
            results[t["id"]] = index_type(conn, client, t, since, limit_pages,
                                           all_types or types)
            log.info("%s: %s in %.1fs", t["id"], results[t["id"]], time.time() - t0)
        if not limit_pages:  # a truncated run must not advance the incremental watermark
            db.mark_index_run(conn)
        conn.commit()
    finally:
        if own_client:
            client.close()
        conn.close()
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--types", help="comma-separated cancer type ids (default: all)")
    ap.add_argument("--since", help="only trials updated on/after YYYY-MM-DD")
    ap.add_argument("--auto-since", action="store_true",
                    help="use the date of the last full run stored in the meta table")
    ap.add_argument("--retag", action="store_true",
                    help="re-apply tag + oncology rules to the existing DB (no fetching)")
    ap.add_argument("--fields-backfill", choices=["interventions"],
                    help="fetch just one extra field set for trials already in the DB")
    ap.add_argument("--limit-pages", type=int, help="stop after N pages per type (testing)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per page is noise
    if args.retag:
        retag()
        return
    if args.fields_backfill == "interventions":
        t0 = time.time()
        log.info("backfill: %s in %.1fs", backfill_interventions(), time.time() - t0)
        return
    since = args.since
    if args.auto_since and not since:
        conn = db.connect()
        db.init_schema(conn)
        since = db.get_meta(conn, "last_index_run")
        conn.close()
    types = load_types(args.types.split(",") if args.types else None)
    t0 = time.time()
    results = run(types, since=since, limit_pages=args.limit_pages, all_types=load_types())
    log.info("done: %d types, %d studies fetched, %.1fs total", len(results),
             sum(r["fetched"] for r in results.values()), time.time() - t0)


if __name__ == "__main__":
    main()
