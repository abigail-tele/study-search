"""Evaluation report: retrieval, off-topic redirect, latency, summary pipeline.

Run with `python -m pipeline.evaluate`. It writes results/eval_retrieval.csv (long format),
results/eval_summary.json and rewrites the generated sections of docs/EVAL.md.

Why this design: two core functions are HUMAN-WRITTEN (reciprocal_rank_fusion and
detect_cancer_types). Anything that needs them is tried inside try/except
NotImplementedError and reported as "pending" until the humans finish, then computed
automatically on the next run. Nothing here invents numbers: every figure comes from
running the real search internals (backend.search) against data/trials.db.

Two kinds of retrieval metric are kept apart on purpose:
  * PROXY metrics need no human labels: a result "counts" if the trial is tagged with one
    of the query's expected cancer types. They show the ranking stays on the right cancer,
    NOT that it found the best trial.
  * TRUE metrics (hit@5, MRR, recall@10) need data/test_queries.json -> relevant_ncts, which
    humans fill in. They appear automatically once at least one query has labels.
The PDQ test split is never touched here (that is pipeline/baselines.py's job).
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
import time
from typing import Callable, Iterable

from backend import config, db, geo, search, taxonomy

PENDING_HUMAN = "pending: human-written function"


# ------------------------------------------------------------------ metric functions
def type_precision_at_k(ranked: list[str], tags: dict[str, set[str]],
                        expected: set[str], k: int) -> float:
    """Share of the top-k results tagged with an expected type (0.0 if nothing returned).

    PROXY metric: the denominator is the number of results actually returned (up to k), so a
    ranking that returns 3 on-type results out of 3 scores 1.0; empty lists score 0.0.
    """
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for n in top if tags.get(n, set()) & expected) / len(top)


def type_hit_at_k(ranked: list[str], tags: dict[str, set[str]],
                  expected: set[str], k: int) -> float:
    """1.0 if at least one of the top-k results has an expected type, else 0.0 (PROXY)."""
    return 1.0 if any(tags.get(n, set()) & expected for n in ranked[:k]) else 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """TRUE metric: 1.0 if any human-labelled relevant trial is in the top-k."""
    return 1.0 if any(n in relevant for n in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """TRUE metric: 1 / rank of the first relevant trial, 0.0 if none is ranked."""
    for pos, n in enumerate(ranked, start=1):
        if n in relevant:
            return 1.0 / pos
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """TRUE metric: share of the labelled relevant trials found in the top-k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0-100) of a non-empty list."""
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct / 100 * len(s) + 0.5)) - 1))
    return s[idx]


# ------------------------------------------------------------------ ranking helpers
def candidates(conn: sqlite3.Connection, types: list[str], zip_code: str | None = None) -> list[str]:
    """The same candidate set search() builds: tagged with types, default status, optional ZIP box."""
    origin = geo.lookup(zip_code) if zip_code else None
    return search.candidate_ids(conn, types, search.allowed_statuses(config.DEFAULT_STATUS),
                                None, origin, config.EVAL_LATENCY_RADIUS_MI if origin else None)


def rankings_for(conn: sqlite3.Connection, query: str, types: list[str]) -> dict[str, list[str] | None]:
    """keyword / meaning / hybrid top lists (hybrid is None while RRF is still a stub)."""
    cands = candidates(conn, types)
    kw = search.fts_ranking(conn, query, cands, config.FTS_TOP_K)
    mn = search.faiss_ranking(query, cands, config.FAISS_TOP_K)
    try:
        hybrid: list[str] | None = [n for n, _ in search.reciprocal_rank_fusion([kw, mn], config.RRF_K)]
    except NotImplementedError:
        hybrid = None
    return {"keyword": kw, "meaning": mn, "hybrid": hybrid}


def load_tags(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, set[str]]:
    """nct -> set of cancer type ids, for the trials we need to judge."""
    tags: dict[str, set[str]] = {}
    for n in set(ids):
        tags[n] = {r[0] for r in conn.execute(
            "SELECT cancer_type_id FROM trial_cancer_types WHERE nct_id=?", (n,))}
    return tags


# ------------------------------------------------------------------ 1a retrieval
def eval_retrieval(conn: sqlite3.Connection, queries: list[dict]) -> list[dict]:
    """Rows (system, metric, value, n, note) for proxy and (if labelled) true retrieval metrics."""
    per: dict[str, list[dict]] = {"keyword": [], "meaning": [], "hybrid": []}
    for q in queries:
        expected = set(q.get("expected_types") or [])
        r = rankings_for(conn, q["query"], sorted(expected))
        tags = load_tags(conn, [n for lst in r.values() if lst for n in lst[:config.EVAL_TOP_K]])
        relevant = set(q.get("relevant_ncts") or [])
        for system, ranked in r.items():
            if ranked is None:
                continue
            m = {"p": type_precision_at_k(ranked, tags, expected, config.EVAL_TOP_K),
                 "h": type_hit_at_k(ranked, tags, expected, config.EVAL_HIT_K),
                 "empty": 0 if ranked else 1}
            if relevant:
                m.update(th=hit_at_k(ranked, relevant, config.EVAL_HIT_K),
                         mrr=reciprocal_rank(ranked, relevant),
                         rec=recall_at_k(ranked, relevant, config.EVAL_TOP_K))
            per[system].append(m)
    rows: list[dict] = []
    for system, ms in per.items():
        if not ms:
            rows.append(_row("retrieval_proxy", system, "type_precision@10", "", 0, PENDING_HUMAN))
            rows.append(_row("retrieval_proxy", system, "type_hit@5", "", 0, PENDING_HUMAN))
            rows.append(_row("retrieval_true", system, "hit@5", "", 0, PENDING_HUMAN))
            continue
        n = len(ms)
        rows.append(_row("retrieval_proxy", system, "type_precision@10",
                         statistics.mean(m["p"] for m in ms), n, "PROXY"))
        rows.append(_row("retrieval_proxy", system, "type_hit@5",
                         statistics.mean(m["h"] for m in ms), n, "PROXY"))
        rows.append(_row("retrieval_proxy", system, "queries_with_no_results",
                         sum(m["empty"] for m in ms), n, "PROXY"))
        lab = [m for m in ms if "th" in m]
        if lab:
            rows.append(_row("retrieval_true", system, "hit@5", statistics.mean(m["th"] for m in lab), len(lab), "TRUE"))
            rows.append(_row("retrieval_true", system, "MRR", statistics.mean(m["mrr"] for m in lab), len(lab), "TRUE"))
            rows.append(_row("retrieval_true", system, "recall@10", statistics.mean(m["rec"] for m in lab), len(lab), "TRUE"))
        else:
            rows.append(_row("retrieval_true", system, "hit@5", "", 0, "pending: relevant_ncts not filled by humans"))
    return rows


def _row(section: str, system: str, metric: str, value, n: int, note: str = "") -> dict:
    v = round(value, 4) if isinstance(value, float) else value
    return {"section": section, "system": system, "metric": metric, "value": v, "n": n, "note": note}


# ------------------------------------------------------------------ 1b off-topic
def eval_off_topic(lay: list[dict], off: list[dict]) -> list[dict]:
    """Off-topic redirected x/10 and on-topic recall, or pending while detection is a stub."""
    try:
        taxonomy.detect_cancer_types("test")
    except NotImplementedError:
        return [_row("off_topic", "detect_cancer_types", "redirected", "", len(off), PENDING_HUMAN),
                _row("off_topic", "detect_cancer_types", "on_topic_recall", "", len(lay), PENDING_HUMAN)]
    redirected = sum(1 for q in off if not taxonomy.detect_cancer_types(q["query"]))
    found = [bool(taxonomy.detect_cancer_types(q["query"])) for q in lay]
    right = sum(1 for q in lay if set(taxonomy.detect_cancer_types(q["query"])) & set(q["expected_types"]))
    return [_row("off_topic", "detect_cancer_types", "redirected", redirected, len(off), f"{redirected}/{len(off)} off-topic queries returned no types"),
            _row("off_topic", "detect_cancer_types", "on_topic_recall", round(sum(found) / len(lay), 4), len(lay),
                 "share of lay queries where any type was detected"),
            _row("off_topic", "detect_cancer_types", "on_topic_expected_type_found", round(right / len(lay), 4), len(lay),
                 "share where at least one detected type is an expected type")]


# ------------------------------------------------------------------ 1c latency
def _time_ms(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def eval_latency(conn: sqlite3.Connection, queries: list[dict]) -> list[dict]:
    """p50/p95 milliseconds for keyword-only and meaning-only paths, warm, with and without ZIP.

    A timed run = candidate filter + one ranking + building the 10 result cards (what search()
    does minus RRF, which is pure arithmetic on 100 ids). One untimed pass warms the caches.
    """
    def path(q: dict, system: str, zip_code: str | None) -> None:
        types = sorted(q["expected_types"])
        cands = candidates(conn, types, zip_code)
        ranked = (search.fts_ranking(conn, q["query"], cands, config.FTS_TOP_K) if system == "keyword"
                  else search.faiss_ranking(q["query"], cands, config.FAISS_TOP_K))
        search.build_results(conn, [(n, 0.0) for n in ranked], zip_code if zip_code and geo.lookup(zip_code) else None,
                             config.EVAL_LATENCY_RADIUS_MI if zip_code else None)

    rows: list[dict] = []
    for system in ("keyword", "meaning"):
        for label, zip_code in (("no_zip", None), ("with_zip", config.EVAL_LATENCY_ZIP)):
            for q in queries:  # warm-up pass (also loads FAISS and MiniLM)
                path(q, system, zip_code)
            times = [_time_ms(lambda q=q: path(q, system, zip_code))
                     for q in queries for _ in range(config.EVAL_LATENCY_REPEATS)]
            rows.append(_row("latency_ms", f"{system}/{label}", "p50", round(statistics.median(times), 1), len(times)))
            rows.append(_row("latency_ms", f"{system}/{label}", "p95", round(percentile(times, 95), 1), len(times)))
    return rows


# ------------------------------------------------------------------ 1d/1e summaries
def read_batch_stats() -> dict | None:
    """data/batch_summary_stats.json (written by batch_summarize.py) or None if absent."""
    if not config.BATCH_SUMMARY_STATS_PATH.exists():
        return None
    return json.loads(config.BATCH_SUMMARY_STATS_PATH.read_text(encoding="utf-8"))


def read_model_comparison() -> list[dict]:
    """Held-out test rows already computed by pipeline/baselines.py (we only read them)."""
    path = config.ROOT_DIR / "results" / "baselines_results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("split") == "test"]


# ------------------------------------------------------------------ markdown rendering
def _fmt(v) -> str:
    if v in ("", None):
        return "pending"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def render_retrieval(rows: list[dict], n_queries: int) -> str:
    proxy = {(r["system"], r["metric"]): r for r in rows if r["section"] == "retrieval_proxy"}
    true = {(r["system"], r["metric"]): r for r in rows if r["section"] == "retrieval_true"}
    out = [
        f"Measured by `python -m pipeline.evaluate` on the {n_queries} lay queries in `data/test_queries.json`, "
        f"inside the same candidate set `search()` uses (tagged with the query's expected types, status "
        f"{config.DEFAULT_STATUS}, no age/ZIP filter). Keyword = FTS5 bm25 top {config.FTS_TOP_K}; meaning = FAISS/MiniLM "
        f"top {config.FAISS_TOP_K}; hybrid = reciprocal rank fusion of the two.",
        "",
        "**Proxy metrics (automatic, no human labels).** A result counts if its stored cancer tags include one of the "
        "query's expected types. This checks the ranking stays on the right cancer; it does NOT prove the best trial "
        "was found, so it is not hit@5. Because candidates are already restricted to trials tagged with the expected types "
        "(rule 1), these proxies are 1.0 whenever a ranking returns anything: they confirm the type filter works and that no "
        "query comes back empty, nothing more. Ranking quality is only measured by the true metrics below.",
        "",
        "| system | type precision@10 (proxy) | type hit@5 (proxy) | queries with no results |",
        "|---|---|---|---|"]
    for s in ("keyword", "meaning", "hybrid"):
        p, h, e = proxy.get((s, "type_precision@10")), proxy.get((s, "type_hit@5")), proxy.get((s, "queries_with_no_results"))
        if p and p["value"] != "":
            out.append(f"| {s} | {_fmt(p['value'])} | {_fmt(h['value'])} | {e['value']} / {e['n']} |")
        else:
            out.append(f"| {s} | {PENDING_HUMAN} | {PENDING_HUMAN} | - |" if s == "hybrid" else f"| {s} | pending | pending | - |")
    out += ["", "**True metrics (need human `relevant_ncts`).**", "",
            "| system | hit@5 | MRR | recall@10 | labelled queries |", "|---|---|---|---|---|"]
    for s in ("keyword", "meaning", "hybrid"):
        h = true.get((s, "hit@5"))
        if h and h["value"] != "":
            out.append(f"| {s} | {_fmt(h['value'])} | {_fmt(true[(s, 'MRR')]['value'])} | {_fmt(true[(s, 'recall@10')]['value'])} | {h['n']} |")
        else:
            why = PENDING_HUMAN if (h and h["note"] == PENDING_HUMAN) else "pending: relevant_ncts not filled in yet"
            out.append(f"| {s} | {why} | - | - | 0 |")
    return "\n".join(out)


def render_off_topic(rows: list[dict]) -> str:
    by = {r["metric"]: r for r in rows}
    red = by["redirected"]
    if red["value"] == "":
        return (f"Status: {PENDING_HUMAN} (`taxonomy.detect_cancer_types`). Once it exists, `python -m pipeline.evaluate` "
                f"reports how many of the {red['n']} off-topic queries are redirected and what share of the lay queries "
                "are recognised as cancer.")
    rec, exp = by["on_topic_recall"], by["on_topic_expected_type_found"]
    return (f"- Off-topic queries redirected (no cancer type detected, chips returned): **{red['value']}/{red['n']}**\n"
            f"- On-topic recall (lay queries where any cancer type was detected): **{rec['value']:.3f}** of {rec['n']}\n"
            f"- Lay queries where a detected type is one of the expected types: **{exp['value']:.3f}** of {exp['n']}")


def render_latency(rows: list[dict]) -> str:
    out = [f"Warm (one untimed pass first), {config.EVAL_LATENCY_REPEATS} timed runs per query, milliseconds. A run = candidate filter + "
           "one ranking + building the result cards; RRF is arithmetic on at most 100 ids and is not included. "
           f"With-ZIP rows use ZIP {config.EVAL_LATENCY_ZIP} and a {config.EVAL_LATENCY_RADIUS_MI}-mile radius. "
           "Measured on the developer PC (CPU search), so the host will differ.", "",
           "| path | p50 ms | p95 ms | runs |", "|---|---|---|---|"]
    systems = list(dict.fromkeys(r["system"] for r in rows))
    for s in systems:
        p50 = next(r for r in rows if r["system"] == s and r["metric"] == "p50")
        p95 = next(r for r in rows if r["system"] == s and r["metric"] == "p95")
        out.append(f"| {s} | {p50['value']} | {p95['value']} | {p50['n']} |")
    return "\n".join(out)


def render_summaries(test_rows: list[dict]) -> str:
    keep = [r for r in test_rows if r["system"] in ("copy-source", "zero-shot-instruct", "flan-t5-pdq-lr1e4-s70", "flan-t5-pdq-clean-sent")]
    if not keep:
        return "pending: results/baselines_results.csv not found."
    out = ["Copied from `results/baselines_results.csv` (held-out PDQ test split, produced once by `pipeline/baselines.py`; "
           "not recomputed here). Full model comparison is further down.", "",
           "| system | ROUGE-L F1 | key-term P / R | support | boilerplate % | FK out (src " + f"{float(keep[0]['fk_src']):.1f}) |",
           "|---|---|---|---|---|---|"]
    for r in keep:
        out.append(f"| {r['system']} | {float(r['rougeL_f']):.3f} | {float(r['keyterm_p']):.3f} / {float(r['keyterm_r']):.3f} | "
                   f"{float(r['support_precision']):.3f} | {float(r['boilerplate_pct']):.1f} | {float(r['fk_out']):.1f} |")
    return "\n".join(out)


def _flatten(prefix: str, obj, out: list[tuple[str, object]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out.append((prefix, obj))


def render_batch(stats: dict | None) -> str:
    if stats is None:
        return ("Status: pending (`data/batch_summary_stats.json` not found). `pipeline/batch_summarize.py` writes it; "
                "re-run `python -m pipeline.evaluate` afterwards to report pieces shown vs fallback by reason.")
    flat: list[tuple[str, object]] = []
    _flatten("", stats, flat)
    out = ["From `data/batch_summary_stats.json` (written by `pipeline/batch_summarize.py`): pieces whose model rewrite was shown "
           "versus replaced by the glossary-swapped original, with the reasons.", "", "| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in flat]
    return "\n".join(out)


# ------------------------------------------------------------------ docs/EVAL.md update
SECTIONS = [  # (existing heading prefix, new heading line, key)
    ("## Retrieval", "## Retrieval (keyword vs meaning vs hybrid)", "retrieval"),
    ("## Off-topic redirect", "## Off-topic redirect", "off_topic"),
    ("## Latency", "## Latency", "latency"),
    ("## Summaries (ROUGE-L", "## Summaries (ROUGE-L, reading level)", "summaries"),
    ("## Faithfulness pass rate", "## Faithfulness pass rate (batch summary pipeline)", "batch"),
]


def replace_section(doc: str, prefix: str, heading: str, body: str) -> str:
    """Replace the section starting with `prefix` (to the next '## ' heading); insert before the
    training section when it does not exist yet. Other sections are never touched."""
    block = f"{heading}\n\n{body.strip()}\n\n"
    m = re.search(rf"^{re.escape(prefix)}[^\n]*\n.*?(?=^## |\Z)", doc, flags=re.S | re.M)
    if m:
        return doc[:m.start()] + block + doc[m.end():]
    anchor = re.search(r"^## Training experiments", doc, flags=re.M)
    pos = anchor.start() if anchor else len(doc)
    return doc[:pos] + block + doc[pos:]


def update_doc(bodies: dict[str, str], path=config.EVAL_DOC_PATH) -> None:
    doc = path.read_text(encoding="utf-8")
    for prefix, heading, key in SECTIONS:
        doc = replace_section(doc, prefix, heading, bodies[key])
    path.write_text(doc, encoding="utf-8")


# ------------------------------------------------------------------ main
def main() -> None:
    data = json.loads(config.TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    lay, off = data["lay_queries"], data["off_topic"]
    conn = db.connect_ro(config.DB_PATH)
    try:
        retrieval = eval_retrieval(conn, lay)
        latency = eval_latency(conn, lay)
    finally:
        conn.close()
    off_rows = eval_off_topic(lay, off)
    stats = read_batch_stats()
    test_rows = read_model_comparison()
    all_rows = retrieval + off_rows + latency
    config.EVAL_RETRIEVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.EVAL_RETRIEVAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "system", "metric", "value", "n", "note"])
        w.writeheader()
        w.writerows(all_rows)
    config.EVAL_JSON.write_text(json.dumps({"rows": all_rows, "batch_summary_stats": stats}, indent=2), encoding="utf-8")
    update_doc({
        "retrieval": render_retrieval(retrieval, len(lay)),
        "off_topic": render_off_topic(off_rows),
        "latency": render_latency(latency),
        "summaries": render_summaries(test_rows),
        "batch": render_batch(stats),
    })
    for r in all_rows:
        print(f"{r['section']:16} {r['system']:22} {r['metric']:28} {r['value']!s:>8} n={r['n']} {r['note']}")


if __name__ == "__main__":
    main()
