"""Build (professional text -> plain-language text) training pairs from NCI PDQ.

Why this exists: the National Cancer Institute (NCI) writes every PDQ cancer
treatment summary TWICE -- a Health Professional version and a Patient version
that say the same things in simpler words. Both are public domain (credit: NCI,
https://www.cancer.gov). That gives us free, expert-written "before/after"
examples to fine-tune our small summarizer, with no chatbot-generated data.

Pipeline (each step logs kept/dropped counts into data/pdq_stats.json):
  1. fetch    download both pages per cancer type (cached, polite, retried)
  2. parse    cut each page into top-level (h2) sections of clean paragraphs
  3. match    pair professional sections with patient sections by heading
  4. align    pair paragraphs inside matched sections by embedding similarity
  5. filter   drop too-short, too-long-answer, and copied-verbatim pairs
  6. split    train/val/test 80/10/10 *by organ family* (colon+rectal together, ...)
              so no page AND no sibling page leaks across splits, then val/test
              pairs whose target nearly duplicates a train target are dropped

Heading matching rule (step 3): we compare top-level (h2) headings only,
because patient pages are organised differently below that level. Headings are
normalised (lower-case, no brackets, no filler words, "stages" -> "stage") and
scored with difflib's similarity ratio. Stage numerals are a hard gate:
"Treatment of Stage I" may only match a patient heading that also names stage I
(e.g. "Treatment of Stages I and II"), never "Stage Information". Matching is
many-to-one (several professional sections may feed one patient section) and a
pair needs a score >= PDQ_HEADING_MATCH_THRESHOLD.

Loader contract for train.py: every record carries its `similarity` score, and
the file keeps ALL candidates at or above PDQ_ALIGN_SIM_THRESHOLD (0.60), so
train.py may keep only `similarity >= X` to choose a stricter threshold at
training time without regenerating pairs.
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import httpx
from bs4 import BeautifulSoup

from backend import config as cfg

log = logging.getLogger("pdq_pairs")

Encoder = Callable[[list[str]], "np.ndarray"]  # noqa: F821  (numpy imported lazily)

# --- Text cleaning ----------------------------------------------------------
# Citation markers like [1], [3,4], [5-7], "[ 12 ]" and evidence tags such as
# "[Level of evidence: 1iiA]". They are noise for a plain-language model.
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]")
_LEVEL_RE = re.compile(r"\[\s*Levels? of evidence[^\]]*\]", re.I)
_EMPTY_BRACKETS_RE = re.compile(r"\[[\s,;–\-]*\]")
_ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
_STOP = {"of", "for", "about", "the", "and", "to", "in", "a", "an", "also", "called", "information"}


def strip_citations(text: str) -> str:
    """Remove [1,2]-style citation numbers and [Level of evidence ...] tags.

    Why: reference numbers and evidence grades mean nothing to patients, and a
    model that learns to copy them would invent fake citations.
    """
    text = _LEVEL_RE.sub(" ", text)
    text = _CITATION_RE.sub("", text)
    # Citation links are deleted from the HTML first, which leaves empty
    # brackets like "[ ]; [, ]" behind -- sweep those and their stray ';'.
    text = _EMPTY_BRACKETS_RE.sub("", text)
    text = re.sub(r"(?:\s*;\s*)+(?=\s*[.,]|\s*$)", "", text)
    text = re.sub(r"([.!?])\s*;", r"\g<1>", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)  # tidy "word ." left by removals
    return re.sub(r"\s+", " ", text).strip()


# --- Data holders -----------------------------------------------------------
@dataclass
class Section:
    """One top-level (h2) section of a PDQ page: heading + clean paragraphs."""

    heading: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class PageSpec:
    """One cancer type: where to find its two PDQ pages."""

    cancer_type: str
    hp_url: str
    patient_url: str
    hp_fetch_url: str | None = None       # optional mirror if the live URL is broken
    patient_fetch_url: str | None = None
    family: str = ""                      # organ family used for splitting (default: own family)


# --- Step 1: fetch ----------------------------------------------------------
def cache_path(url: str, cache_dir: Path) -> Path:
    """Stable file name for a URL so re-runs never re-download."""
    slug = re.sub(r"[^a-z0-9]+", "_", url.lower().split("://", 1)[-1]).strip("_")
    return cache_dir / f"{slug[-150:]}.html"


def http_get(url: str) -> str:
    """The ONLY function that touches the network (mock this in tests).

    Uses a browser-like User-Agent and a redirect cap so a redirect loop (seen
    on one cancer.gov page) fails fast instead of hanging.
    """
    with httpx.Client(
        headers={"User-Agent": cfg.PDQ_USER_AGENT},
        timeout=cfg.PDQ_TIMEOUT_SECONDS,
        follow_redirects=True,
        max_redirects=5,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def fetch_with_retry(
    url: str, getter: Callable[[str], str] = http_get, sleep: Callable[[float], None] = time.sleep
) -> str:
    """Call `getter` with exponential backoff (1s, 2s, 4s ...) on failure."""
    last: Exception | None = None
    for attempt in range(cfg.PDQ_MAX_RETRIES):
        try:
            return getter(url)
        except Exception as exc:  # noqa: BLE001 - any network failure is retryable
            last = exc
            sleep(cfg.PDQ_SLEEP_SECONDS * (2**attempt))
    raise RuntimeError(f"giving up on {url}: {last}")


def load_html(
    url: str,
    fetch_url: str | None,
    cache_dir: Path,
    fetch: bool = True,
    getter: Callable[[str], str] = http_get,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Return page HTML from the cache, downloading (politely) if allowed."""
    path = cache_path(url, cache_dir)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if not fetch:
        raise FileNotFoundError(f"not cached and --no-fetch set: {url}")
    html = fetch_with_retry(fetch_url or url, getter, sleep)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    sleep(cfg.PDQ_SLEEP_SECONDS)  # be polite between downloads
    return html


# --- Step 2: parse ----------------------------------------------------------
def _clean_paragraph(text: str) -> str:
    return strip_citations(re.sub(r"\s+", " ", text)).strip()


def is_skipped_heading(heading: str) -> bool:
    """True for admin/boilerplate headings (References, Disclaimer, ...)."""
    h = re.sub(r"\s*\(\d{2}/\d{2}/\d{4}\)", "", heading.lower()).strip()
    return any(h.startswith(s) for s in cfg.PDQ_SKIP_HEADINGS)


def parse_sections(html: str, is_patient: bool, stats: dict | None = None) -> list[Section]:
    """Cut a PDQ page into top-level sections of clean paragraphs.

    We walk headings and <p> tags in document order. h2 starts a section; h3/h4
    only matter for skipping boilerplate subsections (e.g. "Current Clinical
    Trials"). On patient pages, the "keypoint" sub-headings are themselves
    plain-language sentences, so we keep them as paragraphs.
    """
    stats = stats if stats is not None else {}
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "figure", "table", "figcaption"]):
        tag.decompose()
    for a in soup.select('a[href^="#cit"]'):
        a.decompose()
    root = soup.find("main") or soup.body or soup

    sections: list[Section] = []
    current: Section | None = None
    skip_level: int | None = None  # heading level (2-4) whose content we skip
    empty_level: int | None = None  # newer pages put the heading text in a <p> that
    #                                 the HTML parser pushes out of an empty <h2>
    for el in root.find_all(["h2", "h3", "h4", "p"]):
        text = _clean_paragraph(el.get_text(" ", strip=True))
        if el.name == "p" and empty_level is not None:
            level, empty_level = empty_level, None  # this <p> is really the heading
        elif el.name == "p":
            if current is not None and skip_level is None:
                _add_paragraph(current, text, stats)
            continue
        else:
            level = int(el.name[1])
            if not text:
                empty_level = level
                continue
        if skip_level is not None and level <= skip_level:
            skip_level = None
        if skip_level is None and is_skipped_heading(text):
            skip_level = level
            if level == 2:
                current = None
        elif skip_level is None and level == 2:
            current = Section(heading=text)
            sections.append(current)
        elif skip_level is None and is_patient and el.get("type") == "keypoint" and current is not None:
            _add_paragraph(current, text, stats)
    return [s for s in sections if s.paragraphs]


def _add_paragraph(section: Section, text: str, stats: dict) -> None:
    """Keep a paragraph only if it is a real standalone block of prose."""
    stats["paragraphs_seen"] = stats.get("paragraphs_seen", 0) + 1
    words = len(text.split())
    if re.match(r"(?i)^levels? of evidence", text) or not text:
        reason = "dropped_evidence_or_empty"
    elif words < cfg.PDQ_MIN_WORDS:
        reason = "dropped_too_short"
    elif text.endswith(":"):  # a lead-in to a bullet list; incomplete without it
        reason = "dropped_list_leadin"
    else:
        section.paragraphs.append(text)
        stats["paragraphs_kept"] = stats.get("paragraphs_kept", 0) + 1
        return
    stats[reason] = stats.get(reason, 0) + 1


# --- Step 3: heading matching -----------------------------------------------
def normalize_heading(heading: str) -> str:
    """Lower-case, drop brackets/filler words, 'stages' -> 'stage'."""
    h = re.sub(r"\([^)]*\)", " ", heading.lower())
    h = re.sub(r"[^a-z0-9 ]+", " ", h).replace("stages", "stage")
    return " ".join(t for t in h.split() if t not in _STOP or t in _ROMAN)


def _stages(heading: str) -> set[int]:
    """Stage numerals named in a heading: 'Stages I and II' -> {1, 2}."""
    toks = re.sub(r"\([^)]*\)", " ", heading.lower())
    toks = re.sub(r"[^a-z0-9 ]+", " ", toks).replace("stages", "stage").split()
    if "stage" not in toks:
        return set()
    out: set[int] = set()
    for tok in toks[toks.index("stage") + 1 :]:
        if tok in _ROMAN:
            out.add(_ROMAN[tok])
        elif tok in {"and", "or"}:
            continue
        else:
            break
    return out


def heading_score(hp_heading: str, patient_heading: str) -> float:
    """Similarity 0-1 between two headings (see module docstring for the rule)."""
    hp_stage, pt_stage = _stages(hp_heading), _stages(patient_heading)
    if bool(hp_stage) != bool(pt_stage):
        return 0.0
    if hp_stage and not hp_stage <= pt_stage:
        return 0.0
    a, b = normalize_heading(hp_heading), normalize_heading(patient_heading)
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_sections(
    hp: list[Section], patient: list[Section], threshold: float | None = None
) -> list[tuple[Section, list[Section], float]]:
    """For each patient section, list the professional sections that feed it.

    Each professional section goes to its single best-scoring patient section
    (if the score clears the threshold); a patient section can receive several.
    """
    threshold = cfg.PDQ_HEADING_MATCH_THRESHOLD if threshold is None else threshold
    feeds: dict[int, list[Section]] = {}
    scores: dict[int, list[float]] = {}
    for h in hp:
        best, best_i = 0.0, -1
        for i, p in enumerate(patient):
            s = heading_score(h.heading, p.heading)
            if s > best:
                best, best_i = s, i
        if best_i >= 0 and best >= threshold:
            feeds.setdefault(best_i, []).append(h)
            scores.setdefault(best_i, []).append(best)
    return [
        (patient[i], secs, sum(scores[i]) / len(scores[i])) for i, secs in sorted(feeds.items())
    ]


# --- Step 4: embedding alignment --------------------------------------------
def default_encoder() -> Encoder:
    """all-MiniLM-L6-v2 (same model as search) returning unit-length vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.EMBED_MODEL_NAME)
    return cached_encoder(
        lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        cfg.PDQ_RAW_DIR / "embed_cache.pkl",
    )


def cached_encoder(encode: Encoder, path: Path) -> Encoder:
    """Wrap an encoder with an on-disk text->vector cache.

    Why: joining 1-3 paragraphs triples the texts to embed, and the same text is
    re-embedded on every re-run or threshold experiment. Caching makes re-runs
    near-instant. The cache lives next to the raw HTML (git-ignored).
    """
    import pickle

    import numpy as np

    cache: dict[str, "np.ndarray"] = {}
    if path.exists():
        cache = pickle.loads(path.read_bytes())

    def encode_cached(texts: list[str]) -> "np.ndarray":
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            for t, v in zip(missing, encode(missing)):
                cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([cache[t] for t in texts])

    def save() -> None:  # called once per page by run(), not per call (cache can be tens of MB)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(cache))

    encode_cached.save = save  # type: ignore[attr-defined]
    return encode_cached


def greedy_align(sim, threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one greedy matching on a similarity matrix (rows=source, cols=target).

    Repeatedly take the globally highest remaining cell >= threshold, then
    retire its row and column. Simple, deterministic, and easy to explain.
    """
    import numpy as np

    arr = np.asarray(sim, dtype=float)
    rows, cols = np.nonzero(arr >= threshold)  # only cells that could ever qualify
    order = np.argsort(-arr[rows, cols], kind="stable")
    cells = [(float(arr[rows[k], cols[k]]), int(rows[k]), int(cols[k])) for k in order]
    used_r: set[int] = set()
    used_c: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for s, i, j in cells:
        if s < threshold:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j, s))
    return sorted(pairs)


SrcItem = tuple[str, int, int]  # (paragraph text, section group id, position in section)


def make_windows(items: list[SrcItem], max_window: int) -> list[tuple[tuple[int, ...], str]]:
    """All runs of 1..max_window ADJACENT professional paragraphs, as (indices, joined text).

    Why: one patient paragraph often condenses two or three consecutive
    professional paragraphs. "Adjacent" means same section and neighbouring
    positions (a leftover gap breaks the run), so we never glue unrelated text.
    Runs longer than PDQ_MAX_SOURCE_WORDS are skipped (the model's input limit).
    """
    out: list[tuple[tuple[int, ...], str]] = []
    for start in range(len(items)):
        idxs = [start]
        while True:
            text = " ".join(items[i][0] for i in idxs)
            if len(text.split()) <= cfg.PDQ_MAX_SOURCE_WORDS or len(idxs) == 1:
                out.append((tuple(idxs), text))
            nxt = idxs[-1] + 1
            if len(idxs) >= max_window or nxt >= len(items):
                break
            same = items[nxt][1] == items[idxs[-1]][1] and items[nxt][2] == items[idxs[-1]][2] + 1
            if not same:
                break
            idxs.append(nxt)
    return out


def align_windows(
    items: list[SrcItem], targets: list[str], encode: Encoder, threshold: float,
    max_window: int | None = None,
) -> tuple[list[tuple[tuple[int, ...], int, float]], int]:
    """Greedy one-to-one alignment where a source may be 1-3 adjacent paragraphs.

    Candidates are ranked by cosine similarity; a joined window is penalised by
    PDQ_WINDOW_MIN_GAIN so it only wins when it clearly beats the single
    paragraphs inside it. Joined windows whose text is too short for the target
    (target > PDQ_MAX_LEN_RATIO x source) are skipped so a different candidate
    can win instead of the pair being dropped later. A candidate is rejected if
    ANY of its paragraphs or its target is already used.
    Returns (matches, number of patient paragraphs left unmatched).
    """
    import numpy as np

    wins = make_windows(items, max_window or cfg.PDQ_MAX_WINDOW)
    sim = np.asarray(encode([t for _, t in wins])) @ np.asarray(encode(targets)).T
    sw = np.array([len(t.split()) for _, t in wins])
    tw = np.array([len(t.split()) for t in targets])
    multi = np.array([len(ix) > 1 for ix, _ in wins])
    too_short = tw[None, :] > cfg.PDQ_MAX_LEN_RATIO * sw[:, None]
    rank = sim - cfg.PDQ_WINDOW_MIN_GAIN * multi[:, None]
    ok = (sim >= threshold) & ~(too_short & multi[:, None])
    rows, cols = np.nonzero(ok)
    order = np.argsort(-rank[rows, cols], kind="stable")
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    matches: list[tuple[tuple[int, ...], int, float]] = []
    for k in order:
        idxs, j = wins[rows[k]][0], int(cols[k])
        if j in used_tgt or used_src.intersection(idxs):
            continue
        used_src.update(idxs)
        used_tgt.add(j)
        matches.append((idxs, j, float(sim[rows[k], j])))
    return sorted(matches), len(targets) - len(matches)


# --- Step 5: pair filters ---------------------------------------------------
_XREF_RES = [re.compile(p, re.I) for p in cfg.PDQ_XREF_PATTERNS]
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_STAGE_RE = re.compile(
    r"\bstages?\b((?:\s*(?:,|and\b|or\b|to\b|through\b|-|\u2013)?\s*(?:0|iv|i{1,3}|[1-4])[abc]?\b)+)", re.I
)


def strip_xref_sentences(text: str) -> tuple[str, int]:
    """Remove sentences that point the reader elsewhere ("See the Treatment section").

    Why: a target that says "learn about X in the booklet" is information the
    source does not contain; training on it teaches the model to invent
    pointers and facts (hard rule 3). Returns (clean text, sentences removed).
    """
    sents = _SENT_SPLIT_RE.split(text)
    keep = [x for x in sents if not any(r.search(x) for r in _XREF_RES)]
    return " ".join(keep), len(sents) - len(keep)


def stage_numerals(text: str) -> set[int]:
    """Every stage numeral named in text: 'stages I, II and 3' -> {1, 2, 3}."""
    out: set[int] = set()
    for m in _STAGE_RE.finditer(text):
        for tok in re.findall(r"\b(0|iv|i{1,3}|[1-4])[abc]?\b", m.group(1), re.I):
            out.add(_ROMAN[tok.lower()])
    return out


def numbers_in(text: str) -> set[str]:
    """Digit strings in text with thousands commas removed ('1,000' -> '1000')."""
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text)}


def unsupported_numbers(source: str, target: str) -> set[str]:
    """Numbers / stage numerals in the target that the source never mentions.

    Stage numerals are compared as integers (so 'stage III' in the source
    supports 'stage 3'); they are removed from the digit check so they are not
    counted twice.
    """
    bad = {f"stage {n}" for n in stage_numerals(target) - stage_numerals(source)}
    plain_target = _STAGE_RE.sub(" ", target)
    return bad | (numbers_in(plain_target) - numbers_in(source))


def pair_drop_reason(source: str, target: str, sim: float, min_words: int | None = None) -> str | None:
    """Why a candidate pair must be dropped, or None to keep it."""
    min_words = cfg.PDQ_MIN_WORDS if min_words is None else min_words
    sw, tw = len(source.split()), len(target.split())
    if sw < min_words or tw < min_words:
        return "too_short"
    if sw > cfg.PDQ_MAX_SOURCE_WORDS:
        return "source_too_long"
    if tw > cfg.PDQ_MAX_LEN_RATIO * sw:  # plain text must not balloon
        return "length_ratio"
    if sim > cfg.PDQ_MAX_ALIGN_SIM:
        return "near_identical"
    if unsupported_numbers(source, target):  # the target would add a fact the source lacks
        return "number_not_in_source"
    return None


def drop_near_duplicates(records: list[dict], ratio: float | None = None) -> dict[str, int]:
    """Remove val/test records whose target nearly copies a TRAIN target (in place).

    Why: sibling pages (colon vs rectal, adult vs child) reuse whole paragraphs,
    so even a family-based split leaks. difflib's ratio is the same measure the
    reviewer used; cheap upper bounds (real_quick_ratio, quick_ratio) skip most
    comparisons. Returns {"val": n, "test": n} counts of dropped records.
    """
    ratio = cfg.PDQ_NEAR_DUP_RATIO if ratio is None else ratio
    train = list(dict.fromkeys(r["target"].lower() for r in records if r["split"] == "train"))
    dropped = {"val": 0, "test": 0}
    keep: list[dict] = []
    sm = difflib.SequenceMatcher(None)
    for r in records:
        if r["split"] != "train":
            sm.set_seq2(r["target"].lower())
            dup = False
            for t in train:
                sm.set_seq1(t)
                if sm.real_quick_ratio() >= ratio and sm.quick_ratio() >= ratio and sm.ratio() >= ratio:
                    dup = True
                    break
            if dup:
                dropped[r["split"]] += 1
                continue
        keep.append(r)
    records[:] = keep
    return dropped


# --- Step 6: split by page --------------------------------------------------
def assign_splits(page_counts: dict[str, int]) -> dict[str, str]:
    """Assign whole groups (organ families) to train/val/test, close to 80/10/10.

    Why by group: sections of one page share vocabulary, and sibling pages
    (colon/rectal, NSCLC/SCLC, adult/child) share whole paragraphs, so splitting
    inside a family would let test examples leak into training. run() passes
    FAMILY names here; the keys are just opaque group names. With only a handful of
    pages we cannot hit 80/10/10 exactly, so we try every assignment of pages to
    splits (3^N, tiny for N<=12), require each split non-empty, and pick the one
    whose pair-count fractions are closest to PDQ_SPLIT (ties broken by seeded
    shuffle for reproducibility). With more pages the fit gets better.
    """
    pages = sorted(page_counts)
    random.Random(cfg.RANDOM_SEED).shuffle(pages)
    if len(pages) < 3:
        return {p: "train" for p in pages}
    if len(pages) <= _MAX_EXHAUSTIVE_PAGES:
        return _assign_exhaustive(pages, page_counts)
    return _assign_greedy(pages, page_counts)


_MAX_EXHAUSTIVE_PAGES = 12  # 3^12 = 531k assignments is still instant
_SPLIT_NAMES = ("train", "val", "test")


def _assign_exhaustive(pages: list[str], page_counts: dict[str, int]) -> dict[str, str]:
    """Try every page->split assignment; keep the closest fit to PDQ_SPLIT."""
    total = sum(page_counts.values()) or 1
    best: tuple[float, tuple[int, ...]] | None = None
    for combo in itertools.product(range(3), repeat=len(pages)):
        if len(set(combo)) < 3:
            continue
        got = [0, 0, 0]
        for p, c in zip(pages, combo):
            got[c] += page_counts[p]
        err = sum(abs(g / total - t) for g, t in zip(got, cfg.PDQ_SPLIT))
        if best is None or err < best[0] - 1e-12:
            best = (err, combo)
    assert best is not None
    return {p: _SPLIT_NAMES[c] for p, c in zip(pages, best[1])}


def _assign_greedy(pages: list[str], page_counts: dict[str, int]) -> dict[str, str]:
    """Many pages: place the biggest pages first into the split furthest below target.

    Like filling three buckets to their target sizes, largest items first. Fast
    (no 3^N search) and good once there are many pages. A final pass guarantees
    val and test are never empty.
    """
    total = sum(page_counts.values()) or 1
    got = [0, 0, 0]
    out: dict[str, str] = {}
    for p in sorted(pages, key=lambda x: -page_counts[x]):
        deficit = [cfg.PDQ_SPLIT[k] * total - got[k] for k in range(3)]
        k = deficit.index(max(deficit))
        got[k] += page_counts[p]
        out[p] = _SPLIT_NAMES[k]
    for name in ("val", "test"):  # safety net: move the smallest train page if empty
        if name not in out.values():
            smallest = min((p for p, v in out.items() if v == "train"), key=lambda x: page_counts[x])
            out[smallest] = name
    return out


# --- Orchestration -----------------------------------------------------------
def load_pages(path: Path = cfg.PDQ_PAGES_PATH, only: Iterable[str] | None = None) -> list[PageSpec]:
    """Read the page list (data/pdq_pages.json); optionally keep only some types."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    keep = set(only) if only else None
    return [
        PageSpec(
            r["cancer_type"], r["hp_url"], r["patient_url"],
            r.get("hp_fetch_url"), r.get("patient_fetch_url"), r.get("family") or r["cancer_type"],
        )
        for r in raw
        if keep is None or r["cancer_type"] in keep
    ]


def _emit(
    spec: PageSpec, src: str, tgt: str, sim: float, section: str, how: str, n_src: int, stats: dict
) -> dict | None:
    """Apply the pair filters; return a record (or None and count the drop reason)."""
    tgt, n_xref = strip_xref_sentences(tgt)
    if n_xref:
        stats["xref_sentences_stripped"] += n_xref
        stats["pairs_with_xref_stripped"] += 1
    why = pair_drop_reason(src, tgt, sim)
    if why == "too_short" and n_xref:
        why = "xref_too_short"  # stripping cross-references left too little text
    if why:
        stats["pairs_dropped"][why] = stats["pairs_dropped"].get(why, 0) + 1
        return None
    stats["pairs_kept_" + how] += 1
    if n_src > 1:
        stats["pairs_kept_multi_source"] += 1
    return {
        "source": src, "target": tgt, "cancer_type": spec.cancer_type,
        "hp_url": spec.hp_url, "patient_url": spec.patient_url,
        "section": section, "similarity": round(sim, 4), "source_paragraphs": n_src,
    }


def _align_and_emit(
    spec: PageSpec, items: list[SrcItem], targets: list[tuple[str, str]], encode: Encoder,
    threshold: float, how: str, stats: dict,
) -> tuple[list[dict], set[tuple[int, int]], set[str]]:
    """Align one pool; return (records, used (group,pos) keys, used target texts)."""
    matches, _ = align_windows(items, [t for _, t in targets], encode, threshold)
    recs: list[dict] = []
    used_src = {(items[i][1], items[i][2]) for idxs, _, _ in matches for i in idxs}
    used_tgt = {targets[j][1] for _, j, _ in matches}
    for idxs, j, sim in matches:
        src = " ".join(items[i][0] for i in idxs)
        rec = _emit(spec, src, targets[j][1], sim, targets[j][0], how, len(idxs), stats)
        recs.extend([rec] if rec else [])
    key = "pairs_below_similarity" if how == "heading" else "pairs_below_similarity_fallback"
    stats[key] += len(targets) - len(matches)
    return recs, used_src, used_tgt


def build_page_pairs(
    spec: PageSpec, hp_html: str, pt_html: str, encode: Encoder, stats: dict,
    threshold: float | None = None,
) -> list[dict]:
    """Steps 2-5 for one cancer type; returns pair records (no split yet).

    Pass 1 (heading evidence): align paragraphs only inside heading-matched
    sections, accepting similarity >= `threshold` (default PDQ_ALIGN_SIM_THRESHOLD).
    Pass 2 (page-wide second chance): newer patient pages are organised by
    treatment type, so many sections have no heading twin. Leftover patient
    paragraphs are aligned against leftover professional paragraphs anywhere on
    the same page, but with the stricter PDQ_FALLBACK_SIM_THRESHOLD because we
    no longer have heading agreement as supporting evidence.
    In both passes a source may be 1-3 adjacent professional paragraphs joined
    (many-to-one), see align_windows.
    """
    threshold = cfg.PDQ_ALIGN_SIM_THRESHOLD if threshold is None else threshold
    hp_stats: dict = {}
    pt_stats: dict = {}
    hp = parse_sections(hp_html, False, hp_stats)
    pt = parse_sections(pt_html, True, pt_stats)
    matches = match_sections(hp, pt)
    stats.update(
        hp_sections=len(hp), patient_sections=len(pt), sections_matched=len(matches),
        hp_paragraph_stats=hp_stats, patient_paragraph_stats=pt_stats,
        pairs_below_similarity=0, pairs_below_similarity_fallback=0, pairs_dropped={},
        xref_sentences_stripped=0, pairs_with_xref_stripped=0,
        pairs_kept=0, pairs_kept_heading=0, pairs_kept_fallback=0, pairs_kept_multi_source=0,
    )
    records: list[dict] = []
    used_src: set[tuple[int, int]] = set()
    used_tgt: set[str] = set()
    group_of = {id(sec): g for g, sec in enumerate(hp)}
    for pt_sec, hp_secs, _score in matches:
        items = [(p, group_of[id(s)], k) for s in hp_secs for k, p in enumerate(s.paragraphs)]
        recs, us, ut = _align_and_emit(
            spec, items, [(pt_sec.heading, p) for p in pt_sec.paragraphs], encode, threshold, "heading", stats
        )
        records += recs
        used_src |= us
        used_tgt |= ut
    # pass 2: page-wide second chance for everything left over
    left_items = [
        (p, g, k) for g, s in enumerate(hp) for k, p in enumerate(s.paragraphs) if (g, k) not in used_src
    ]
    left_tgt = [(s.heading, p) for s in pt for p in s.paragraphs if p not in used_tgt]
    if left_items and left_tgt:
        recs, _, _ = _align_and_emit(
            spec, left_items, left_tgt, encode, cfg.PDQ_FALLBACK_SIM_THRESHOLD, "fallback", stats
        )
        records += recs
    stats["pairs_kept"] = len(records)
    stats["patient_paragraphs_total"] = sum(len(s.paragraphs) for s in pt)
    return records


def run(
    types: list[str] | None = None,
    fetch: bool = True,
    encode: Encoder | None = None,
    pages_path: Path = cfg.PDQ_PAGES_PATH,
    out_path: Path = cfg.PDQ_PAIRS_PATH,
    stats_path: Path = cfg.PDQ_STATS_PATH,
    cache_dir: Path = cfg.PDQ_RAW_DIR,
    getter: Callable[[str], str] = http_get,
    sleep: Callable[[float], None] = time.sleep,
    sim_threshold: float | None = None,
    borderline_path: Path = cfg.PDQ_BORDERLINE_PATH,
) -> dict:
    """Run the whole pipeline, write the JSONL + stats, and return the stats."""
    specs = load_pages(pages_path, types)
    stats: dict = {"pages": {}, "failed_pages": {}}
    per_page: dict[str, list[dict]] = {}
    for spec in specs:
        try:
            hp_html = load_html(spec.hp_url, spec.hp_fetch_url, cache_dir, fetch, getter, sleep)
            pt_html = load_html(spec.patient_url, spec.patient_fetch_url, cache_dir, fetch, getter, sleep)
        except Exception as exc:  # noqa: BLE001 - record and continue with other pages
            log.warning("page %s failed: %s", spec.cancer_type, exc)
            stats["failed_pages"][spec.cancer_type] = str(exc)
            continue
        if encode is None:
            encode = default_encoder()
        page_stats: dict = {}
        per_page[spec.cancer_type] = build_page_pairs(spec, hp_html, pt_html, encode, page_stats, sim_threshold)
        stats["pages"][spec.cancer_type] = page_stats
        getattr(encode, "save", lambda: None)()
        log.info("%s: kept %d pairs", spec.cancer_type, page_stats["pairs_kept"])

    family_of = {s.cancer_type: s.family for s in specs if s.cancer_type in per_page}
    fam_counts: dict[str, int] = {}
    for ctype, recs in per_page.items():
        fam_counts[family_of[ctype]] = fam_counts.get(family_of[ctype], 0) + len(recs)
    fam_split = assign_splits(fam_counts)
    split_of = {ctype: fam_split[family_of[ctype]] for ctype in per_page}
    all_records: list[dict] = []
    for ctype, recs in per_page.items():
        for n, rec in enumerate(recs):
            rec.update(id=f"{ctype}-{n:04d}", split=split_of[ctype])
            all_records.append(rec)
    stats["split_before_near_dup"] = {
        s: sum(1 for r in all_records if r["split"] == s) for s in ("train", "val", "test")
    }
    stats["near_dup"] = {"ratio": cfg.PDQ_NEAR_DUP_RATIO, "dropped": drop_near_duplicates(all_records)}
    stats["dedupe_dropped"] = dedupe_records(all_records)
    stats["pairs_dropped_total"] = _sum_page_stat(stats["pages"], "pairs_dropped")
    stats["xref_sentences_stripped_total"] = sum(p["xref_sentences_stripped"] for p in stats["pages"].values())
    stats["pairs_with_xref_stripped_total"] = sum(p["pairs_with_xref_stripped"] for p in stats["pages"].values())
    stats["sim_threshold"] = cfg.PDQ_ALIGN_SIM_THRESHOLD if sim_threshold is None else sim_threshold
    stats["family_to_split"] = fam_split
    stats["page_to_family"] = family_of
    _write_outputs(all_records, split_of, stats, out_path, stats_path)
    _write_borderline(all_records, borderline_path)
    return stats


# --- Sentence-level pairs (extra training data) -------------------------------
def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences (same splitter the xref filter uses)."""
    return [x.strip() for x in _SENT_SPLIT_RE.split(paragraph) if x.strip()]


def _sentence_targets(
    paragraphs: list[str], max_window: int, exclude: set[str] | None = None
) -> list[tuple[tuple[int, ...], str]]:
    """Patient candidates: every sentence and every run of up to `max_window` sentences.

    Runs stay inside one paragraph and never contain a sentence in `exclude`
    (already paired). Indices are global sentence numbers so the aligner can
    stop two overlapping candidates from both being used.
    """
    exclude = exclude or set()
    out: list[tuple[tuple[int, ...], str]] = []
    n = 0
    for para in paragraphs:
        sents = split_sentences(para)
        for a in range(len(sents)):
            for b in range(a, min(a + max_window, len(sents))):
                if not any(x in exclude for x in sents[a : b + 1]):
                    out.append((tuple(range(n + a, n + b + 1)), " ".join(sents[a : b + 1])))
        n += len(sents)
    return out


def align_sentence_windows(
    items: list[SrcItem], targets: list[tuple[tuple[int, ...], str]], encode: Encoder, threshold: float,
) -> list[tuple[tuple[int, ...], tuple[int, ...], float, str, str]]:
    """Greedy one-to-one alignment of professional sentence windows to patient sentence windows.

    Same idea as align_windows, but BOTH sides can be windows (professional
    1-3 sentences, patient 1-2). Multi-sentence candidates are penalised by
    PDQ_WINDOW_MIN_GAIN each, so they only win when clearly better. Candidates
    whose target would be longer than PDQ_MAX_LEN_RATIO x source, or that are
    near-copies (> PDQ_MAX_ALIGN_SIM), are skipped so a better candidate can win.
    Returns (source idxs, target sentence idxs, cosine, source text, target text).
    """
    import numpy as np

    if not items or not targets:
        return []
    wins = make_windows(items, cfg.PDQ_SENT_MAX_WINDOW)
    sim = np.asarray(encode([t for _, t in wins])) @ np.asarray(encode([t for _, t in targets])).T
    sw = np.array([len(t.split()) for _, t in wins])
    tw = np.array([len(t.split()) for _, t in targets])
    s_multi = np.array([len(ix) > 1 for ix, _ in wins], dtype=float)
    t_multi = np.array([len(ix) > 1 for ix, _ in targets], dtype=float)
    rank = sim - cfg.PDQ_WINDOW_MIN_GAIN * (s_multi[:, None] + t_multi[None, :])
    ok = (sim >= threshold) & (sim <= cfg.PDQ_MAX_ALIGN_SIM) & (tw[None, :] <= cfg.PDQ_MAX_LEN_RATIO * sw[:, None])
    rows, cols = np.nonzero(ok)
    order = np.argsort(-rank[rows, cols], kind="stable")
    used_src: set[int] = set()
    used_tgt: set[int] = set()
    out = []
    for k in order:
        si, tj = wins[rows[k]], targets[cols[k]]
        if used_src.intersection(si[0]) or used_tgt.intersection(tj[0]):
            continue
        used_src.update(si[0])
        used_tgt.update(tj[0])
        out.append((si[0], tj[0], float(sim[rows[k], cols[k]]), si[1], tj[1]))
    return out


def _sentence_items(secs: list[tuple[int, Section]]) -> list[SrcItem]:
    """Professional sentences as (text, section id, position); position runs across paragraphs."""
    items: list[SrcItem] = []
    for gid, sec in secs:
        sents = [x for para in sec.paragraphs for x in split_sentences(para)]
        items += [(x, gid, k) for k, x in enumerate(sents)]
    return items


def build_sentence_page_pairs(
    spec: PageSpec, hp_html: str, pt_html: str, encode: Encoder, stats: dict
) -> list[dict]:
    """Sentence-level pairs for one page (no split yet). Same two passes as build_page_pairs.

    Pass 1 uses heading-matched sections at PDQ_SENT_SIM_THRESHOLD; pass 2 aligns
    leftover sentences page-wide at the stricter of that and PDQ_FALLBACK_SIM_THRESHOLD.
    The same filters apply (length ratio, number-not-in-source, xref stripping,
    near-identical), with the lower PDQ_SENT_MIN_WORDS floor.
    """
    hp = parse_sections(hp_html, False)
    pt = parse_sections(pt_html, True)
    group_of = {id(sec): g for g, sec in enumerate(hp)}
    stats.update(candidates=0, dropped={}, xref_sentences_stripped=0, kept=0)
    recs: list[dict] = []
    used_src: set[tuple[int, int]] = set()
    used_tsent: set[str] = set()

    def run_pool(items: list[SrcItem], tgt_secs: list[Section], threshold: float) -> None:
        items = [it for it in items if (it[1], it[2]) not in used_src]
        for pt_sec in tgt_secs:
            targets = _sentence_targets(pt_sec.paragraphs, cfg.PDQ_SENT_PATIENT_WINDOW, used_tsent)
            for idxs, _t, sim, src, tgt in align_sentence_windows(items, targets, encode, threshold):
                stats["candidates"] += 1
                clean, n_xref = strip_xref_sentences(tgt)
                stats["xref_sentences_stripped"] += n_xref
                why = pair_drop_reason(src, clean, sim, cfg.PDQ_SENT_MIN_WORDS)
                if why:
                    stats["dropped"][why] = stats["dropped"].get(why, 0) + 1
                    continue
                used_src.update((items[i][1], items[i][2]) for i in idxs)
                used_tsent.update(split_sentences(tgt))
                recs.append({
                    "source": src, "target": clean, "cancer_type": spec.cancer_type, "hp_url": spec.hp_url,
                    "patient_url": spec.patient_url, "section": pt_sec.heading, "similarity": round(sim, 4),
                    "source_paragraphs": 1, "granularity": "sentence",
                })

    for pt_sec, hp_secs, _ in match_sections(hp, pt):
        run_pool(_sentence_items([(group_of[id(sec)], sec) for sec in hp_secs]), [pt_sec], cfg.PDQ_SENT_SIM_THRESHOLD)
    run_pool(_sentence_items(list(enumerate(hp))), pt,
             max(cfg.PDQ_SENT_SIM_THRESHOLD, cfg.PDQ_FALLBACK_SIM_THRESHOLD))
    stats["kept"] = len(recs)
    return recs


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def drop_sentence_duplicates(sent_recs: list[dict], para_recs: list[dict]) -> dict[str, int]:
    """Drop sentence pairs whose source is already inside a kept paragraph pair of the SAME split (in place).

    Why: the paragraph pair already teaches that text; repeating it as a
    sentence pair would double-weight it. Also drops repeated identical
    (source, target) sentence pairs.
    """
    by_split: dict[str, list[str]] = {}
    for r in para_recs:
        by_split.setdefault(r["split"], []).append(_norm(r["source"]))
    seen: set[tuple[str, str]] = set()
    keep: list[dict] = []
    dropped = {"contained_in_paragraph_pair": 0, "repeated": 0}
    for r in sent_recs:
        src = _norm(r["source"])
        if any(src in big for big in by_split.get(r["split"], [])):
            dropped["contained_in_paragraph_pair"] += 1
        elif (src, _norm(r["target"])) in seen:
            dropped["repeated"] += 1
        else:
            seen.add((src, _norm(r["target"])))
            keep.append(r)
    sent_recs[:] = keep
    return dropped


def run_sentences(
    encode: Encoder | None = None,
    pages_path: Path = cfg.PDQ_PAGES_PATH,
    para_path: Path = cfg.PDQ_PAIRS_PATH,
    stats_path: Path = cfg.PDQ_STATS_PATH,
    out_path: Path = cfg.PDQ_SENT_PAIRS_PATH,
    cache_dir: Path = cfg.PDQ_RAW_DIR,
) -> dict:
    """Build data/pdq_pairs_sent.jsonl from CACHED html; splits copied from the paragraph run.

    Why reuse the paragraph run's page->split map (pdq_stats.json): a sentence
    must land in the same split as its page, otherwise val/test text could leak
    into training. data/pdq_pairs.jsonl is only read, never changed.
    """
    page_split = json.loads(stats_path.read_text(encoding="utf-8"))["split"]["page_to_split"]
    para_recs = [json.loads(l) for l in para_path.open(encoding="utf-8") if l.strip()]
    stats: dict = {"pages": {}}
    recs: list[dict] = []
    for spec in load_pages(pages_path):
        if spec.cancer_type not in page_split:
            continue
        hp_html = load_html(spec.hp_url, spec.hp_fetch_url, cache_dir, fetch=False)
        pt_html = load_html(spec.patient_url, spec.patient_fetch_url, cache_dir, fetch=False)
        if encode is None:
            encode = default_encoder()
        pstats: dict = {}
        page_recs = build_sentence_page_pairs(spec, hp_html, pt_html, encode, pstats)
        for n, r in enumerate(page_recs):
            r.update(id=f"{spec.cancer_type}-s{n:04d}", split=page_split[spec.cancer_type])
        recs += page_recs
        stats["pages"][spec.cancer_type] = pstats
        getattr(encode, "save", lambda: None)()
    stats["before_dedupe"] = len(recs)
    stats["dedupe"] = drop_sentence_duplicates(recs, para_recs)
    # leakage guard: a val/test sentence target nearly equal to ANY train target is dropped
    n_before = len(recs)
    combined = [r for r in para_recs if r["split"] == "train"] + recs
    drop_near_duplicates(combined)
    survivors = {r["id"] for r in combined}
    recs = [r for r in recs if r["id"] in survivors]
    stats["near_dup_dropped"] = n_before - len(recs)
    stats["dedupe_dropped"] = dedupe_records(recs)
    order = ("id", "source", "target", "split", "cancer_type", "hp_url", "patient_url", "section",
             "similarity", "source_paragraphs", "granularity")
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: r[k] for k in order}, ensure_ascii=False) + "\n")
    stats["pairs_per_split"] = {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "val", "test")}
    stats["sim_threshold"] = cfg.PDQ_SENT_SIM_THRESHOLD
    return stats


# --- Deduplication (applies to paragraph, sentence and combined files) --------
def norm_key(text: str) -> str:
    """Comparison key: lower-case, collapsed whitespace, punctuation stripped at both ends."""
    return re.sub(r"\s+", " ", text.lower()).strip(" \t.,;:!?\"'()[]-–—")


def _best_first(records: list[dict]) -> list[dict]:
    """Highest similarity first; on ties prefer paragraph over sentence pairs, then file order."""
    return sorted(records, key=lambda r: (-r["similarity"], r.get("granularity") == "sentence"))


def _keep_best_per_key(records: list[dict], key_of: Callable[[dict], tuple]) -> tuple[list[dict], int]:
    """Keep the best record per key (same split only); original order is preserved."""
    keep_ids: set[int] = set()
    seen: set[tuple] = set()
    for r in _best_first(records):
        k = (r["split"],) + key_of(r)
        if k not in seen:
            seen.add(k)
            keep_ids.add(id(r))
    return [r for r in records if id(r) in keep_ids], len(records) - len(keep_ids)


def drop_near_duplicates_in_train(records: list[dict], ratio: float | None = None) -> int:
    """Within TRAIN, drop records whose target is >= ratio similar (difflib) to a better kept one (in place).

    Why: near-copies in training over-weight one sentence and make val loss
    look better than it is. Speed: targets are sorted by length and only
    compared with kept targets whose length could reach `ratio` (difflib's
    ratio is bounded by 2*min/(a+b)), so ~3k rows take seconds.
    """
    ratio = cfg.PDQ_NEAR_DUP_RATIO if ratio is None else ratio
    train = _best_first([r for r in records if r["split"] == "train"])
    kept: list[tuple[int, str]] = []  # (length, normalized target) of kept records
    drop_ids: set[int] = set()
    sm = difflib.SequenceMatcher(None)
    for r in train:
        t = norm_key(r["target"])
        sm.set_seq2(t)
        dup = False
        for ln, other in kept:
            if 2 * min(ln, len(t)) / max(ln + len(t), 1) < ratio:
                continue
            sm.set_seq1(other)
            if sm.real_quick_ratio() >= ratio and sm.quick_ratio() >= ratio and sm.ratio() >= ratio:
                dup = True
                break
        if dup:
            drop_ids.add(id(r))
        else:
            kept.append((len(t), t))
    records[:] = [r for r in records if id(r) not in drop_ids]
    return len(drop_ids)


def drop_sentence_substrings(records: list[dict]) -> int:
    """Drop sentence pairs whose source OR target is inside a paragraph pair of the same split (in place)."""
    paras: dict[str, tuple[list[str], list[str]]] = {}
    for r in records:
        if r.get("granularity", "paragraph") != "sentence":
            srcs, tgts = paras.setdefault(r["split"], ([], []))
            srcs.append(norm_key(r["source"]))
            tgts.append(norm_key(r["target"]))
    keep: list[dict] = []
    for r in records:
        if r.get("granularity") == "sentence":
            srcs, tgts = paras.get(r["split"], ([], []))
            s, t = norm_key(r["source"]), norm_key(r["target"])
            if any(s in x for x in srcs) or any(t in x for x in tgts):
                continue
        keep.append(r)
    n = len(records) - len(keep)
    records[:] = keep
    return n


def dedupe_records(records: list[dict], ratio: float | None = None) -> dict[str, int]:
    """Apply every dedupe rule in place; returns {rule: number dropped}.

    Rules (each within one split): exact (source, target) repeats; repeated
    targets and repeated sources (best similarity wins); sentence pairs that
    sit inside a paragraph pair; near-duplicate targets inside train. The
    separate val/test-vs-train check (drop_near_duplicates) stays in place.
    """
    counts = {"exact_pair": 0, "duplicate_target": 0, "duplicate_source": 0,
              "sentence_inside_paragraph_pair": 0, "near_duplicate_in_train": 0}
    records[:], counts["exact_pair"] = _keep_best_per_key(
        records, lambda r: (norm_key(r["source"]), norm_key(r["target"])))
    counts["sentence_inside_paragraph_pair"] = drop_sentence_substrings(records)
    records[:], counts["duplicate_target"] = _keep_best_per_key(records, lambda r: (norm_key(r["target"]),))
    records[:], counts["duplicate_source"] = _keep_best_per_key(records, lambda r: (norm_key(r["source"]),))
    counts["near_duplicate_in_train"] = drop_near_duplicates_in_train(records, ratio)
    counts["total"] = sum(counts.values())
    return counts


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _split_counts(records: list[dict]) -> dict[str, int]:
    return {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")}


def dedupe_files(
    para_path: Path = cfg.PDQ_PAIRS_PATH, sent_path: Path = cfg.PDQ_SENT_PAIRS_PATH,
    combined_path: Path = cfg.PDQ_COMBINED_PAIRS_PATH, stats_path: Path = cfg.PDQ_STATS_PATH,
    min_para_sim: float | None = None,
) -> dict:
    """Dedupe the paragraph file, the sentence file, and write the combined training file.

    Paragraph and sentence files are rewritten in place (originals are kept as
    *.predup.jsonl the first time). The combined file = paragraph pairs +
    sentence pairs, deduped together, with the sentence-inside-paragraph rule
    checked only against paragraph pairs that training would actually use
    (similarity >= min_para_sim, default PDQ_COMBINED_MIN_PARA_SIM). Counts go
    into pdq_stats.json under "dedupe".
    """
    min_para_sim = cfg.PDQ_COMBINED_MIN_PARA_SIM if min_para_sim is None else min_para_sim
    out: dict = {}
    para = _read_jsonl(para_path)
    for path in (para_path, sent_path):
        bak = path.with_suffix(".predup.jsonl")
        if path.exists() and not bak.exists():
            bak.write_bytes(path.read_bytes())
    out["paragraph_before"] = _split_counts(para)
    out["paragraph_dropped"] = dedupe_records(para)
    out["paragraph_after"] = _split_counts(para)
    _write_jsonl(para_path, para)
    result: dict = {"paragraph": out}
    if sent_path.exists():
        sent = _read_jsonl(sent_path)
        before = _split_counts(sent)
        dropped = dedupe_records(sent)
        # also against the paragraph pairs training will use
        usable = [r for r in para if r["similarity"] >= min_para_sim]
        combined = usable + sent
        dropped["sentence_inside_paragraph_pair"] += drop_sentence_substrings(combined)
        dropped["total"] = sum(v for k, v in dropped.items() if k != "total")
        sent[:] = [r for r in combined if r.get("granularity") == "sentence"]
        result["sentence"] = {"before": before, "dropped": dropped, "after": _split_counts(sent)}
        _write_jsonl(sent_path, sent)
        every = para + sent
        result["combined_extra_dropped"] = dedupe_records(every)
        result["combined"] = _split_counts(every)
        _write_jsonl(combined_path, every)
    if stats_path.exists():
        st = json.loads(stats_path.read_text(encoding="utf-8"))
        st["dedupe"] = result
        stats_path.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _sum_page_stat(pages: dict, key: str) -> dict[str, int]:
    """Add up a {reason: count} dict across all pages."""
    total: dict[str, int] = {}
    for p in pages.values():
        for reason, n in p[key].items():
            total[reason] = total.get(reason, 0) + n
    return total


def _write_borderline(records: list[dict], path: Path) -> None:
    """Save random pairs from each similarity band for a human to judge.

    Why: the alignment threshold is a judgement call (is "same topic" good
    enough, or must the patient text be a true paraphrase?). Showing a few real
    pairs per band lets a person pick it.
    """
    rng = random.Random(cfg.RANDOM_SEED)
    keep = ("id", "similarity", "source_paragraphs", "section", "source", "target")
    out: dict[str, list[dict]] = {}
    for lo, hi in cfg.PDQ_BORDERLINE_BANDS:
        band = [r for r in records if lo <= r["similarity"] < hi]
        picks = rng.sample(band, min(cfg.PDQ_BORDERLINE_COUNT, len(band)))
        out[f"{lo:.2f}-{hi:.2f}"] = [{k: r[k] for k in keep} for r in picks]
    if any(out.values()):
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_outputs(records: list[dict], split_of: dict, stats: dict, out: Path, stats_path: Path) -> None:
    order = ("id", "source", "target", "split", "cancer_type", "hp_url", "patient_url", "section", "similarity", "source_paragraphs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in order}, ensure_ascii=False) + "\n")
    counts = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")}
    total = sum(counts.values()) or 1
    stats["split"] = {
        "method": "whole organ families assigned to splits (closest fit to PDQ_SPLIT by pair count), then near-duplicate val/test targets dropped",
        "page_to_split": split_of,
        "pairs_per_split": counts,
        "fractions": {k: round(v / total, 3) for k, v in counts.items()},
    }
    stats["total_pairs"] = len(records)
    stats["multi_source_pairs"] = sum(1 for r in records if r["source_paragraphs"] > 1)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build PDQ professional->patient training pairs.")
    ap.add_argument("--types", help="comma-separated cancer_type keys from data/pdq_pages.json")
    ap.add_argument("--no-fetch", action="store_true", help="use cached HTML only")
    ap.add_argument("--sim-threshold", type=float,
                    help="override PDQ_ALIGN_SIM_THRESHOLD for this run (default stays in config.py)")
    ap.add_argument("--out", type=Path, default=cfg.PDQ_PAIRS_PATH, help="pairs JSONL path")
    ap.add_argument("--stats-out", type=Path, default=cfg.PDQ_STATS_PATH, help="stats JSON path")
    ap.add_argument("--sentences", action="store_true",
                    help="build sentence-level pairs from cached HTML into data/pdq_pairs_sent.jsonl")
    ap.add_argument("--dedupe", action="store_true",
                    help="dedupe data/pdq_pairs.jsonl + pdq_pairs_sent.jsonl in place and write pdq_pairs_combined.jsonl")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.dedupe:
        print(json.dumps(dedupe_files(), indent=2))
        return
    if args.sentences:
        st = run_sentences()
        print(json.dumps({k: v for k, v in st.items() if k != "pages"}, indent=2))
        return
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    stats = run(types=types, fetch=not args.no_fetch, out_path=args.out,
                stats_path=args.stats_out, sim_threshold=args.sim_threshold)
    print(json.dumps({k: stats[k] for k in ("total_pairs", "split", "failed_pages")}, indent=2))


if __name__ == "__main__":
    main()
