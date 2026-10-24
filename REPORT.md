# Project Report: Cancer Trials Plain-Language Finder

Living status report for the team. Updated by Claude Code (orchestrator) and edited by humans.
Last updated: **2026-10-24 (Sat)**. Code freeze: **Sun Oct 25, 6 pm**.
Detail lives in: `docs/PLAN.md` (plan), `TODO.md` (checklist), `docs/HOW_IT_WORKS.md` (module explanations), `docs/EVAL.md` (numbers), `docs/TRAINING.md` (how to train), `AI_USE.md` + `CREDITS.md` (disclosure).

---

## 1. Where we are

| Area | Status | Key numbers |
|---|---|---|
| Trial index (ClinicalTrials.gov → SQLite) | ✅ Done (Gate 1 passed) | 27,657 cancer trials, 35 cancer types, ~420k sites (98.5% geocoded) |
| Tag accuracy (reviewer samples) | ✅ | 90–100% per type after fixing "non-small cell", "non-Hodgkin", "cervical = neck" mismatches |
| PDQ training pairs (NCI, public domain) | ✅ | 168 page pairs → 1,649 pairs (≥0.60 similarity) / 1,411 (≥0.70) |
| Model fine-tuning (flan-t5-base) | 🔄 Experiments running | Baseline run done; lower-learning-rate + dedup + sentence-pair runs in progress |
| Embeddings + FAISS index | ✅ | 27,657 vectors, 334 s to build on CPU |
| Search pipeline (`search.py`) | ✅ built, ⏳ waits on 2 human-written functions | ~90 ms per search (375 ms with ZIP) |
| FastAPI backend (`app.py`) | ✅ | 4 endpoints, read-only DB, query/ZIP never logged |
| Frontend | ✅ on mock + live API | needs a human click-through |
| Summarizer (`summarize.py`) | 🔄 being written | |
| Tests | ✅ | 178 passed, 41 "xfail" = tests waiting for the human-written functions |

## 2. Human-written functions (team members write and explain these)
The code around them is finished and tested; each has a stub that raises `NotImplementedError` and a full test suite marked `xfail(strict=True)`. When you implement one, its tests will start passing and **you must delete the `xfail` marker** (pytest will tell you).

| Function | File | Tests | What it does |
|---|---|---|---|
| `detect_cancer_types` | `backend/taxonomy.py` | `tests/test_taxonomy.py` (32) | plain-words query → cancer type ids, or `[]` if off-topic. Use `data/umbrella_terms.json`; never match the bare word "all". |
| `reciprocal_rank_fusion` | `backend/search.py` | `tests/test_search.py` (9) | merge keyword + meaning rankings: score = Σ 1/(k + rank) |
| `faithfulness_check` | `backend/summarize.py` | `tests/test_summarize.py` | every number, age, unit, drug, gene/marker, stage term in the model output must appear in the source |

Until these exist: search returns a clean "ranking not implemented yet" (HTTP 503), and every summary shows the glossary-swapped original wording (safe default for rule 3).

## 3. Model training

**Data:** `data/pdq_pairs.jsonl` (professional NCI PDQ text → patient-version text). Split train/val/test by *organ family* so near-identical pages (e.g. colon vs rectal) never leak across splits. Filters: similarity threshold, length ratio ≤1.5×, numbers-not-in-source dropped, cross-reference sentences ("Learn more…") stripped, near-duplicates of train removed from val/test.

**Run 1 (baseline fine-tune):** min-similarity 0.70, lr 3e-4, 1,072 train / 125 val pairs, RTX 5090, ~3 min.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 1.93 | **1.97** (best, saved) |
| 2 | 1.45 | 1.98 |
| 3 | 1.20 | 2.00 → early stop |

Train loss kept falling while val loss rose after epoch 1 → **overfitting** (small dataset). Fixes being tried: lower learning rate (`--lr 1e-4`), training-data de-duplication, more data via sentence-level alignment, and a fixed 125-pair comparison set (`data/pdq_val_fixed.jsonl`) so all runs are scored on the same examples.

**Experiments:** _results table will be filled in here and in `docs/EVAL.md` when runs finish (real numbers only)._

## 4. Decisions the team needs to make
1. Similarity cutoff for training pairs: 0.60 (more data, looser pairs) vs 0.70 (reviewer recommends).
2. textstat for reading level? It pulls in pyphen (GPL/LGPL/MPL tri-license). Current choice: our own small Flesch-Kincaid function (no extra dependency).
3. Sign off on permissive-but-unlisted licenses: certifi, tqdm (MPL-2.0), regex (CNRI-Python), typing_extensions (PSF-2.0), shellingham (ISC).
4. Deployment: `trials.db` (~460 MB) and `faiss.index` (~43 MB) are not in git → choose Hugging Face Spaces upload / Git LFS / build step.
5. 333 trials that say only "Cancer"/"Neoplasms" (no specific type) are excluded. Add an "any cancer" type?

## 5. How to run (Windows PowerShell, from `study-search\`)
```powershell
.\run_api.ps1                 # whole app at http://127.0.0.1:8000 (needs data/trials.db + data/faiss.index)
.\run_frontend.ps1            # frontend only, mock data, http://localhost:8765
.\.venv\Scripts\python -m pytest -q
.\.venv-train\Scripts\python -m pipeline.train --min-similarity 0.70 --lr 1e-4   # GPU training (see docs/TRAINING.md)
```

## 6. Changelog
- **Fri–Sat night:** Phase 1: scaffold, taxonomy, full index (35 types), tag verification + non-cancer filter (3 review rounds), PDQ pairs, frontend on mocks. Gate 1 passed.
- **Sat:** `train.py` + first fine-tune (overfits after epoch 1). Phase 2 started in parallel: embeddings, FAISS, search, API, test queries draft, summarizer.
