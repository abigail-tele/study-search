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

**⚠️ Known risk (found 2026-10-24):** on real trial text the first checkpoint lowers reading grade (~25 → ~10) but writes generic or made-up sentences that a number/drug check cannot catch, e.g. *"Women age 18 years or older are more likely to have breast cancer than men."* (source was only an age rule) and openers like *"There are different types of treatment for…"*. Cause: trained on NCI education pages, used on trial eligibility text (domain mismatch). Being addressed with a sentence-level "support" check (each output sentence must match something in the source) and a trial-text evaluation set. Until the human-written `faithfulness_check` exists, the app shows glossary-swapped original wording, so patients never see unchecked model text.

**Experiments (2026-10-24):** training data de-duplicated first (paragraph pairs 1,649 → 1,440; sentence pairs 1,556 → 918; per-rule counts in `data/pdq_stats.json`). Model choice used the fixed val set + a real-trial set only; the test split was scored once at the end. Full P/R/F1 tables: `docs/EVAL.md`, `results/baselines_results.csv`, per-system raw predictions and error analyses in `results/`.

| Run | lr | Min sim | Train pairs | Best epoch | Fixed-val loss |
|---|---|---|---|---|---|
| baseline (before dedup) | 3e-4 | 0.70 | 1,072 | 1 | 1.967 |
| **s70 (winner)** | 1e-4 | 0.70 | 911 | 3 | 1.947 |
| s60 | 1e-4 | 0.60 | 1,124 | 3 | 1.932 |
| sent (paragraph + sentence pairs) | 1e-4 | 0.70 | 1,492 | 3 | 1.914 |
| sent-lr5e5 | 5e-5 | 0.70 | 1,492 | 5 | 1.920 |

**On 240 real trial texts** (40 brief summaries + 200 eligibility criteria, 35 cancer types). Key-term = numbers, ages, units, drugs, markers, stages; precision < 1 means invented facts. Support = share of output sentences backed by the source. Boilerplate = generic textbook openers not in the source.

| System | Key-term P / R / F1 | Support | Boilerplate | Reading grade (source 15.4) |
|---|---|---|---|---|
| copy source (no change) | 1.00 / 1.00 / 1.00 | 1.00 | 0% | 15.4 |
| zero-shot flan-t5 | 0.985 / 0.726 / 0.836 | 0.96 | 0% | 14.7 |
| few-shot flan-t5 (3 examples) | 0.960 / 0.786 / 0.865 | 0.89 | 0% | 15.6 |
| fine-tuned baseline | 1.00 / 0.477 / 0.646 | 0.73 | 43.8%* | 10.7 |
| fine-tuned s70 (previous winner) | 0.996 / 0.582 / 0.735 | 0.83 | 21.3%* | 11.6 |
| fine-tuned s60 | 0.996 / 0.488 / 0.655 | 0.80 | 23.8%* | 11.3 |
| fine-tuned sent | 0.996 / 0.532 / 0.693 | 0.78 | 22.1%* | 11.0 |
| fine-tuned clean-s70 (cleaned targets, paragraphs) | 0.988 / 0.516 / 0.678 | 0.82 | 15.8% | 11.4 |
| **fine-tuned clean-sent (cleaned targets, paragraphs + sentences; current best)** | 0.992 / 0.534 / 0.694 | 0.84 | 13.8% | 11.3 |

ROUGE-L F1 vs NCI patient text: val (fixed 125) s70 0.221, clean-sent 0.219, baseline 0.187, copy 0.198; test (136, run once per winner) s70 0.231, clean-sent 0.240, copy 0.205.

\* Boilerplate was re-scored with a broader pattern list added after the first comparison, so these values are higher than first reported (s70 was 10.4%, sent 10.4%, s60 13.3%, baseline 37.9%); rows without * were scored with the current list.

**Cleaned-target experiment:** generic PDQ intro sentences (101) and target sentences unsupported by their source (252) were removed from the training targets (`pipeline/clean_pairs.py`, 2,358 -> 2,162 pairs after re-dedupe). Runs: clean-s70 (paragraphs, 846 train pairs) and clean-sent (paragraphs + sentences, 1,434 train pairs). clean-sent has the best support (0.84) and lowest boilerplate (13.8%) of the fine-tuned runs but lower key-term recall than s70 (0.534 vs 0.582); key-term precision differs by only 0.004, so it replaced s70 as `checkpoints/flan-t5-pdq-best`. Details: `docs/EVAL.md`.

**What this means (honest read):** retraining improved the model a lot (support 0.73 → 0.83, boilerplate 38% → 10%, key-term F1 0.65 → 0.74). Fine-tuning is the only thing that actually lowers reading level (~15 → ~11.6); zero/few-shot mostly copy the source. But the fine-tuned model still drops content (key-term recall 0.58) and ~17% of its sentences aren't backed by the source, so it can't be shown to patients unchecked. Plan: every model sentence must pass the support check *and* the human-written `faithfulness_check`; anything that fails is replaced by the glossary-swapped original for that piece.

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
