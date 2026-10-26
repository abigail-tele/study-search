# To-do list: Cancer Trials Plain-Language Finder

Code freeze **Sun Oct 25, 6 pm**. Submit Sunday night (buffer: Mon Oct 26 morning, if that is the district deadline).
Full plan: `docs/PLAN.md`. Last updated 2026-10-24 (after Gate 1).

## ⚠️ Teammate branches not merged yet (abigail/attributions, abigail/vector-search)
Both branches start from commit `6f65350`, before `detect_cancer_types` was implemented on `main`. A trial merge
(cancelled, nothing changed on their branches) showed everything merges except the taxonomy tests. To do, **by the
teammate in their own files**:
- [x] `tests/test_taxonomy.py`: removed the `HUMAN` definition (unused markers already gone).
- [x] `tests/fixtures/tiny_taxonomy.py`: added `monkeypatch.setattr(taxonomy.config, "TYPE_DETECT_CUE_REGEX", r"")`.
- [x] Implemented the cancer-word guard in `backend/taxonomy.py::_embedding_match` (uses TYPE_DETECT_CUE_REGEX).
- [x] Merged `abigail/vector-search` and `abigail/attributions` into `main` (already on main).
      `backend/config.py` (keep `TYPE_DETECT_EMBED_MARGIN`/`_MAX_TYPES` **and** `TYPE_DETECT_CUE_REGEX`),
      `docs/EVAL.md`, `docs/HOW_IT_WORKS.md` (taxonomy paragraph), `backend/taxonomy.py` (keep
      `cancer_type_term_matrix()` and make `_embedding_match` use it with the margin and cap; the trial merge did
      exactly this and 50/53 taxonomy tests passed before the two edits above).
- Why keep the cancer-word guard: with the real model and no guard, "stomach ache" → stomach cancer and "bone pain"
  → sarcoma (rule 1). With it, `python -m pipeline.evaluate_types` gives off-topic 10/10 at every threshold.
- Backup of `main` before the trial merge: local branch `backup/main-before-abigail-merge`.

## Done: Phase 1 (commit `8025be2`)
- [x] Repo scaffold, `backend/config.py`, requirements, CREDITS / AI_USE / README / docs skeletons
- [x] `data/cancer_types.json` (35 types) + `data/umbrella_terms.json`
- [x] ClinicalTrials.gov v2 API verified (PLAN §6)
- [x] `db.py` + `indexer.py`: all 35 types indexed, 27,657 cancer trials, tags verified, non-cancer trials dropped
- [x] `geo.py` + Census ZIP centroids
- [x] `pdq_pairs.py`: 168 NCI PDQ pages → 1,649 training pairs
- [x] Frontend (search, cards, off-topic chips, summary panel) on mock data
- [x] 93 tests passing, Gate 1 reviewed

## Human: do these first (only a person can)
- [ ] Confirm district + deadline on congressionalappchallenge.us
- [ ] Confirm every member is eligible (middle/high school, ≥ half in district, one app per student)
- [ ] Register the team on the portal
- [ ] Decide who runs training on the GPU (see `docs/TRAINING.md` once it exists)
- [ ] Pick the training-pair similarity cutoff: 0.60 (1,649 pairs) or 0.70 (1,411 pairs; reviewer recommends 0.70)
- [ ] Model: flan-t5-base only, or also try a small Qwen as a comparison?
- [ ] textstat license call: it pulls in pyphen (GPL/LGPL/MPL tri-license). Use it under MPL-1.1, or write our own Flesch-Kincaid?
- [ ] Sign off on non-listed permissive licenses: certifi + tqdm (MPL-2.0), regex (CNRI-Python), typing_extensions (PSF-2.0), shellingham (ISC)
- [ ] Spot-check `data/retag_dropped.csv` (trials we removed) and `data/pdq_borderline_samples.json`
- [ ] Fill in the "human review" column in `AI_USE.md` (all rows say "pending")
- [ ] Click through the frontend in a real browser (keyboard only + phone width + print)
- [ ] Push to GitHub (repo owner: abigail-tele)

## Human-written code (team members write and explain these)
- [ ] `search.py::reciprocal_rank_fusion`
- [ ] `summarize.py::faithfulness_check`
- [x] `taxonomy.py::detect_cancer_types`: uses `data/umbrella_terms.json`, never matches bare "all", and uses `cancer_type_term_matrix()` for the semantic fallback after exact/umbrella/fuzzy fail
- [ ] Run `python -m pipeline.evaluate_types`, pick the semantic threshold, and paste the real counts into `docs/EVAL.md`

## Phase 2: Sat Oct 24 (search + model)
- [ ] `pipeline/train.py` + `docs/TRAINING.md` (in progress)
- [ ] HUMAN: run training on the GPU → checkpoint + val loss curve
- [ ] `pipeline/embed.py` + FAISS index (all-MiniLM-L6-v2)
- [ ] `search.py`: tag candidates → FTS5 top 50 + FAISS top 50 → RRF → filters → top 10, citation builder
- [ ] `taxonomy.py` loader + tests for the human-written detector
- [ ] `data/test_queries.json`: 25 lay queries + 10 off-topic (grunt drafts, humans pick correct NCTs)
- [ ] `summarize.py`: chunking, beam 4, include/exclude split, faithfulness stub + tests
- [ ] `backend/app.py` FastAPI endpoints wired to real search. **Turn uvicorn's access log off** (it would record `?q=` and `?zip=`, which breaks rule 8)
- [ ] Gate 2: reviewer checks rules 1–4 and the faithfulness logic

## Phase 3: Sun Oct 25 (batch, deploy, eval, submit)
- [ ] `pipeline/batch_summarize.py` (HUMAN runs on GPU) → summaries table + pass rate
- [ ] `/api/summary` + summary panel on live data (flip `USE_MOCK = false` in `frontend/api.js`)
- [ ] `pipeline/evaluate.py` → `docs/EVAL.md` with real numbers
- [ ] Deploy (Dockerfile). `trials.db` is 460 MB and not in git → plan how to ship it (Git LFS / HF Spaces upload / build step)
- [ ] Final HOW_IT_WORKS.md, CREDITS.md, AI_USE.md
- [ ] HUMAN: 3–5 people read summaries; record ≤3 min demo; write technical-challenge answer
- [ ] Gate 3: reviewer full pass; code freeze 6 pm; submit

## Known small issues (Phase 2)
- 333 trials that mention cancer but fit no type were dropped (e.g. conditions just "Cancer"). Could add a "solid tumors / any cancer" type.
- Inverted names like "Myeloid Leukemia, Philadelphia Positive" can still mis-tag (1 known trial).
- Incremental index refresh (`python -m pipeline.indexer --auto-since`) not scheduled yet (cut-list item 2).
