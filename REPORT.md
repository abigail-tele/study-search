# Project Report: Cancer Trials Plain-Language Finder

Living status report for the team. Updated by Claude Code (orchestrator) and edited by humans.
Last updated: **2026-10-24 (Sat)**. Code freeze: **Sun Oct 25, 6 pm**.
Detail lives in: `docs/PLAN.md` (plan), `TODO.md` (open items), `docs/HOW_IT_WORKS.md` (module explanations), `docs/EVAL.md` (numbers), `docs/TRAINING.md` (how to train), `docs/DEPLOY.md` (Hugging Face Spaces), `AI_USE.md` + `CREDITS.md` (disclosure).

---

## 1. Where we are

| Area | Status | Key numbers |
|---|---|---|
| Trial index (ClinicalTrials.gov to SQLite) | Done | 27,657 cancer trials, 35 cancer types, ~420k sites (98.5% geocoded); interventions (drugs/treatments) stored for 25,401 trials |
| Tag accuracy (reviewer samples) | Done | 90-100% per type after fixing "non-small cell", "non-Hodgkin", "cervical = neck" mismatches |
| Embeddings + FAISS index | Done | 27,657 vectors, built on CPU |
| Search (`search.py`, `taxonomy.py`) | Works end to end | off-topic queries redirected 10/10; the 25 lay queries all resolve to the right cancer type (25/25); ~90 ms per search (375 ms with ZIP); models load from local files only (rule 4) |
| FastAPI backend (`app.py`) | Done | 4 endpoints, read-only DB, query/ZIP never logged |
| Frontend | Done; checked with headless screenshots only | "About you" condition matching, "Study at a glance" (including what is being tested), "What they're looking for" / "Who can't join". Still needs a human click-through |
| Summarizer (`summarize.py`, `support.py`) | Done | per-piece faithfulness check + sentence guards; a failing piece shows the glossary-swapped original |
| Batch summaries | ✅ Done with model G (Oct 25) | All 27,657 trials (581,161 pieces): 20,432 RECRUITING / NOT_YET_RECRUITING, then 7,225 ACTIVE_NOT_RECRUITING / ENROLLING_BY_INVITATION. Model G: 69.4% of pieces shown as model text after all checks (67.1% on the first 20,432, 75.1% on the later 7,225); 1,110 trials with no fallback. Earlier model (clean-sent): 61.7% |
| Model experiments | Done | PDQ-only, cleaned targets, ClinicalTrials.gov pairs (7,873), flan-t5-large, Qwen LoRA (rejected); see section 3 |
| Tests | Passing | 452 collected (`pytest --co`) |
| Deployment | Guide written, not deployed yet | Hugging Face Spaces, Docker (`docs/DEPLOY.md`); needs a team HF account |

## 2. Who wrote the three formerly human-owned functions
`detect_cancer_types` (`backend/taxonomy.py`), `reciprocal_rank_fusion` (`backend/search.py`) and `faithfulness_check` (`backend/summarize.py`) were first left as stubs for the team. At the team's explicit request they were implemented by Claude Code (Claude Sonnet subagents), marked AI-written in the code, and logged in `AI_USE.md`. **The team must still review each one and be able to explain it to judges; the "human review" cells in `AI_USE.md` are still pending.** All three have full test suites (the old strict-xfail markers are gone).

| Function | What it does |
|---|---|
| `detect_cancer_types` | plain-words query to cancer type ids, or `[]` if off-topic. Union of exact + umbrella matches ("breast and lung cancer" gives both); fuzzy spelling, then a meaning comparison, only if nothing matched, and only when the query has a cancer word ("stomach ache" is never stomach cancer); short codes (AML, CLL) only as whole words; bare "all" never counts. |
| `reciprocal_rank_fusion` | merges the keyword and meaning rankings: score = sum of 1/(60 + rank); ties broken by NCT ID. |
| `faithfulness_check` | every number, age, unit, dose, gene/marker, stage term and drug (list in `data/drug_lexicon.txt`) in the model output must appear in the source; flipped comparisons and negations are rejected. |

## 3. Model training
**Data:** NCI PDQ professional-to-patient pairs (168 pages; 2,162 cleaned paragraph + sentence pairs), split by organ family so near-identical pages never straddle train/val/test, plus 7,873 in-domain pairs from ClinicalTrials.gov (a trial's detailed description paired with its easier brief summary; `pipeline/ctgov_pairs.py`, public domain; every trial in the evaluation set is excluded). Full numbers: `docs/EVAL.md`.

**What we learned along the way.** The first fine-tune overfit after epoch 1 and, on real trial text, wrote generic or invented sentences (for example "Women age 18 years or older are more likely to have breast cancer than men" from a bare age rule). Fixes: lower learning rate, de-duplication, sentence-level pairs, cleaned training targets (generic intro sentences and unsupported target sentences removed), and automatic guards in `backend/support.py` (every output sentence must be backed by the source; generic openers, new risk words and flipped comparisons are dropped).

**Bigger models and in-domain data** (240 real trial texts; guard-pass = no sentence flagged by the guards; the source text reads at FK grade 15.4):

| System | Key-term P / R / F1 | Support | Boilerplate | Guard-pass | FK out |
|---|---|---|---|---|---|
| clean-sent (PDQ only; was production) | 0.992 / 0.534 / 0.694 | 0.843 | 13.8% | 67.5% | 11.3 |
| **G flan-t5-base + ClinicalTrials.gov pairs (production, switching now)** | 0.972 / 0.684 / 0.803 | 0.951 | 5.8% | 84.6% | 13.4 |
| H flan-t5-large, PDQ only | 0.979 / 0.597 / 0.742 | 0.885 | 13.3% | 75.4% | 12.0 |
| I flan-t5-large + ClinicalTrials.gov pairs | 0.980 / 0.741 / 0.844 | 0.945 | 2.9% | 90.8% | 14.1 |
| J Qwen2.5-1.5B LoRA (rejected) | 0.580 / 0.174 / 0.268 | 0.235 | 10.0% | 18.8% | 12.4 |
| J2 Qwen2.5-1.5B LoRA (rejected) | 0.518 / 0.157 / 0.241 | 0.221 | 10.8% | 16.3% | 12.4 |

**Why G and not I.** I scores best on the safety measures but barely simplifies (FK 14.1 vs 15.4 for the source), and regenerating every summary with flan-t5-large would take about 5-6 hours. G keeps far more facts than clean-sent at the same CPU cost (recall 0.684 vs 0.534, guard-pass 84.6% vs 67.5%) and reads at FK 13.4. Both Qwen LoRA runs ignored the source and were rejected.

**Production summaries (model G, final, Oct 25):** all 27,657 trials, 581,161 pieces. After the faithfulness check (numbers, units, markers, stages, drug names, flipped or dropped comparisons like "≥ 18" → "18") and the guards (unsupported, generic, invented-definition and invented-claim sentences), **69.4%** of pieces are shown as model text (67.1% for the 20,432 RECRUITING / NOT_YET_RECRUITING trials, 75.1% for the 7,225 ACTIVE_NOT_RECRUITING / ENROLLING_BY_INVITATION trials added later, whose model input had the ≥/≤ symbols spelled out); the rest fall back to the original wording with plain definitions. Effective reading grade of what patients see ≈ 14.8 (the raw source text measures ≈ 20.8 on the same 1,000-trial sample). Details: `docs/EVAL.md` "Production summaries".

**Known limitation (honest):** a human spot-check of 30 random shown bullets after the final checks still found ~4 that state something the source doesn't (e.g. an exclusion rewritten as a promise, an invented referral). Automatic word-level checks cannot catch every fluent paraphrase. Mitigation: the eligibility lists ("What they're looking for" / "Who can't join") now show the **original wording with plain definitions by default**, and the model's version only behind a "Show simplified wording" button; every panel says the plain text is computer-made and links the original. We also tested zero-shot 7–8B models (Qwen2.5-7B, Qwen3-8B): plainer text, but only ~40% of it could be verified by our checks and generating all summaries would take 38–68 h, so we kept the small fine-tuned model.

**What this means (honest read):** adding in-domain ClinicalTrials.gov pairs helped more than anything else: the model keeps more of the facts and writes fewer unsupported sentences. The cost is that it simplifies less (reading grade ~13 vs ~11 for the PDQ-only model, where the source is ~15). Even the best model drops some content and still produces some unbacked sentences, so its output is never shown unchecked: every piece passes the faithfulness check and the guards, or falls back to the original wording with hard words explained. The reading-level gain is modest, not dramatic.

## 4. Decisions and open items for the team
1. Sign off on permissive-but-unlisted licenses: certifi, tqdm (MPL-2.0), regex (CNRI-Python), typing_extensions + defusedxml (PSF-2.0), shellingham (ISC).
2. Fill in the "human review" column of `AI_USE.md` (all rows still say pending) and review the three AI-written functions in section 2.
3. Merge PR #3 (`abigail/vector-search`) after the teammate's two test edits (see `TODO.md`).
4. Proofread `data/glossary.json` (76 plain definitions an AI agent paraphrased from the NCI Dictionary; patients see these).
5. Deployment: create the Hugging Face account and Space and follow `docs/DEPLOY.md` (`trials_ship.db` ~430 MB and `faiss.index` ~43 MB go through Git LFS).
6. 333 trials that say only "Cancer"/"Neoplasms" (no specific type) are excluded. Add an "any cancer" type?

## 5. How to run (Windows PowerShell, from `study-search\`)
```powershell
.\.venv\Scripts\python -m pipeline.fetch_models   # run once first: puts the embedding model in models/ (the app never downloads at request time)
.\run_api.ps1                 # whole app at http://127.0.0.1:8000 (needs data/trials.db, data/faiss.index, models/, checkpoints/flan-t5-pdq-best)
.\run_frontend.ps1            # frontend only, mock data, http://localhost:8765
.\.venv\Scripts\python -m pytest -q
.\.venv-train\Scripts\python -m pipeline.train --min-similarity 0.70 --lr 1e-4   # GPU training (see docs/TRAINING.md)
```

## 6. Changelog
- **Fri-Sat night:** Phase 1: scaffold, taxonomy, full index (35 types), tag verification + non-cancer filter, PDQ pairs, frontend on mocks. Gate 1 passed.
- **Sat:** training experiments (dedupe, sentence pairs, cleaned targets), summarizer guards, embeddings/FAISS, search, API. Gate 2 fixes (disclosure gaps, model download at request time).
- **Sat (later):** the three formerly human-owned functions implemented by Claude Code at the team's request; batch summaries for 20,432 trials; "About you" matching and study overview with interventions; ClinicalTrials.gov pairs, flan-t5-large and Qwen LoRA experiments; model G chosen for production; Gate 3 doc refresh.
