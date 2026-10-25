# To-do list: Cancer Trials Plain-Language Finder

Code freeze **Sun Oct 25, 6 pm**. Submit Sunday night (buffer: Mon Oct 26 morning, if that is the district deadline).
Full plan: `docs/PLAN.md`. Status: `REPORT.md`. Last updated 2026-10-24 (Gate 3 doc refresh).

## Teammate branches not merged yet
- `abigail/attributions`: merged (PR #1, brought in through #4).
- PR #3 `abigail/vector-search`: still open. It starts from a commit before `detect_cancer_types` was implemented on `main`, so, **by the teammate in their own files**:
  - [ ] `tests/test_taxonomy.py`: remove the `@HUMAN` (strict xfail) markers on the 8 new semantic-fallback tests: the
        function is implemented now, so strict xfail turns them into failures. `main` also no longer defines `HUMAN`, so the merged file won't load until this is done.
  - [ ] `tests/fixtures/tiny_taxonomy.py`: inside `install()`, add
        `monkeypatch.setattr(taxonomy.config, "TYPE_DETECT_CUE_REGEX", r"")`. The real detector only runs the semantic
        step when the query has a cancer word ("stomach ache" guard); the hand-set test phrases ("close pair",
        "clear winner", "four types") have none, so 3 tests fail without this line.
  - [ ] Then merge. Conflicts to resolve, all keep-both: `AI_USE.md` (rows), `backend/config.py` (keep `TYPE_DETECT_EMBED_MARGIN`/`_MAX_TYPES` and `TYPE_DETECT_CUE_REGEX`), `docs/EVAL.md`, `docs/HOW_IT_WORKS.md` (taxonomy paragraph), `backend/taxonomy.py` (keep `cancer_type_term_matrix()` and make `_embedding_match` use it with the margin and cap).
  - Note: `pipeline/evaluate_types.py` exists only on that branch (not on `main`).
  - Why keep the cancer-word guard: with the real model and no guard, "stomach ache" gives stomach cancer and "bone pain" gives sarcoma (rule 1). With it, off-topic is 10/10 at every threshold.

## Teammate-owned files: changes needed (NOTICE and data/README.md; not edited by the docs agent)
- [ ] `NOTICE` (results section, around line 54): it says `results/` holds model outputs and source-text excerpts. Only `results/baselines_results.csv` (metrics) is tracked in git; the raw predictions and error analyses are git-ignored. Reword so it matches.
- [ ] `data/README.md`: the file table lists only some of the tracked/untracked data files. Add `ctgov_pairs_stats.json` (counts from the ClinicalTrials.gov pair builder; public domain source, project annotations MIT) and `drug_lexicon.txt` (drug names for the faithfulness check; project data, MIT). Also list `ctgov_pairs.jsonl` / `ctgov_pairs_mix.jsonl` (ClinicalTrials.gov detailed-to-brief training pairs, public domain), `batch_summary_stats*.json`, and `glossary.json` already listed; check the "git-ignored" sentence against `.gitignore`.

## Done
- Phase 1: scaffold, `config.py`, 35-type taxonomy + umbrella terms, indexer (27,657 trials, tags verified), geo + ZIP centroids, PDQ pairs, frontend on mocks. Gate 1 passed.
- Phase 2: embeddings + FAISS, search (tags, FTS5, FAISS, RRF, filters, citations), API (4 endpoints, no query/ZIP logging), 25 lay + 10 off-topic test queries drafted, summarizer + guards (`support.py`), own Flesch-Kincaid, glossary. Gate 2 passed after fixes.
- The three formerly human-owned functions implemented by Claude Code at the team's request (team must review).
- Phase 3: `faithfulness_check`, batch summaries for 20,432 trials, `evaluate.py` and `make_ship_db.py`, "About you" condition matching, "Study at a glance" with interventions, ClinicalTrials.gov training pairs, bigger-model experiments, Dockerfile + `docs/DEPLOY.md`, copyright audit (`NOTICE`, `data/README.md`).
- Production model switched to G (flan-t5-base + ClinicalTrials.gov pairs); summaries regenerating tonight.

## Human to-dos (only a person can)
- [ ] Fill in the "human review" column in `AI_USE.md` (all rows say "pending"); review and be able to explain `detect_cancer_types`, `reciprocal_rank_fusion`, `faithfulness_check`
- [ ] License sign-off: certifi + tqdm (MPL-2.0), regex (CNRI-Python), typing_extensions + defusedxml (PSF-2.0), shellingham (ISC)
- [x] Hugging Face account + Space `ngroborocks/CongessionalAppChallenge` created; app uploaded (Docker, 46 files)
- [ ] Space is PAUSED: "Quota exceeded for flavor cpu-basic … limit=0" (account limit). Try a teammate's HF account, or add billing / paid CPU, then Restart (see `docs/DEPLOY.md`)
- [ ] Delete the stray public *model* repo `ngroborocks/CongessionalAppChallenge` (created by an upload without `--repo-type space`); keep the Space
- [ ] 3-5 human readers check the plain-language summaries; proofread `data/glossary.json`
- [ ] Fill `relevant_ncts` in `data/test_queries.json` so true hit@5 / MRR / recall@10 can be computed
- [ ] Click through the frontend in a real browser (keyboard only, phone width, print)
- [ ] Record the demo (3 minutes max)
- [ ] Write the technical-challenge answer (PDQ alignment + faithfulness)
- [ ] Confirm district + deadline, team registration and eligibility on congressionalappchallenge.us; submit

## Known small issues
- 333 trials that mention cancer but fit no type were dropped (conditions just "Cancer"). Could add a "solid tumors / any cancer" type.
- Inverted names like "Myeloid Leukemia, Philadelphia Positive" can still mis-tag (1 known trial).
- Incremental index refresh (`python -m pipeline.indexer --auto-since`) not scheduled yet (cut-list item 2).
