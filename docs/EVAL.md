# Evaluation

Real numbers only. Each section is filled in by `pipeline/evaluate.py` or by a human run; do not write estimates here.

## Retrieval (keyword vs meaning vs hybrid)

Measured by `python -m pipeline.evaluate` on the 25 lay queries in `data/test_queries.json`, inside the same candidate set `search()` uses (tagged with the query's expected types, status RECRUITING, no age/ZIP filter). Keyword = FTS5 bm25 top 50; meaning = FAISS/MiniLM top 50; hybrid = reciprocal rank fusion of the two.

**Proxy metrics (automatic, no human labels).** A result counts if its stored cancer tags include one of the query's expected types. This checks the ranking stays on the right cancer; it does NOT prove the best trial was found, so it is not hit@5. Because candidates are already restricted to trials tagged with the expected types (rule 1), these proxies are 1.0 whenever a ranking returns anything: they confirm the type filter works and that no query comes back empty, nothing more. Ranking quality is only measured by the true metrics below.

| system | type precision@10 (proxy) | type hit@5 (proxy) | queries with no results |
|---|---|---|---|
| keyword | 1.000 | 1.000 | 0 / 25 |
| meaning | 1.000 | 1.000 | 0 / 25 |
| hybrid | 1.000 | 1.000 | 0 / 25 |

**True metrics (need human `relevant_ncts`).**

| system | hit@5 | MRR | recall@10 | labelled queries |
|---|---|---|---|---|
| keyword | pending: relevant_ncts not filled in yet | - | - | 0 |
| meaning | pending: relevant_ncts not filled in yet | - | - | 0 |
| hybrid | pending: relevant_ncts not filled in yet | - | - | 0 |

## Off-topic redirect

- Off-topic queries redirected (no cancer type detected, chips returned): **10/10**
- On-topic recall (lay queries where any cancer type was detected): **1.000** of 25
- Lay queries where a detected type is one of the expected types: **1.000** of 25

## Summaries (ROUGE-L, reading level)

Copied from `results/baselines_results.csv` (held-out PDQ test split, produced once by `pipeline/baselines.py`; not recomputed here). Full model comparison is further down.

| system | ROUGE-L F1 | key-term P / R | support | boilerplate % | FK out (src 16.2) |
|---|---|---|---|---|---|
| copy-source | 0.205 | 1.000 / 1.000 | 1.000 | 0.0 | 16.2 |
| flan-t5-pdq-clean-sent | 0.240 | 1.000 / 0.237 | 0.919 | 14.0 | 12.4 |
| flan-t5-pdq-lr1e4-s70 | 0.231 | 1.000 / 0.197 | 0.804 | 27.2 | 12.1 |
| zero-shot-instruct | 0.204 | 0.985 / 0.390 | 0.979 | 0.0 | 15.5 |

## Faithfulness pass rate (batch summary pipeline)

See the "Production summaries" section (added at the end of this file) for the pass rate of the summaries the site actually serves. The earlier 200-trial run in this spot was made before `faithfulness_check` existed (every piece fell back by design), so it is not a result and was removed.

## Latency

Warm (one untimed pass first), 5 timed runs per query, milliseconds. A run = candidate filter + one ranking + building the result cards; RRF is arithmetic on at most 100 ids and is not included. With-ZIP rows use ZIP 10001 and a 100-mile radius. Measured on the developer PC (CPU search), so the host will differ.

| path | p50 ms | p95 ms | runs |
|---|---|---|---|
| keyword/no_zip | 41.1 | 156.3 | 125 |
| keyword/with_zip | 255.8 | 380.9 | 125 |
| meaning/no_zip | 17.0 | 41.2 | 125 |
| meaning/with_zip | 201.6 | 233.7 | 125 |

## Training experiments (fixed val set, 125 pairs)

All runs: flan-t5-base, batch 8 x accum 2, bf16, RTX 5090, seed 42, early stopping on each run's OWN val split.
Val losses of different runs use different data, so they are NOT compared directly; every checkpoint was re-scored on the same frozen set
`data/pdq_val_fixed.jsonl` (125 val pairs at similarity >= 0.70, taken before any data change) with `pipeline/compare_checkpoints.py`
(beam 4, no_repeat_ngram 3). The "test" split was not used for any choice.
Data was deduplicated before the runs below (paragraph pairs 1,649 -> 1,440; sentence pairs 1,556 -> 918; details in `data/pdq_stats.json` -> "dedupe").

| run | lr | min sim | train pairs | best epoch | fixed-set val loss | ROUGE-L F1 | FK grade out (src 15.8) | numbers faithful |
|---|---|---|---|---|---|---|---|---|
| flan-t5-pdq-baseline (old data, not deduped) | 3e-4 | 0.70 | 1,072 | 1 | 1.967 | 0.187 | 11.3 | 100% |
| flan-t5-pdq-lr1e4-s70 (paragraphs) | 1e-4 | 0.70 | 911 | 3 | 1.947 | 0.221 | 12.9 | 100% |
| flan-t5-pdq-lr1e4-s60 (paragraphs) | 1e-4 | 0.60 | 1,124 | 3 | 1.932 | 0.210 | 12.2 | 100% |
| flan-t5-pdq-sent (paragraphs + sentences) | 1e-4 | 0.70 | 1,492 | 3 | 1.914 | 0.215 | 12.3 | 100% |
| flan-t5-pdq-sent-lr5e5 (paragraphs + sentences) | 5e-5 | 0.70 | 1,492 | 5 | 1.920 | 0.210 | 12.5 | 100% |

Lower lr (1e-4) delayed overfitting from epoch 1 to epoch 3; train loss still falls much faster than val loss, so the data (not the lr) is the limit.
Leakage check: no fixed-set target or source text occurs verbatim in any train row, and no train page is a fixed-set page (one fixed-set target is a substring of / contains a train target).

## Model comparison (baselines vs fine-tuned, precision / recall / F1)

Produced by `pipeline/baselines.py` (`results/baselines_results.csv`, `results/raw_predictions_*.jsonl`, `results/error_analysis_*.txt`).
**Model selection used the val set and the trial set only. The test split was run once, afterwards, for all systems, and nothing was changed after seeing it.**
Key-term P = share of output numbers/units/stages/genes/drugs that are in the source (low = invented facts); R = share of source key terms kept.
Support = share of output sentences close in meaning to a source sentence AND made of source/glossary words (`backend/support.py`). Boilerplate = % outputs with a PDQ opener ("There are different types of ...") not in the source.

Trial-domain set (what the app actually does): 40 real trials from `data/trials.db` (35 cancer types, seed 42), 240 texts = 40 brief summaries + 200 eligibility criteria; no reference text.

| system | key-term P / R / F1 | support precision | fully supported | boilerplate | FK out (src 15.4) |
|---|---|---|---|---|---|
| copy-source | 1.000 / 1.000 / 1.000 | 1.000 | 100% | 0% | 15.4 |
| zero-shot flan-t5-base "simplify:" | 0.984 / 0.667 / 0.795 | 0.912 | 90.4% | 0% | 15.2 |
| zero-shot flan-t5-base instruction | 0.985 / 0.726 / 0.836 | 0.960 | 95.8% | 0% | 14.7 |
| few-shot k=3 (about 2.9 shots fit) | 0.960 / 0.786 / 0.865 | 0.890 | 90.8% | 0% | 15.6 |
| fine-tuned baseline (lr 3e-4, old data) | 1.000 / 0.477 / 0.646 | 0.730 | 72.5% | 37.9% | 10.7 |
| fine-tuned lr1e4-s70 | 0.996 / 0.582 / 0.735 | 0.826 | 81.7% | 10.4% | 11.6 |
| fine-tuned lr1e4-s60 | 0.996 / 0.488 / 0.655 | 0.799 | 79.2% | 13.3% | 11.3 |
| fine-tuned sent | 0.996 / 0.532 / 0.693 | 0.779 | 76.7% | 10.4% | 11.0 |
| fine-tuned sent-lr5e5 | 0.988 / 0.523 / 0.684 | 0.783 | 76.7% | 11.3% | 11.0 |

PDQ val (fixed, 125 pairs; reference = NCI patient text). ROUGE-1/2 P/R/F1 are in the CSV.

| system | ROUGE-L F1 | token P / R / F1 | key-term P / R / F1 | support | boilerplate | FK out (src 15.8) |
|---|---|---|---|---|---|---|
| copy-source | 0.198 | 0.193 / 0.428 / 0.266 | 1.000 / 1.000 / 1.000 | 1.000 | 0% | 15.8 |
| zero-shot "simplify:" | 0.195 | 0.246 / 0.273 / 0.259 | 0.991 / 0.411 / 0.581 | 0.987 | 0% | 16.6 |
| zero-shot instruction | 0.207 | 0.265 / 0.250 / 0.257 | 1.000 / 0.336 / 0.503 | 0.967 | 0% | 15.6 |
| few-shot k=3 | 0.194 | 0.218 / 0.306 / 0.255 | 1.000 / 0.463 / 0.632 | 0.929 | 0% | 16.0 |
| fine-tuned baseline | 0.187 | 0.316 / 0.166 / 0.217 | 1.000 / 0.166 / 0.285 | 0.587 | 60.0% | 11.3 |
| fine-tuned lr1e4-s70 | 0.221 | 0.337 / 0.217 / 0.264 | 1.000 / 0.249 / 0.399 | 0.851 | 17.6% | 12.9 |
| fine-tuned lr1e4-s60 | 0.210 | 0.336 / 0.199 / 0.250 | 1.000 / 0.209 / 0.346 | 0.787 | 20.0% | 12.2 |
| fine-tuned sent | 0.215 | 0.328 / 0.207 / 0.254 | 1.000 / 0.190 / 0.319 | 0.849 | 11.2% | 12.3 |
| fine-tuned sent-lr5e5 | 0.210 | 0.306 / 0.223 / 0.258 | 1.000 / 0.261 / 0.414 | 0.859 | 12.8% | 12.5 |

PDQ test (held-out, 136 pairs, run once). Columns: ROUGE-L F1 / token F1 / key-term P and R / support / boilerplate / FK out (src 16.2):
- copy-source 0.205 / 0.264 / 1.000, 1.000 / 1.000 / 0% / 16.2
- zero-shot simplify 0.210 / 0.268 / 1.000, 0.323 / 0.983 / 0% / 16.6
- zero-shot instruction 0.204 / 0.267 / 0.985, 0.390 / 0.979 / 0% / 15.5
- few-shot 0.200 / 0.271 / 0.981, 0.449 / 0.972 / 0% / 16.2
- fine-tuned baseline 0.201 / 0.235 / 1.000, 0.091 / 0.567 / 58.8% / 11.0
- lr1e4-s70 0.231 / 0.265 / 1.000, 0.197 / 0.804 / 16.2% / 12.1
- lr1e4-s60 0.239 / 0.276 / 1.000, 0.141 / 0.781 / 18.4% / 11.9
- sent 0.232 / 0.273 / 1.000, 0.170 / 0.796 / 16.2% / 12.3
- sent-lr5e5 0.233 / 0.274 / 1.000, 0.185 / 0.827 / 11.8% / 12.0

### Winner and honest reading
Winner: **flan-t5-pdq-lr1e4-s70** (copied to `checkpoints/flan-t5-pdq-best`). Rule: key-term precision first (all fine-tuned runs are within 0.01 of 1.0, so a tie), then trial-domain support and boilerplate, then ROUGE-L F1, then FK drop.
It has the best trial support precision (0.826) of the fine-tuned runs, the lowest boilerplate on trials (10.4%, tied), the best val ROUGE-L (0.221) and the highest key-term recall on trials (0.582).
It beat the old baseline on every faithfulness measure (trial support 0.730 -> 0.826, boilerplate 37.9% -> 10.4%). On test it is not the top ROUGE-L (lr1e4-s60 is, by 0.008); the choice was made before test was run.

What the numbers also say, plainly:
- Fine-tuning does simplify (FK about 11-13 vs 15.4-15.8 for the source) but every fine-tuned model is LESS faithful than copy-source and the zero-shot models on the trial set: key-term recall is 0.48-0.58 (it drops numbers, ages and criteria) and support precision is 0.78-0.83 versus 0.91-0.96 for zero-shot.
- Zero-shot flan-t5-base mostly copies, so it looks faithful but barely simplifies (FK 14.7-15.2). Few-shot is similar.
- Key-term precision is about 1.0 for all models because the number check is easy; the remaining risk is meaning drift, which the support metric catches. 10-18% of outputs still contain a PDQ-style opener that the source lacks.
- Mitigations to consider (not done): remove generic PDQ intro sentences ("There are different types of ...", "X is a type of cancer that forms ...") from training targets; train on sentence-level pairs only; use the model for brief summaries and `glossary_swap` for eligibility criteria; use `backend.support.sentence_support` as an extra automatic fallback trigger in summarize.py (fall back to the glossary-swapped original when any sentence is unsupported).

## Cleaned-target experiment (added later; supersedes the winner above)

Targets were cleaned by `pipeline/clean_pairs.py` (stats: `data/pdq_clean_stats.json`): generic PDQ intro/definition sentences were removed unless the source has them (101 sentences), and target sentences with MiniLM similarity < 0.45 to every source sentence window were removed (252 sentences; the run-time support thresholds 0.60 + 70% word overlap would have removed about two thirds of PDQ target sentences because patient wording differs from professional wording, so they were relaxed for cleaning). 89 pairs left too short were dropped, then everything was re-deduplicated (107 more). Result: 2,358 -> 2,162 pairs (train 1,626 / val 273 / test 263; paragraph-only 1,038 / 156 / 154). Runs: lr 1e-4, min sim 0.70; E = cleaned paragraphs only (846 train pairs, best epoch 3, own val loss 1.955), F = cleaned paragraphs + sentences (1,434 train pairs, best epoch 3, own val loss 1.941).

Note: the boilerplate pattern list was broadened by another change after the first comparison, and all systems were re-scored with it, so boilerplate values below are higher than in the tables above for the same checkpoints (e.g. lr1e4-s70 on trials: 10.4% -> 21.3%). Other metrics are unchanged.

Trial set (240 texts), re-scored with the current patterns:

| system | key-term P / R / F1 | support | fully supported | boilerplate | FK out (src 15.4) |
|---|---|---|---|---|---|
| fine-tuned lr1e4-s70 (previous winner) | 0.996 / 0.582 / 0.735 | 0.826 | 81.7% | 21.3% | 11.6 |
| **clean-sent (F, new winner)** | 0.992 / 0.534 / 0.694 | 0.843 | 82.9% | 13.8% | 11.3 |
| clean-s70 (E) | 0.988 / 0.516 / 0.678 | 0.822 | 81.7% | 15.8% | 11.4 |

Fixed val (125 pairs): s70 ROUGE-L F1 0.221, token 0.337/0.217/0.264, key-term 1.000/0.249/0.399, support 0.851, boilerplate 28.8%, FK 12.9; clean-sent 0.219, 0.307/0.238/0.268, 1.000/0.288/0.448, 0.901, 20.8%, 12.9; clean-s70 0.214, 0.317/0.214/0.256, 1.000/0.233/0.378, 0.913, 13.6%, 13.4.
Test (136, run once for the new winner only): clean-sent ROUGE-L F1 0.240, token 0.331/0.251/0.285, key-term 1.000/0.237/0.384, support 0.919, boilerplate 14.0%, FK 12.4 (s70 on the same patterns: 0.231, 0.337/0.218/0.265, 1.000/0.197/0.328, 0.804, 27.2%, 12.1).

Decision: key-term precision differs by 0.004 (about one term; treated as a tie, as before), then clean-sent has better support (0.843 vs 0.826) and much lower boilerplate (13.8% vs 21.3%), at the cost of lower key-term recall (0.534 vs 0.582). A strict reading of the rule (key-term precision first, no tolerance) would keep s70. `checkpoints/flan-t5-pdq-best` is now a copy of `flan-t5-pdq-clean-sent`. Cleaning helped on support and boilerplate but did not fix content dropping (recall about 0.5) or the remaining 14-16% of outputs with a generic opener.

## Bigger models and in-domain data (added later)

**New data.** `pipeline/ctgov_pairs.py` aligns each trial's brief-summary sentence (or 2-sentence window) to the best 1-3 consecutive
detailed-description sentences (MiniLM cosine >= 0.70) from ClinicalTrials.gov (public domain). Kept only if the target is >= 2.0 FK grades
easier, <= 1.5x as long, adds no number/marker/stage/drug, has no PDQ boilerplate; then deduplicated. Every trial-set NCT is excluded (25 of its 40 trials had both texts);
train/val split by NCT id. Of 115,295 candidates from 16,493 trials: low similarity 63,739; FK drop too small 21,422; near-identical
8,035; length ratio 5,925; new key term/number 3,164; length bounds 406; boilerplate 306; dedupe 4,425 -> **7,873 pairs** (7,143 train / 730 val,
5,307 trials). Training used PDQ-clean (1,434 train / 239 val) plus a random 2,000 train / 250 val ctgov sample (so PDQ is not drowned out):
3,434 train / 489 val. Caution: val loss in these runs is on the mixed val set and is not comparable with earlier PDQ-only runs.

**Runs** (lr 1e-4, early stopping; trial-set = 240 pieces; guard-pass = piece has no sentence flagged by support / boilerplate / risk / definition / direction):

| system | split | key-term P | R | F1 | support | boilerplate % | guard-pass % | FK out |
|---|---|---|---|---|---|---|---|---|
| flan-t5-pdq-clean-sent (previous best) | trials | 0.992 | 0.534 | 0.694 | 0.843 | 13.8 | 67.5 | 11.3 |
| G flan-t5-base pdq+ctgov | trials | 0.972 | 0.684 | 0.803 | 0.951 | 5.8 | 84.6 | 13.4 |
| H flan-t5-large pdq-clean | trials | 0.979 | 0.597 | 0.742 | 0.885 | 13.3 | 75.4 | 12.0 |
| I flan-t5-large pdq+ctgov | trials | 0.980 | 0.741 | 0.844 | 0.945 | 2.9 | 90.8 | 14.1 |
| J Qwen2.5-1.5B LoRA (r16) pdq+ctgov | trials | 0.580 | 0.174 | 0.268 | 0.235 | 10.0 | 18.8 | 12.4 |
| J2 Qwen2.5-1.5B LoRA (r64, a16, lr2e-4, b8 x 3 ep) | trials | 0.518 | 0.157 | 0.241 | 0.221 | 10.8 | 16.3 | 12.4 |
| previous best | val | 1.000 | 0.288 | 0.448 | 0.901 | 20.8 | 63.2 | 12.9 |
| G | val | 0.991 | 0.439 | 0.608 | 0.933 | 16.8 | 71.2 | 13.4 |
| H | val | 1.000 | 0.265 | 0.419 | 0.929 | 11.2 | 74.4 | 13.5 |
| I | val | 1.000 | 0.324 | 0.490 | 0.950 | 8.0 | 76.0 | 13.6 |
| J | val | 0.600 | 0.012 | 0.023 | 0.312 | 22.4 | 18.4 | 10.6 |
| J2 | val | 0.636 | 0.028 | 0.053 | 0.378 | 25.6 | 16.8 | 10.5 |

Reference: the source text reads at FK 15.4 (trials) / 15.8 (val); zero-shot-instruct has guard-pass 94.6% but FK 14.7.
J2 sweep (batch 8 x 1 vs 3 epochs) was chosen on val loss (1.4104 vs 1.4080), not on test. Both Qwen LoRA runs write generic PDQ-style
sentences that ignore the source (precision 0.5-0.6), so they are rejected; the cause was not investigated further.

**Winner: I (flan-t5-large on PDQ-clean + ctgov).** On val it ties on key-term precision (1.000), and is best on support (0.950), boilerplate
(8.0%), recall and guard-pass. On the trial set its key-term precision is 0.980 vs 0.992 for the previous best (a small drop), but support is
0.945 vs 0.843, boilerplate 2.9% vs 13.8%, and 91% of pieces pass the guards vs 68%. Trade-off: it simplifies less (FK 14.1 vs 11.3 on trials;
the ctgov targets are close in wording to their sources). Test split, run once for I only: key-term P 1.000 / R 0.308 / F1 0.471, support 0.905,
boilerplate 7.4%, FK 13.6, ROUGE-L 0.228.
Inference cost on CPU (12 threads, batch 1, 4 beams, trial pieces): previous best 1.5 s/piece (~9 s per trial of 6 pieces); I 4.9 s/piece
(~30 s per trial). Summaries are pre-generated, so this only matters for the live fallback. G (flan-t5-base, same cost as the previous best) is
the cheap alternative: guard-pass 84.6%, key-term P 0.972.
Copied to `checkpoints/candidate-best` (not `flan-t5-pdq-best`).


### 8B models (zero-shot, added 2026-10-24)

Qwen/Qwen2.5-7B-Instruct and Qwen/Qwen3-8B (both Apache-2.0 on their model cards), bf16 on the RTX 5090, greedy, max 256 new tokens,
Qwen3 thinking off (`enable_thinking=False`). System prompt: "You rewrite clinical-trial text for patients at a 6th-8th grade reading level. Keep every
number, drug name, gene/marker, stage and requirement exactly. Do not add facts, definitions, or advice. Output only the rewrite." Systems
`zero-shot-qwen2.5-7b` / `zero-shot-qwen3-8b` in `pipeline/baselines.py` (opt-in via `--systems`). All rows below were re-scored with the
broadened guards in `backend/support.py` (file dated 2026-10-24 22:24), so guard-pass numbers differ slightly from the table above.
"Effective grade" = guard-pass x FK out + (1 - guard-pass) x FK source (failed pieces show the original text).

| system (trials) | key-term P | R | F1 | support | boilerplate % | guard-pass % | FK out | effective grade |
|---|---|---|---|---|---|---|---|---|
| copy-source | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 100 | 15.4 | 15.4 |
| zero-shot-instruct (flan-t5-base) | 0.985 | 0.726 | 0.836 | 0.960 | 0.4 | 93.3 | 14.7 | 14.7 |
| G flan-t5-base pdq+ctgov | 0.972 | 0.684 | 0.803 | 0.951 | 5.8 | 81.7 | 13.4 | 13.8 |
| I flan-t5-large pdq+ctgov | 0.980 | 0.741 | 0.844 | 0.945 | 2.9 | 90.8 | 14.1 | 14.2 |
| flan-t5-pdq-clean-sent | 0.992 | 0.534 | 0.694 | 0.843 | 13.8 | 55.8 | 11.3 | 13.1 |
| zero-shot Qwen2.5-7B | 0.923 | 0.832 | 0.875 | 0.423 | 1.7 | 39.6 | 9.2 | 13.0 |
| zero-shot Qwen3-8B | 0.995 | 0.917 | 0.955 | 0.551 | 5.8 | 39.6 | 10.1 | 13.3 |

Val (125 pieces): Qwen2.5-7B P 0.845 / R 0.688 / support 0.291 / guard-pass 7.2% / FK 8.6; Qwen3-8B P 0.991 / R 0.866 / support 0.459 /
guard-pass 11.2% / FK 10.4 (I: 1.000 / 0.324 / 0.950 / 67.2% / 13.6).

Reading: Qwen3-8B keeps far more key terms (recall 0.92) and simplifies a lot, but the support check rejects ~60% of pieces: it paraphrases and
reorders sentences, and sometimes rewrites a term ("metastatic" -> "stage IV", an invented key term). Gate for LoRA fine-tuning
(key-term P >= 0.97 AND support >= 0.90 on trials) is NOT met (Qwen3 support 0.55, Qwen2.5 P 0.92), so no LoRA run was done.
Cost (HF transformers, batch 8, greedy, 5090): trials 240 pieces in 80 s (Qwen2.5-7B, 3.0 pieces/s) and 140 s (Qwen3-8B, 1.7 pieces/s);
414k pieces would take ~38 h and ~68 h. vLLM (WSL `qwen-infer`) would likely be many times faster; not set up.

## Production summaries (switch to G, flan-t5-base on PDQ-clean + ClinicalTrials.gov)

The shipped summaries now come from G (`checkpoints/flan-t5-pdq-best` is a copy of `flan-t5-base-pdq-ctgov`; model_version
`flan-t5-pdq-best@b34a546a-4cecc6`, the suffix is a hash of train_metadata.json so two models in one folder never collide). The previous
model (clean-sent) is kept in `checkpoints/flan-t5-pdq-best-prev-clean-sent`, its stats in `data/batch_summary_stats_clean-sent.json`, and its raw
outputs stay in `model_outputs` under `flan-t5-pdq-best@0d5c3292`. Same scope for both: 20,432 trials, 413,998 pieces. Generation of G took
1h53m on the RTX 5090; the check stage took about 14 min on the RTX 4070 Ti (24-30 trials/s, versus about 4 trials/s on CPU).
Guards were widened for G (config.CLAIM_PATTERNS plus a general "X is a/an/the Y" rule in `support.adds_invented_claim`) after review found invented
definitions such as "PMBCL ... is a rare type of cancer"; 3,678 of G's raw sentences in a 1,500-trial sample trip it.

| | clean-sent | G |
|---|---|---|
| pieces shown as model text | 61.7% | 67.1% |
| fell back: faithfulness_check / guards / empty | 21,988 / 136,549 / 1 | 63,732 / 72,398 / 1 |
| trials with no fallback at all | 325 (1.6%) | 756 (3.7%) |
| effective reading grade shown (FK, 1,000 random trials, seed 42) | 13.86 | 14.54 |

The source text of those trials reads at FK 20.09. The effective grade is the mean of `reading_after`, which is computed over the text actually displayed
(model text where shown, glossary-swapped original where a piece fell back); G shows more model text but its text is a little harder than clean-sent's.
Function: `pipeline/batch_summarize.py::effective_reading_grade` (CLI: `batch_summarize reading`). Note the clean-sent figure was measured on the
summaries table before it was rewritten.

Examples from G (no fallback): "This study aims to evaluate the abnormalities of CASP3 gene in chemo resistance in breast cancer by FISH technique.";
"The study will monitor the long-term safety and tolerability of participants from AstraZeneca for up to 15 years post last cell or gene therapy treatment.";
"The ANALLISA study is a fast, proof-of-concept, phase II clinical trial which aims to assess the efficacy of niraparib rechallenge treatment after
secondary cytoreductive surgery in ovarian cancer patients ...". Known weakness: the symbol ">=" is lost by the tokenizer ("Female patients  18 years of age").

Comparator guard (added last). The T5 tokenizer cannot write the symbols >=, <=, < and >, so ">= 18 years" came out as "18 years" and the requirement lost its
direction. `support.direction_check` now fails any output that repeats a number the source bounds (symbols or words) without a bound of the same
direction (a range with both ends kept is fine); faithfulness_check calls the same function. Effect on G: faithfulness fallbacks rose from 17,802 to 63,732 pieces
(+45,930), and model text fell from 78.3% to 67.6% (67.1% after the last claim-phrase round: "may require", "may be done/needed", "you may have", "it may", "is very long", and an acronym defined as a thing; still above clean-sent's 61.7%). For future generation `summarize.model_input_text` spells the symbols out
("at least", "at most", "more than", "less than") in the model prompt only; the stored source and the checks keep the original. G's current outputs were NOT regenerated.
