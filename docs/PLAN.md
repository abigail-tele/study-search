# Implementation Plan — Cancer Trials Plain-Language Finder

Today: Fri Oct 23, 2026. Code freeze: Sun Oct 25, 6 pm. Submit: Sun night (buffer: Mon morning, if district deadline is Mon Oct 26).

## 0. Human checklist (before code)
- [ ] Confirm district + deadline on congressionalappchallenge.us
- [ ] Confirm every member is a middle/high school student, ≥ half live or attend school in the district, and no member has another entry this year (one app per student)
- [ ] Register team on the portal
- [ ] Decide where training runs (home GPU or Colab T4) and who runs it

## 1. User journey (acceptance spec for the whole app)
1. Landing: search bar, cancer chips (from `/api/cancer-types`), ZIP + radius (25/50/100/250 mi), status (default Recruiting), age, disclaimer.
2. Query in plain words or tap a chip.
3. Backend detects cancer type(s). None → "We only cover cancer trials" + chips.
4. Hybrid search inside detected types, filters applied, top 10.
5. Citation cards: title, NCT ID, status badge (word + color), phase, ≤3 cancer tags, nearest site + distance (if ZIP), sponsor, last updated, "View on ClinicalTrials.gov" (`https://clinicaltrials.gov/study/{NCT}`, new tab), "Copy citation", "Explain this study".
6. Summary panel: Study at a glance (API fields) · What this study is testing (model) · What they're looking for / Who can't join (criteria; highlighted if "About you" was filled in) · Key facts (API fields) · Original text (collapsible) · reading level before→after · faithfulness check mark or "showing original wording".
7. Copy citation, print view, fixed "Questions to ask your doctor" template.
8. Back to results with filters kept.

Citation string:
`{Sponsor}. {Official title}. ClinicalTrials.gov ID: {NCT}. Last updated {YYYY-MM-DD}. Available from: https://clinicaltrials.gov/study/{NCT}`

## 2. API contract
| Endpoint | Params | Returns |
|---|---|---|
| GET /api/cancer-types | — | `[{id, label, lay_terms[]}]` |
| GET /api/search | q, cancer?, zip?, radius_mi?, status?, age?, sex? (female/male/any), conditions? (<=600 chars), sort? (relevance/match) | `{query, detected_types[], off_topic, suggested_types[{id,label}], results[]}` — `suggested_types` = cancer chips, filled when off_topic (rule 1), else `[]` |
| GET /api/study/{nct} | age?, sex?, conditions?, zip? | key facts object + `overview` + `criteria` + `match` (null without a profile) |
| GET /api/summary/{nct} | — | summary object |

Search result item: `nct_id, title, status, phase, cancer_tags[], sponsor, last_updated, nearest_site{city,state,distance_mi}|null, url, citation, score`
"About you" (optional, never stored or logged, rule 8; query params only, never in the URL hash): when age, sex or conditions is given, each search result gets `match` and `sort=match` lists "Possible match" first (relevance order kept inside each group).
`match`: `{age: ok|outside|unknown, sex: ok|no|unknown, mentioned_in_requirements[{term,criterion}], mentioned_in_exclusions[{term,criterion}], possible_conflicts[{term,criterion}], ignored_terms[], summary_label: "Possible match"|"Check with study team"|"Likely not a match", label_text}`. Never says "eligible"; the study team decides.
`overview` (study panel top, stored API fields only, rule 2): `{sentence, facts[{label,value}]}`. `criteria`: `{include[], exclude[]}` of `{text (original + glossary), original, flag: match|concern|null, terms[]}`.
Summary object: `nct_id, what_is_tested, who_can_join{include[],exclude[]}, original{summary,eligibility}, faithful, fallback_used, reading_level{before,after}, model_version, generated_at`
Errors: `{"error": "..."}` + HTTP status. Off-topic is not an error.

## 3. Phases, owners, done-checks
Parallel lanes run at the same time. Gate = test-runner + reviewer pass.

### Phase 1 — Fri Oct 23: data + skeleton (parallel)
| Task | Agent | Done when |
|---|---|---|
| Repo scaffold, requirements.txt, config.py, CREDITS.md, AI_USE.md, README stub | grunt | tree matches CLAUDE.md layout |
| Mock JSON for all 4 endpoints (`frontend/mock/`) | grunt | valid against contract above |
| `data/cancer_types.json`: ~25 types, lay + clinical terms (NCI Dictionary/Thesaurus wording) | grunt, then reviewer spot-check | ≥25 types, each ≥3 lay + ≥3 clinical terms |
| Verify CT.gov v2 params/fields on https://clinicaltrials.gov/data-api/api (condition query, overallStatus filter, pageSize ≤1000, pageToken, fields, countTotal) | scout | ≤15-line note in docs/PLAN.md §6 |
| `db.py` schema + `pipeline/indexer.py` (dedupe by NCT, tags, locations w/ lat/lon, FTS5, incremental mode) | data-engineer | 3 cancer types indexed; counts logged |
| `pipeline/pdq_pairs.py`: scrape professional + patient PDQ pages per cancer type, match sections, align paragraphs by embedding similarity, clean, split by page 80/10/10 | ml-engineer | `pdq_pairs.jsonl` + stats (pairs kept/dropped per step) |
| Frontend: search page + cards against mock JSON | api-builder | renders mock results, keyboard usable |
| ZIP centroid file (Census gazetteer) + `geo.py` | grunt → data-engineer | haversine tests pass |
**Gate 1:** full indexer run for all types started (can run overnight).

### Phase 2 — Sat Oct 24: search + model (parallel)
| Task | Agent | Done when |
|---|---|---|
| `pipeline/embed.py` + FAISS index | data-engineer | index built, row map matches DB |
| `search.py`: candidates by tag, FTS5 top50, FAISS top50, RRF (HUMAN-WRITTEN), filters, citation builder | data-engineer + human | 25-query test set runs |
| `taxonomy.detect_cancer_types` (HUMAN-WRITTEN; exact/fuzzy → embedding threshold) | human, tests by data-engineer | off-topic set 10/10 redirected |
| `data/test_queries.json`: 25 lay queries + 10 off-topic | grunt drafts, human picks correct NCTs | file complete |
| `pipeline/train.py` (flan-t5-base, prefix `simplify: `, in 512/out 256, lr 3e-4, bs 8–16, 3–5 epochs, early stop) | ml-engineer writes; HUMAN runs on GPU | checkpoint + val loss curve |
| `summarize.py`: chunking, beam 4, no_repeat_ngram 3, include/exclude split, faithfulness_check (HUMAN-WRITTEN) | ml-engineer + human | unit tests pass |
| FastAPI endpoints wired to real search | api-builder | `/api/search` live locally |
Carry-overs from Gate 1 review (2026-10-24):
- Rule 8: uvicorn's default access log records `/api/search?q=…&zip=…` → app.py must run with access_log off or a filter that strips query strings.
- Type detection must use `data/umbrella_terms.json` (lung/blood/skin cancer, leukemia, lymphoma → several types) and must never match the bare word "all".
- Reading level: textstat pulls in pyphen (GPL/LGPL/MPL-1.1 tri-license) → human decides: use pyphen under MPL-1.1, or our own Flesch-Kincaid.
**Gate 2:** reviewer checks rules 1–4 and the faithfulness logic.

### Phase 3 — Sun Oct 25: batch, deploy, eval, submit
| Task | Agent | Done when |
|---|---|---|
| `pipeline/batch_summarize.py` over all indexed trials | ml-engineer writes; human runs on GPU | summaries table filled; pass rate logged |
| `/api/summary` + frontend summary panel on live data | api-builder | full journey works locally |
| `pipeline/evaluate.py` → `docs/EVAL.md`: hit@5 hybrid vs keyword, off-topic 10/10, ROUGE-L + reading level on PDQ test split, faithfulness pass rate | ml-engineer | EVAL.md filled with real numbers |
| Deploy (Dockerfile; ship trials.db, faiss.index, no model if memory-tight) | api-builder | public URL works on phone |
| HOW_IT_WORKS.md, CREDITS.md, AI_USE.md final | grunt; humans edit | complete and accurate |
| Human: 3–5 people read summaries; record demo (≤3 min); write technical-challenge answer (PDQ alignment + faithfulness) | humans | submitted |
**Gate 3:** reviewer full pass; code freeze 6 pm.

## 4. Cut list (in order)
1. Embedding model A/B  2. nightly refresh  3. age filter  4. live generation on cache miss (show original instead)  5. human check down to 3 people
Never cut: cancer restriction, citation links, faithfulness check, disclosure files.

## 5. Risks
| Risk | Fallback |
|---|---|
| <1,000 PDQ pairs | loosen threshold, add pages, report honestly |
| Low faithfulness pass rate | glossary version default; model behind "Simplify more" |
| API limits/downtime | everything pre-indexed; app never calls API live |
| Host memory | serve pre-generated summaries only |
| Wrong type detection | user can tap a chip to override |

## 6. Verified API notes
Checked 2026-10-24 against live API (apiVersion 2.0.5) + OAS at https://clinicaltrials.gov/api/oas/v2. Base: `https://clinicaltrials.gov/api/v2/studies`.
- Condition: `query.cond=lung cancer OR breast cancer` (OR works). Status: `filter.overallStatus=RECRUITING,NOT_YET_RECRUITING` (comma-separated; enum e.g. RECRUITING, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION, COMPLETED, TERMINATED).
- Paging: `pageSize=1000` accepted; response `nextPageToken` → pass as `pageToken`; absent on last page. `countTotal=true` → `totalCount`.
- `fields=NCTId,BriefTitle,...` (short names in), but response is nested `protocolSection.*Module`:
  - identificationModule.{nctId, briefTitle, officialTitle}; statusModule.{overallStatus, startDateStruct.date, lastUpdatePostDateStruct.date}
  - designModule.{phases[] e.g. ["PHASE2"], enrollmentInfo.count}; conditionsModule.{conditions[], keywords[]*}; sponsorCollaboratorsModule.leadSponsor.name
  - descriptionModule.{briefSummary, detailedDescription*}; eligibilityModule.{eligibilityCriteria, healthyVolunteers, sex "ALL", minimumAge "18 Years", maximumAge* (absent if none), stdAges}
  - contactsLocationsModule.locations[].{facility, city, state, zip, country, status*, geoPoint.lat/lon}
  - `*` = not seen in samples; treat as optional.
- Incremental: `filter.advanced=AREA[LastUpdatePostDate]RANGE[2026-10-01,MAX]` works (URL-encode brackets).
- Ages are strings ("18 Years"), dates "YYYY-MM" or "YYYY-MM-DD". No documented rate limit → sleep between pages, back off on 429/5xx. Browser-like User-Agent used; no 403 seen.
