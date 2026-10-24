# How It Works

Plain-English tour of every module. Each section should be 3 to 5 sentences that a team member can explain to a judge.

## config.py

Central settings for the whole project. Every path, threshold, k value, and UI string lives in `backend/config.py`, so tuning happens in one place and every "magic number" is visible at once. Paths are built from the project root, so the app runs the same way on any computer. Search settings (how many candidates each method returns, the RRF constant, and the thresholds for detecting cancer types) and PDQ and model settings each have their own section. Other modules import these names instead of hard-coding values.

## app.py

The web server only reads files already on disk (SQLite, the FAISS index and the local MiniLM model in `models/`); it never calls ClinicalTrials.gov or Hugging Face while a person is waiting. When it starts, it loads the model and index once so the first visitor is fast and any missing file shows up immediately instead of mid-search. Hugging Face offline mode is switched on in code and in `run_api.ps1` as a second safety lock. Search text and ZIP codes are scrubbed from logs. Run `python -m pipeline.fetch_models` once before the first start.

## embedding.py

This small module is the single place that turns text into 384 numbers with all-MiniLM-L6-v2, so trials (offline) and questions (live) share one vector space. It loads the model only from the local `models/all-MiniLM-L6-v2` folder and raises a clear error if it is missing, rather than quietly downloading it during a request.

## db.py

db.py defines our database, a single SQLite file that holds every trial, the cancer types it was matched to, its hospital sites with map coordinates, a keyword-search index, and the saved plain-language summaries. SQLite needs no server, so the website can simply read the file. A helper turns ages like "6 Months" or "18 Years" into one number of years so we can filter by age later. When a trial is saved again, its old row is replaced so the database always matches the newest version from ClinicalTrials.gov.

## taxonomy.py

This file holds the list of cancer types (breast, lung, leukemia and so on) and broad phrases like "blood cancer" that stand for several types at once. Its main job is `detect_cancer_types`, which reads what a patient typed and decides which cancer type they mean, combining exact wording and broad phrases (so "breast and lung cancer" finds breast plus both lung types), and only if neither matches trying close spellings ("brest") and last a meaning comparison. Very short names like AML or CLL only count as whole words, so "small" or "camel" never trigger a leukemia. If nothing matches, the question is not about cancer and the app shows cancer buttons instead of searching, which is how we keep to the cancer-only rule. The detection function is written by a team member, so this section is still theirs to fill in.

## search.py

Search first narrows the database to trials tagged with every cancer type found (a query naming two cancers keeps trials of both), and to the status, age and distance the patient chose, so every ranked result is usable. Inside that group it makes two ranked lists: one from keyword matching (SQLite FTS5) and one from meaning matching (FAISS over MiniLM vectors, which finds trials that say the same thing in different words). Reciprocal rank fusion then merges the two lists by giving each trial 1/(60 + position) points from each list, so trials that both methods like rise to the top. Finally the top 10 become cards, each with a citation built only from stored ClinicalTrials.gov fields and, if a ZIP was given, the distance to the nearest site. Nothing here calls the internet, and the query and ZIP are never saved or logged.

## summarize.py

summarize.py turns a trial's brief summary and eligibility text into plain language. It first splits the eligibility text into "can join" and "cannot join" lists (so the two are never mixed up), then cuts long text into whole-sentence chunks that fit the small model's input limit. The fine-tuned flan-t5 model rewrites each chunk, and every rewritten chunk must pass a faithfulness check: any number, age, unit, drug, gene or stage term in the rewrite has to appear in the original. If that check ever crashes, only that chunk falls back (the error type and trial id are logged, never the text). After that check, automatic sentence-level guards from backend/support.py run: a meaning-and-word-overlap check against the source, a generic-opener filter, and a direction check that catches flipped comparisons or negations. Sentences caught by any guard are dropped on their own as long as at least half of the rewrite survives; otherwise the whole chunk falls back. A chunk that fails is replaced by the original wording with hard words explained from a short glossary based on the NCI Dictionary of Cancer Terms. The result has two flags: `fallback_used` is true if any part fell back to original wording, and `faithful` is true only when nothing fell back, meaning everything shown is checked model text. Until the team writes the faithfulness check, every chunk fails on purpose, so patients only ever see checked or original text. Only the summary and eligibility text are rewritten; titles, ages, sites and dates always come straight from ClinicalTrials.gov, and nothing about the visitor is stored.

## support.py

support.py holds the automatic safety nets that run after the model rewrites a chunk, on top of the team's number-and-drug faithfulness check. The support check compares each rewritten sentence with the source using sentence embeddings (MiniLM) and also requires most of its words to come from the source or the NCI glossary, so a fluent but invented sentence like "Women over 18 are more likely to get breast cancer" is dropped. Further filters drop generic textbook openers, "X is a type of X" sentences, risk words the trial never used ("likely", "cure"), and definitions the source did not give. The direction check looks for rules turned into their opposite: a number whose bound flips ("at least 18" becoming "under 18", ">= 2" becoming "<= 2"), and a statement whose negation flips ("no prior chemotherapy" becoming "prior chemotherapy", "excluded" becoming "included"). It does this by comparing short clauses that share most of their topic words; when the wording differs too much to compare, it stays quiet instead of guessing, so it is a cheap tripwire rather than a proof. All thresholds and word lists live in config.py.

## readability.py

readability.py estimates how hard a text is to read, using the Flesch-Kincaid grade formula: longer sentences and longer words raise the grade. We wrote the short formula ourselves instead of adding a library, with a simple vowel-group syllable counter, which is accurate enough to compare the original text with our rewrite but not to publish as an exact grade. summarize.py stores a "before" and an "after" grade with every summary so the app can show that the plain-language version is easier to read.

## geo.py

geo.py answers "how far is this trial from me?". It looks up the center point of a ZIP code in a public-domain US Census table, then uses the haversine formula (straight-line distance over the curved Earth) to measure miles to each trial site. It picks the nearest site and reports its city, state, and distance. These are plain functions that keep nothing, so a visitor's ZIP code is never stored.

## zip_centroids.py

zip_centroids.py builds `data/zip_centroids.csv`, the table geo.py uses to turn a ZIP code into a map point. It downloads the US Census Bureau's public-domain ZIP-area (ZCTA) gazetteer once, keeps only the ZIP, latitude and longitude columns, and writes a small CSV. It runs by hand, not while the website is running, so no visitor ever triggers a download and no ZIP code is stored.

## indexer.py

indexer.py is the offline script that fills the database; the live website never talks to ClinicalTrials.gov. For each cancer type it sends that type's search phrase to the ClinicalTrials.gov API, asks only for the fields we need, and walks through the results page by page, pausing between pages and retrying with longer waits if the server is busy. We keep only open or ongoing trials, and a trial that matches two cancer types is stored once with two tags. A later run can fetch just trials changed since a given date to keep the data fresh; that run asks for trials of every status, so a trial that has since closed is noticed and deleted instead of staying "Recruiting" forever.

ClinicalTrials.gov's search is loose ("small cell" also finds "non-small cell", and "cervical" also finds the neck), so we do not trust its match. Before saving, the indexer re-reads each trial's conditions, keywords and titles and tags it with a cancer type only if one of that type's own words appears as a whole word. Each type can list `exclude_patterns` in `data/cancer_types.json` (for example "non-small cell" for small cell lung cancer, "cervical spine" for cervical cancer) that are blanked out first. A trial must also pass a general cancer check (words like cancer, tumor, leukemia, lymphoma, or acronyms like NSCLC and AML); anything else, such as PCOS or burn studies that slipped in, is never stored. If a trial is clearly cancer but names no type, it keeps the type of the search that found it, unless an exclusion pattern fired. The command `python -m pipeline.indexer --retag` applies these same rules to the existing database without downloading anything, and writes every removed trial to `data/retag_dropped.csv` so a person can spot-check it.

## embed.py

This offline script turns every trial into a list of 384 numbers that captures its meaning, using the small open model all-MiniLM-L6-v2. The text we feed it is the title, conditions, keywords and brief summary, because those say what the trial is about while the long eligibility list would add noise. The vectors go into a FAISS index (a fast lookup file) and a companion file that says which trial each row belongs to; the script checks the two counts match. Run it again whenever the indexer adds trials: `python -m pipeline.embed`.

## pdq_pairs.py

The National Cancer Institute publishes many cancer summaries twice: once for doctors and once, in simpler words, for patients, and both are public domain. This script downloads both versions of each page (treatment, screening, prevention and supportive care, listed in `data/pdq_pages.json`), cuts them into sections, and pairs up sections whose headings match (for example "Treatment of Stage III Colon Cancer" on both pages). Inside each matched section it turns every paragraph into a list of numbers that captures its meaning and pairs each patient paragraph with the doctor text that means the most similar thing; the doctor side may be one to three paragraphs joined, because one patient paragraph often condenses several. A second pass then looks across the whole page for leftover patient paragraphs, using a stricter similarity bar because the headings no longer back the match up. Weak matches, copied text, answers much longer than the original, and pairs where the patient text mentions a number or stage that the doctor text never mentions are thrown away (so the model is not taught to invent facts), and patient sentences that just point elsewhere ("see the booklet...") are cut out first. Finally, whole organ families (colon with rectal, the two lung cancers, Hodgkin with non-Hodgkin lymphoma, adult with child versions) are assigned together to the training, validation, or test group, and any validation or test example whose answer is almost a copy of a training answer is removed, so the model is never tested on text it effectively studied. The result is the file of before-and-after examples used to teach our small model to simplify medical text.

## train.py

This script teaches a small existing language model (Google's flan-t5-base) to do one job: take a paragraph written for doctors and rewrite it the way the National Cancer Institute writes it for patients. It learns from the before-and-after pairs that `pdq_pairs.py` made, keeping only pairs whose similarity score is high enough (0.60 by default) so it is not taught to say things the original never said. Training repeats the same loop many times: show the model a batch of doctor paragraphs, measure how far its rewrite is from the real patient version (the "loss"), and nudge its numbers to shrink that error. After each full pass it checks the loss on separate validation pairs, saves the best version so far, and stops early if the model has stopped improving, which prevents it from just memorising the training pairs. The test pairs are never touched, so the final evaluation stays fair.

## batch_summarize.py

Not built yet (Phase 3). This offline script will run the fine-tuned model over every stored trial ahead of time, so patients get a summary instantly instead of waiting for the model. Each rewrite will go through the same faithfulness check as live summaries, and the result (or the safe fallback text) will be saved in the database's summaries table. The live website only reads these saved rows, and falls back to on-the-spot generation if one is missing.

## evaluate.py

Not built yet (Phase 3). This script will grade the final model on the held-out test pairs that training never saw, using scores for readability, overlap with the NCI patient text, and how often the faithfulness check passes. It is run once at the end so the numbers are honest, and the results go into `docs/EVAL.md`.

## frontend

The frontend is one plain HTML page with CSS and JavaScript, with no build step and no outside libraries. It reads data through `api.js`, which has one flag (`USE_MOCK`) that switches between fake files in `frontend/mock/` and the real `/api/...` endpoints. `app.js` draws the search form, result cards and summary panel, and it only inserts API text with `textContent` so data can never run as code. The query and filters are kept in the page address (never the ZIP code), and nothing is saved in cookies or local storage. Colors always come with words (for example the status badge), text is at least 16px, and every control works with a keyboard.

## data/umbrella_terms.json (note for type detection)

Broad phrases such as "lung cancer", "skin cancer" or "blood cancer" are not single cancer types, so they live in `data/umbrella_terms.json` (phrase to list of type ids) instead of any one type's lay terms. Detection must show all the listed types for these, and must never match the bare word "all" (the lay term "all leukemia" only counts as that whole phrase).

## baselines.py

baselines.py answers "is our fine-tuned model actually better than simple alternatives?" It runs the same examples through easy comparisons (copying the source unchanged, the untrained flan-t5 with a simple prompt, the untrained model shown three examples) and through each of our trained checkpoints. Every output gets scored for how close it is to the NCI patient wording (ROUGE and word-overlap precision, recall and F1) and for key terms like numbers, drugs and stages, where low precision means the model invented facts. The scores are written to a results table and error-analysis files so we can show judges the evidence.

## clean_pairs.py

The first models we trained picked up textbook habits such as "X is a disease in which..." and wrote them even when a trial never said so. clean_pairs.py goes through every training answer sentence by sentence and deletes the ones that are generic openers or that the source text does not support, using the same support check the live app uses (with looser limits, since patient and doctor wording differ). If too little of an answer is left the whole pair is dropped, and duplicates are removed again. The original files are never overwritten, so the cleaning can be re-run or undone.

## compare_checkpoints.py

Each training run splits its own validation data, so their loss numbers can't be compared fairly. This script freezes one fixed set of 125 validation pairs and scores every checkpoint on it the same way: loss, ROUGE-L against the NCI patient text, reading grade level (lower is easier), and the share of outputs whose numbers all appear in the source. That lets us pick the best checkpoint by facts instead of by feel. It never touches the test split, which stays reserved for the final evaluation.
