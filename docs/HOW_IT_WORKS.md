# How It Works

Plain-English tour of every module. Each section should be 3 to 5 sentences that a team member can explain to a judge.

## config.py

Central settings for the whole project. Every path, threshold, k value, and UI string lives in `backend/config.py`, so tuning happens in one place and every "magic number" is visible at once. Paths are built from the project root, so the app runs the same way on any computer. Search settings (how many candidates each method returns, the RRF constant, and the thresholds for detecting cancer types) and PDQ and model settings each have their own section. Other modules import these names instead of hard-coding values.

## app.py

TODO: 3–5 plain-English sentences.

## db.py

db.py defines our database, a single SQLite file that holds every trial, the cancer types it was matched to, its hospital sites with map coordinates, a keyword-search index, and the saved plain-language summaries. SQLite needs no server, so the website can simply read the file. A helper turns ages like "6 Months" or "18 Years" into one number of years so we can filter by age later. When a trial is saved again, its old row is replaced so the database always matches the newest version from ClinicalTrials.gov.

## taxonomy.py

TODO: 3–5 plain-English sentences.

## search.py

TODO: 3–5 plain-English sentences.

## summarize.py

TODO: 3–5 plain-English sentences.

## geo.py

geo.py answers "how far is this trial from me?". It looks up the center point of a ZIP code in a public-domain US Census table, then uses the haversine formula (straight-line distance over the curved Earth) to measure miles to each trial site. It picks the nearest site and reports its city, state, and distance. These are plain functions that keep nothing, so a visitor's ZIP code is never stored.

## zip_centroids.py

zip_centroids.py builds `data/zip_centroids.csv`, the table geo.py uses to turn a ZIP code into a map point. It downloads the US Census Bureau's public-domain ZIP-area (ZCTA) gazetteer once, keeps only the ZIP, latitude and longitude columns, and writes a small CSV. It runs by hand, not while the website is running, so no visitor ever triggers a download and no ZIP code is stored.

## indexer.py

indexer.py is the offline script that fills the database; the live website never talks to ClinicalTrials.gov. For each cancer type it sends that type's search phrase to the ClinicalTrials.gov API, asks only for the fields we need, and walks through the results page by page, pausing between pages and retrying with longer waits if the server is busy. We keep only open or ongoing trials, and a trial that matches two cancer types is stored once with two tags. A later run can fetch just trials changed since a given date to keep the data fresh; that run asks for trials of every status, so a trial that has since closed is noticed and deleted instead of staying "Recruiting" forever.

ClinicalTrials.gov's search is loose ("small cell" also finds "non-small cell", and "cervical" also finds the neck), so we do not trust its match. Before saving, the indexer re-reads each trial's conditions, keywords and titles and tags it with a cancer type only if one of that type's own words appears as a whole word. Each type can list `exclude_patterns` in `data/cancer_types.json` (for example "non-small cell" for small cell lung cancer, "cervical spine" for cervical cancer) that are blanked out first. A trial must also pass a general cancer check (words like cancer, tumor, leukemia, lymphoma, or acronyms like NSCLC and AML); anything else, such as PCOS or burn studies that slipped in, is never stored. If a trial is clearly cancer but names no type, it keeps the type of the search that found it, unless an exclusion pattern fired. The command `python -m pipeline.indexer --retag` applies these same rules to the existing database without downloading anything, and writes every removed trial to `data/retag_dropped.csv` so a person can spot-check it.

## embed.py

TODO: 3–5 plain-English sentences.

## pdq_pairs.py

The National Cancer Institute publishes many cancer summaries twice: once for doctors and once, in simpler words, for patients, and both are public domain. This script downloads both versions of each page (treatment, screening, prevention and supportive care, listed in `data/pdq_pages.json`), cuts them into sections, and pairs up sections whose headings match (for example "Treatment of Stage III Colon Cancer" on both pages). Inside each matched section it turns every paragraph into a list of numbers that captures its meaning and pairs each patient paragraph with the doctor text that means the most similar thing; the doctor side may be one to three paragraphs joined, because one patient paragraph often condenses several. A second pass then looks across the whole page for leftover patient paragraphs, using a stricter similarity bar because the headings no longer back the match up. Weak matches, copied text, answers much longer than the original, and pairs where the patient text mentions a number or stage that the doctor text never mentions are thrown away (so the model is not taught to invent facts), and patient sentences that just point elsewhere ("see the booklet...") are cut out first. Finally, whole organ families (colon with rectal, the two lung cancers, Hodgkin with non-Hodgkin lymphoma, adult with child versions) are assigned together to the training, validation, or test group, and any validation or test example whose answer is almost a copy of a training answer is removed, so the model is never tested on text it effectively studied. The result is the file of before-and-after examples used to teach our small model to simplify medical text.

## train.py

This script teaches a small existing language model (Google's flan-t5-base) to do one job: take a paragraph written for doctors and rewrite it the way the National Cancer Institute writes it for patients. It learns from the before-and-after pairs that `pdq_pairs.py` made, keeping only pairs whose similarity score is high enough (0.60 by default) so it is not taught to say things the original never said. Training repeats the same loop many times: show the model a batch of doctor paragraphs, measure how far its rewrite is from the real patient version (the "loss"), and nudge its numbers to shrink that error. After each full pass it checks the loss on separate validation pairs, saves the best version so far, and stops early if the model has stopped improving, which prevents it from just memorising the training pairs. The test pairs are never touched, so the final evaluation stays fair.

## batch_summarize.py

TODO: 3–5 plain-English sentences.

## evaluate.py

TODO: 3–5 plain-English sentences.

## frontend

The frontend is one plain HTML page with CSS and JavaScript, with no build step and no outside libraries. It reads data through `api.js`, which has one flag (`USE_MOCK`) that switches between fake files in `frontend/mock/` and the real `/api/...` endpoints. `app.js` draws the search form, result cards and summary panel, and it only inserts API text with `textContent` so data can never run as code. The query and filters are kept in the page address (never the ZIP code), and nothing is saved in cookies or local storage. Colors always come with words (for example the status badge), text is at least 16px, and every control works with a keyboard.

## data/umbrella_terms.json (note for type detection)

Broad phrases such as "lung cancer", "skin cancer" or "blood cancer" are not single cancer types, so they live in `data/umbrella_terms.json` (phrase to list of type ids) instead of any one type's lay terms. Detection must show all the listed types for these, and must never match the bare word "all" (the lay term "all leukemia" only counts as that whole phrase).
