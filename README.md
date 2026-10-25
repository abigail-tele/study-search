# Cancer Trials Plain-Language Finder (study-search)

A web app that helps patients and families find cancer clinical trials in plain words. Search by cancer type, get ranked trial cards that link to ClinicalTrials.gov, then open a plain-language summary of any study.

For education, not medical advice. Talk with your doctor before making any decision about a clinical trial.

## Data sources and attribution

Trial records come from [ClinicalTrials.gov](https://clinicaltrials.gov/), a service of the U.S. National Library of Medicine (NLM) at the National Institutes of Health (NIH). ClinicalTrials.gov data is public domain. This project is not affiliated with, endorsed by, or sponsored by NLM, NIH, or the U.S. government.

Plain-language training pairs are adapted from the National Cancer Institute (NCI) PDQ cancer information summaries. Additional training pairs match each ClinicalTrials.gov trial's detailed description to its easier brief summary (public domain, built by `pipeline/ctgov_pairs.py`). Glossary definitions are paraphrased from the NCI Dictionary of Cancer Terms. NCI text is public domain; credit: National Cancer Institute. NCI does not endorse this project. PDQ is a registered trademark of NCI; the name is used here only to identify the source data.

ZIP centroids come from the U.S. Census Bureau 2024 Gazetteer ZCTA file (public domain).

See [NOTICE](NOTICE) and [data/README.md](data/README.md) for full attribution and per-file details.

## Setup

Requires Python 3.11.

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

One-time offline steps (internet needed only here, never while the app runs):

```
.venv\Scripts\python -m pipeline.indexer          # ClinicalTrials.gov -> data/trials.db (~1 hour for all 35 types)
.venv\Scripts\python -m pipeline.zip_centroids    # Census ZIP centroids -> data/zip_centroids.csv
.venv\Scripts\python -m pipeline.fetch_models     # embedding model -> models/
.venv\Scripts\python -m pipeline.embed            # FAISS index -> data/faiss.index
```

The summarizer needs a trained checkpoint in `checkpoints/flan-t5-pdq-best` (see [docs/TRAINING.md](docs/TRAINING.md)).

Start the whole app (API + frontend) at http://127.0.0.1:8000:

```
.\run_api.ps1
```

Frontend only, with mock data: `.\run_frontend.ps1` → http://localhost:8765. Tests: `.venv\Scripts\python -m pytest -q`.

## Documentation

- [docs/PLAN.md](docs/PLAN.md): schedule, user journey, API contract
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): plain-English tour of each module
- [docs/EVAL.md](docs/EVAL.md): evaluation results
- [NOTICE](NOTICE): data attribution and third-party licenses
- [data/README.md](data/README.md): per-file data provenance
- [CREDITS.md](CREDITS.md): dependencies, data sources, and licenses
- [AI_USE.md](AI_USE.md): log of AI-assisted work

## License

The original code and documentation are MIT-licensed; see [LICENSE](LICENSE). Third-party data and models keep their own public-domain or Apache-2.0 status as described in [NOTICE](NOTICE).
