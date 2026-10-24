# Cancer Trials Plain-Language Finder (study-search)

A web app that helps patients and families find cancer clinical trials in plain words. Search by cancer type, get ranked trial cards that link to ClinicalTrials.gov, then open a plain-language summary of any study.

For education, not medical advice. Talk with your doctor before making any decision about a clinical trial.

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
.un_api.ps1
```

Frontend only, with mock data: `.un_frontend.ps1` → http://localhost:8765. Tests: `.venv\Scripts\python -m pytest -q`.

## Documentation

- [docs/PLAN.md](docs/PLAN.md): schedule, user journey, API contract
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): plain-English tour of each module
- [docs/EVAL.md](docs/EVAL.md): evaluation results
- [CREDITS.md](CREDITS.md): dependencies, data sources, and licenses
- [AI_USE.md](AI_USE.md): log of AI-assisted work

## License

MIT License. See [LICENSE](LICENSE).
