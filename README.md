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

- TODO: build the offline index (`pipeline/indexer.py`, `pipeline/embed.py`)
- TODO: start the API (`uvicorn backend.app:app`)
- TODO: open the frontend (`frontend/index.html`)

## Documentation

- [docs/PLAN.md](docs/PLAN.md): schedule, user journey, API contract
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): plain-English tour of each module
- [docs/EVAL.md](docs/EVAL.md): evaluation results
- [CREDITS.md](CREDITS.md): dependencies, data sources, and licenses
- [AI_USE.md](AI_USE.md): log of AI-assisted work

## License

MIT License. See [LICENSE](LICENSE).
