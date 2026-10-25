# Credits and Licenses

Every dependency and data source used by this project. The Version column shows the version installed in `.venv` (checked with `pip list`) and, in parentheses, the minimum pinned in `requirements.txt`. Licenses were read from each installed package's metadata (`pip show`). Rows marked "(verify)" are unconfirmed, and rows marked "(not yet installed)" have not been checked against installed metadata.

## Python dependencies

| Name | Version | License | URL | What we use it for |
|---|---|---|---|---|
| fastapi | 0.162.0 (min >=0.110.0) | MIT | https://pypi.org/project/fastapi/ | Web API (`/api/*` endpoints) |
| uvicorn[standard] | 0.54.0 (min >=0.29.0) | BSD-3-Clause | https://pypi.org/project/uvicorn/ | ASGI server that runs FastAPI |
| httpx | 0.28.1 (min >=0.27.0) | BSD-3-Clause | https://pypi.org/project/httpx/ | Offline calls to the ClinicalTrials.gov v2 API and PDQ pages |
| beautifulsoup4 | 4.15.0 (min >=4.12.0) | MIT | https://pypi.org/project/beautifulsoup4/ | Parsing cancer.gov PDQ HTML |
| lxml | 6.1.3 (min >=5.2.0) | BSD-3-Clause | https://pypi.org/project/lxml/ | Fast HTML/XML parser used by BeautifulSoup |
| numpy | 2.4.6 (min >=1.26.0) | BSD-3-Clause (pip metadata also lists 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | https://pypi.org/project/numpy/ | Vector math for embeddings and RRF helpers |
| faiss-cpu | 1.15.1 | MIT | https://pypi.org/project/faiss-cpu/ | Vector similarity index (flat inner-product search) |
| sentence-transformers | 6.2.0 (min >=3.0.0) | Apache-2.0 | https://pypi.org/project/sentence-transformers/ | Loads all-MiniLM-L6-v2 to embed trials and queries |
| transformers | 5.20.2 (min >=4.41.0) | Apache-2.0 | https://pypi.org/project/transformers/ | Loads and runs flan-t5-base for plain-language rewriting |
| torch | 2.14.1+cpu (min >=2.3.0) | Combined expression per package metadata: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (all permissive) | https://pypi.org/project/torch/ | Deep learning backend for embeddings and T5 training/inference |
| sentencepiece | 0.2.2 (min >=0.2.0) | Apache-2.0 | https://pypi.org/project/sentencepiece/ | Tokenizer required by flan-t5 |
| rapidfuzz | 3.14.6 (min >=3.9.0) | MIT | https://pypi.org/project/rapidfuzz/ | Fuzzy string matching for cancer-type detection (used in `backend/taxonomy.py` for misspelled cancer names) |
| rouge-score | 0.1.2 (min >=0.1.2) | Apache-2.0 | https://pypi.org/project/rouge-score/ | ROUGE-L evaluation of generated summaries |
| pytest | 9.1.1 (min >=8.0.0) | MIT | https://pypi.org/project/pytest/ | Test suite |

## Transitive dependencies

These are not listed in `requirements.txt` but are installed as dependencies of the packages above.

| Name | Version | License | URL | Why it is here |
|---|---|---|---|---|
| httpcore | 1.0.9 | BSD-3-Clause | https://pypi.org/project/httpcore/ | Transport layer under httpx |
| anyio | 4.15.1 | MIT | https://pypi.org/project/anyio/ | Async I/O layer used by httpx and FastAPI/Starlette |
| h11 | 0.16.0 | MIT | https://pypi.org/project/h11/ | HTTP/1.1 protocol parser under httpcore |
| certifi | 2026.10.23 | MPL-2.0 | https://pypi.org/project/certifi/ | CA certificate bundle used by httpx. Weak copyleft, transitive dependency of httpx, used unmodified; flagged for human review under rule 5. |

## Data sources

| Name | Version / date | License | URL | What we use it for |
|---|---|---|---|---|
| ClinicalTrials.gov API v2 | live API, fetched offline | U.S. National Library of Medicine terms; public domain (U.S. government work) | https://clinicaltrials.gov/data-api/api | Source of every trial record (title, status, sites, eligibility, dates) |
| NCI PDQ cancer information summaries | cancer.gov pages, fetched offline | Public domain (U.S. government work); credit: National Cancer Institute | https://www.cancer.gov/publications/pdq | Professional and patient-version text pairs for training the plain-language model; the pairs in `data/` are adapted (cleaned, split, aligned, filtered) from NCI PDQ |
| NCI PDQ breast cancer treatment patient page (Internet Archive Wayback Machine snapshot) | Wayback snapshot 2025-11-22 (`web/20251122021816`) | Public domain (U.S. government work); content remains NCI public domain | http://web.archive.org/web/20251122021816id_/https://www.cancer.gov/types/breast/patient/breast-treatment-pdq | Used because the live cancer.gov URL redirect-looped on 2026-10-24. Stored in `data/pdq_pages.json` as `patient_fetch_url`. |
| NCI Dictionary of Cancer Terms / NCI Thesaurus | current online edition | Public domain (U.S. government work); credit: National Cancer Institute | https://www.cancer.gov/publications/dictionaries | Clinical and lay wording used in `data/cancer_types.json` |
| NCI Dictionary of Cancer Terms (cancer.gov) | Online edition, paraphrased 2026-10-24 | Public domain (U.S. government work); credit: National Cancer Institute | https://www.cancer.gov/publications/dictionaries | Plain-language definitions paraphrased (by an AI agent) into `data/glossary.json` (76 terms), shown to patients as a fallback |
| U.S. Census Bureau Gazetteer ZCTA file | 2024 release (`2024_Gaz_zcta_national.zip`) | Public domain (U.S. government work) | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip | ZIP code centroid coordinates for distance calculation (`data/zip_centroids.csv`, built by `pipeline/zip_centroids.py`) |
| ClinicalTrials.gov detailed-description to brief-summary pairs | Fetched offline, 2026-10-24 | Public domain (U.S. government work, NLM) | https://clinicaltrials.gov/data-api/api | Training data only: brief-summary sentences aligned to detailed-description sentences (`pipeline/ctgov_pairs.py`, output `data/ctgov_pairs*.jsonl`); trials in the trial evaluation set are excluded |
| data/drug_lexicon.txt | Written 2026-10-24 (85 lines) | Public domain (generic drug names; the file header says generic names are not copyrightable) | Not applicable (no third-party list copied) | List of generic drug names used by the faithfulness check (`backend/summarize.py`) for drugs its suffix rules miss; compiled by an AI agent for this project |

### Data licensing and attribution

- NCI PDQ and NCI Dictionary text is public domain (U.S. government work). Credit: National Cancer Institute. The PDQ-derived pairs in `data/` are adapted from NCI PDQ; they are not verbatim NCI products. NCI does not endorse this project. PDQ is a registered trademark of the National Cancer Institute and is used here only to identify the source data. See https://www.cancer.gov/policies/copyright-reuse.
- ClinicalTrials.gov data is public domain (U.S. government work). ClinicalTrials.gov is a service of NLM/NIH; this project is not affiliated with or endorsed by NLM, NIH, or the U.S. government.
- U.S. Census Bureau Gazetteer data is public domain (U.S. government work).
- The repository's MIT license covers original code and documentation only. It does not relicense the public-domain source text or third-party model weights.
- Full attribution and per-file details: `NOTICE` and `data/README.md`.

## Models

| Name | Version | License | URL | What we use it for |
|---|---|---|---|---|
| google/flan-t5-base | Hugging Face hub, base checkpoint | Apache-2.0 | https://huggingface.co/google/flan-t5-base | Base model that we fine-tune to rewrite trial text in plain language |
| sentence-transformers/all-MiniLM-L6-v2 | Hugging Face hub | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Sentence embeddings for semantic search and PDQ paragraph alignment |
| google/flan-t5-large | Hugging Face hub, large checkpoint | Apache-2.0 | https://huggingface.co/google/flan-t5-large | Evaluated, not shipped: compared with flan-t5-base in `docs/EVAL.md` |
| Qwen/Qwen2.5-1.5B-Instruct | Hugging Face hub | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Evaluated, not shipped: LoRA fine-tuned with `pipeline/train_lora.py` and compared in `docs/EVAL.md` |

## Training environment (.venv-train)

A separate virtual environment used only for fine-tuning on the GPU: flan-t5-base (`pipeline/train.py`) and Qwen2.5-1.5B-Instruct with LoRA (`pipeline/train_lora.py`). Kept apart from `.venv` so the CUDA build of torch does not replace the CPU build the server uses. Versions and licenses were read with `.venv-train/Scripts/python -m pip show` and `importlib.metadata`.

| Name | Version | License | URL | What we use it for |
|---|---|---|---|---|
| torch | 2.14.1+cu130 | BSD-3-Clause (pip metadata gives a combined expression: Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, MIT) | https://pypi.org/project/torch/ | GPU training of flan-t5-base (CUDA 13.0 build) |
| transformers | 5.20.2 | Apache-2.0 | https://pypi.org/project/transformers/ | Loads flan-t5-base and runs the seq2seq training loop |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://pypi.org/project/numpy/ | Array handling used by transformers and torch |
| sentencepiece | 0.2.2 | Apache-2.0 | https://pypi.org/project/sentencepiece/ | T5 tokenizer |
| tokenizers | 0.23.3 | Apache-2.0 (pip classifier: Apache Software License) | https://pypi.org/project/tokenizers/ | Fast tokenizer backend used by transformers |
| safetensors | 0.8.0 | Apache-2.0 (pip classifier: Apache Software License) | https://pypi.org/project/safetensors/ | Saving and loading model checkpoints |
| peft | 0.21.2 | Apache-2.0 (pip classifier: Apache Software License) | https://pypi.org/project/peft/ | LoRA adapters for the Qwen2.5-1.5B runs (`pipeline/train_lora.py`); training only |
| trl | 1.16.0 | Apache-2.0 | https://pypi.org/project/trl/ | Supervised fine-tuning trainer with completion-only loss for the Qwen2.5 LoRA runs; training only |

Attribution: LoRA hyperparameters for run J2 (r=64, alpha=16, lr 2e-4, completion-only loss) adapted from a team member's friend's fine-tuning script; no code copied. Our `pipeline/train_lora.py` code was written fresh. `peft` and `trl` are used only in `.venv-train` and are not imported by the app or the tests.

## Full installed dependency list (auto-generated 2026-10-24)

Generated from the installed distributions in `.venv` (importlib.metadata), excluding pip, setuptools and wheel. Includes both direct and transitive packages. License lookup order: License-Expression, then License, then classifiers (each short field under 80 characters); if none of those is short, the full License-Expression is used.

| Name | Version | License |
|---|---|---|
| absl-py | 2.5.1 | Apache-2.0 |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| beautifulsoup4 | 4.15.0 | MIT License |
| certifi | 2026.10.23 | MPL-2.0 |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| defusedxml | 0.7.1 | PSFL |
| faiss-cpu | 1.15.1 | MIT |
| fastapi | 0.162.0 | MIT |
| filelock | 3.32.3 | MIT |
| fsspec | 2026.7.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.7.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httptools | 0.9.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| httpx2 | 2.16.0 | BSD-3-Clause |
| huggingface_hub | 2.3.0 | Apache-2.0 |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| Jinja2 | 3.1.6 | BSD License |
| joblib | 1.6.0 | BSD-3-Clause |
| lxml | 6.1.3 | BSD-3-Clause |
| markdown-it-py | 4.2.0 | MIT License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT License |
| mpmath | 1.3.0 | BSD |
| narwhals | 2.27.1 | MIT |
| networkx | 3.6.1 | BSD-3-Clause |
| nltk | 3.10.4 | Apache License, Version 2.0 |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.7.0 | MIT |
| pydantic | 2.14.1 | MIT |
| pydantic_core | 2.50.1 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| RapidFuzz | 3.14.6 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| rich | 15.0.0 | MIT |
| rouge_score | 0.1.2 | Apache Software License |
| safetensors | 0.8.0 | Apache Software License |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.17.1 | BSD License |
| sentence-transformers | 6.2.0 | Apache-2.0 |
| sentencepiece | 0.2.2 | Apache-2.0 |
| shellingham | 1.5.4 | ISC License |
| six | 1.17.0 | MIT |
| soupsieve | 3.0.4 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| sympy | 1.14.0 | BSD |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| tokenizers | 0.23.3 | Apache Software License |
| torch | 2.14.1+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 5.20.2 | Apache 2.0 License |
| truststore | 0.10.5 | MIT |
| typer | 0.27.3 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |
| watchfiles | 1.3.0 | MIT |
| websockets | 17.3 | BSD-3-Clause |

### License notes

- certifi MPL-2.0 (transitive via httpx, unmodified): reviewer says acceptable.
- tqdm MPL-2.0 AND MIT (transitive, unmodified): acceptable.
- textstat removed; reading level uses our own Flesch-Kincaid in `backend/readability.py`.
- NEEDS HUMAN REVIEW (permissive, but not on the rule 5 list of MIT, Apache 2.0, BSD, public domain):
  - regex: Apache-2.0 AND CNRI-Python (transitive via transformers).
  - typing_extensions: PSF-2.0 (transitive).
  - shellingham: ISC (transitive via typer).
  - defusedxml: PSF-2.0 (transitive via rouge-score, which is in `pipeline/compare_checkpoints.py` and `pipeline/baselines.py`).
- No package with GPL, LGPL or AGPL in its license was found. No package has an unknown license.
