# Credits and Licenses

Every dependency and data source used by this project. The Version column shows the version installed in `.venv` (checked with `pip list`) and, in parentheses, the minimum pinned in `requirements.txt`. Licenses were read from each installed package's metadata (`pip show`). Rows marked "(verify)" are unconfirmed, and rows marked "(not yet installed)" have not been checked against installed metadata.

## Python dependencies

| Name | Version | License | URL | What we use it for |
|---|---|---|---|---|
| fastapi | (not yet installed); min >=0.110.0 | MIT (verify) | https://pypi.org/project/fastapi/ | Web API (`/api/*` endpoints) |
| uvicorn[standard] | (not yet installed); min >=0.29.0 | BSD-3-Clause (verify) | https://pypi.org/project/uvicorn/ | ASGI server that runs FastAPI |
| httpx | 0.28.1 (min >=0.27.0) | BSD-3-Clause | https://pypi.org/project/httpx/ | Offline calls to the ClinicalTrials.gov v2 API and PDQ pages |
| beautifulsoup4 | 4.15.0 (min >=4.12.0) | MIT | https://pypi.org/project/beautifulsoup4/ | Parsing cancer.gov PDQ HTML |
| lxml | 6.1.3 (min >=5.2.0) | BSD-3-Clause | https://pypi.org/project/lxml/ | Fast HTML/XML parser used by BeautifulSoup |
| numpy | 2.4.6 (min >=1.26.0) | BSD-3-Clause (pip metadata also lists 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | https://pypi.org/project/numpy/ | Vector math for embeddings and RRF helpers |
| faiss-cpu | (not yet installed); min >=1.8.0 | MIT (verify) | https://pypi.org/project/faiss-cpu/ | Vector similarity index (flat inner-product search) |
| sentence-transformers | 6.2.0 (min >=3.0.0) | Apache-2.0 | https://pypi.org/project/sentence-transformers/ | Loads all-MiniLM-L6-v2 to embed trials and queries |
| transformers | 5.20.2 (min >=4.41.0) | Apache-2.0 | https://pypi.org/project/transformers/ | Loads and runs flan-t5-base for plain-language rewriting |
| datasets | (not yet installed); min >=2.19.0 | Apache-2.0 (verify) | https://pypi.org/project/datasets/ | Dataset handling for model training |
| torch | 2.14.1+cpu (min >=2.3.0) | BSD-3-Clause (verify: pip metadata lists a combined expression of Apache-2.0, BSD-2-Clause, BSD-3-Clause, BSL-1.0, MIT and others) | https://pypi.org/project/torch/ | Deep learning backend for embeddings and T5 training/inference |
| accelerate | (not yet installed); min >=0.30.0 | Apache-2.0 (verify) | https://pypi.org/project/accelerate/ | Training loop support for the Hugging Face stack |
| sentencepiece | (not yet installed); min >=0.2.0 | Apache-2.0 (verify) | https://pypi.org/project/sentencepiece/ | Tokenizer required by flan-t5 |
| textstat | (not yet installed); min >=0.7.3 | MIT (verify) | https://pypi.org/project/textstat/ | Reading-level scores (Flesch-Kincaid grade) before and after rewriting |
| rapidfuzz | (not yet installed); min >=3.9.0 | MIT (verify) | https://pypi.org/project/rapidfuzz/ | Fuzzy string matching for cancer-type detection |
| rouge-score | (not yet installed); min >=0.1.2 | Apache-2.0 (verify) | https://pypi.org/project/rouge-score/ | ROUGE-L evaluation of generated summaries |
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
| NCI PDQ cancer information summaries | cancer.gov pages, fetched offline | Public domain (U.S. government work); credit: National Cancer Institute | https://www.cancer.gov/publications/pdq | Professional and patient-version text pairs for training the plain-language model |
| NCI PDQ breast cancer treatment patient page (Internet Archive Wayback Machine snapshot) | Wayback snapshot 2025-11-22 (`web/20251122021816`) | Public domain (U.S. government work); content remains NCI public domain | http://web.archive.org/web/20251122021816id_/https://www.cancer.gov/types/breast/patient/breast-treatment-pdq | Used because the live cancer.gov URL redirect-looped on 2026-10-24. Stored in `data/pdq_pages.json` as `patient_fetch_url`. |
| NCI Dictionary of Cancer Terms / NCI Thesaurus | current online edition | Public domain (U.S. government work); credit: National Cancer Institute | https://www.cancer.gov/publications/dictionaries | Clinical and lay wording used in `data/cancer_types.json` |
| U.S. Census Bureau Gazetteer ZCTA file | 2024 release (`2024_Gaz_zcta_national.zip`) | Public domain (U.S. government work) | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip | ZIP code centroid coordinates for distance calculation (`data/zip_centroids.csv`, built by `pipeline/zip_centroids.py`) |

## Models

| Name | Version | License | URL | What we use it for |
|---|---|---|---|---|
| google/flan-t5-base | Hugging Face hub, base checkpoint | Apache-2.0 | https://huggingface.co/google/flan-t5-base | Base model that we fine-tune to rewrite trial text in plain language |
| sentence-transformers/all-MiniLM-L6-v2 | Hugging Face hub | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Sentence embeddings for semantic search and PDQ paragraph alignment |

## Full installed dependency list (auto-generated 2026-10-24)

Generated from the installed distributions in  (importlib.metadata), excluding pip, setuptools and wheel. Includes both direct and transitive packages.

| Name | Version | License |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| anyio | 4.15.1 | MIT |
| beautifulsoup4 | 4.15.0 | MIT License |
| certifi | 2026.10.23 | MPL-2.0 |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| filelock | 3.32.3 | MIT |
| fsspec | 2026.7.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.7.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.16.0 | BSD-3-Clause |
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
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.7.0 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| rich | 15.0.0 | MIT |
| safetensors | 0.8.0 | Apache Software License |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.17.1 | BSD License |
| sentence-transformers | 6.2.0 | Apache-2.0 |
| shellingham | 1.5.4 | ISC License |
| soupsieve | 3.0.4 | MIT |
| sympy | 1.14.0 | BSD |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| tokenizers | 0.23.3 | Apache Software License |
| torch | 2.14.1+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 5.20.2 | Apache 2.0 License |
| truststore | 0.10.5 | MIT |
| typer | 0.27.3 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |

### License notes

- certifi MPL-2.0 (transitive via httpx, unmodified): reviewer says acceptable.
- tqdm MPL-2.0 AND MIT (transitive, unmodified): acceptable.
- CNRI-Python (regex, via transformers), PSF-2.0 (typing_extensions), ISC (shellingham): permissive but not in the rule 5 list (MIT, Apache 2.0, BSD, public domain). NEEDS HUMAN REVIEW.
- No package with GPL, LGPL or AGPL in its license was found. No package has an unknown license.
- textstat (planned, not installed) depends on pyphen (GPLv2+/LGPLv2+/MPL-1.1 tri-license): needs human decision before install. Use under MPL-1.1, or write our own Flesch-Kincaid.
