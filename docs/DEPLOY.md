# Deploying (Hugging Face Spaces, Docker SDK; Render as fallback)

Rule 4: nothing is downloaded or fetched while serving users. All artifacts are baked in at build time.

## What ships

| Artifact | Size | In plain git? |
|---|---|---|
| `data/trials_ship.db` (copied as `data/trials.db` in the Space) | ~430 MB | no (Git LFS) |
| `data/faiss.index` | 43 MB | no (Git LFS) |
| `data/faiss_rowmap.json`, `cancer_types.json`, `umbrella_terms.json`, `glossary.json`, `zip_centroids.csv` | ~1.5 MB | yes |
| `models/all-MiniLM-L6-v2/` | 88 MB | no (Git LFS) |
| `checkpoints/flan-t5-pdq-best/` (optional) | ~950 MB | no (Git LFS) |

Build the slim DB first: `.venv\Scripts\python -m pipeline.make_ship_db` (after `batch_summarize` has filled `summaries`).

## Recommended: Hugging Face Spaces + Git LFS (option a)

Human steps (agents must not do these):
1. Create a free Hugging Face account, then a new Space: SDK = Docker, blank template, public.
2. Install Git LFS and log in (`hf auth login` or a write token).
3. Stage the folder (no network, no push): `powershell -File deploy\stage_space.ps1` creates `..\study-search-space\`.
   Add `-IncludeModel` to also ship the T5 checkpoint; the default serves pre-generated summaries only.
4. Push:
   ```
   cd ..\study-search-space
   git init -b main
   git lfs install
   git add .
   git commit -m "Initial Space"
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
5. Wait for the build (about 5-10 min), then open `https://<user>-<space-name>.hf.space` on a phone.

`.gitattributes` written by the stage script (exact lines):
```
*.db filter=lfs diff=lfs merge=lfs -text
*.index filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
data/faiss_rowmap.json filter=lfs diff=lfs merge=lfs -text
```
Equivalent: `git lfs track "*.db" "*.index" "*.safetensors" "*.bin"`.

Folder layout the Dockerfile expects (the stage script builds it):
```
Dockerfile  .dockerignore  README.md (Space YAML header)  .gitattributes
backend/  frontend/  deploy/requirements-deploy.txt
data/{trials.db,faiss.index,faiss_rowmap.json,cancer_types.json,umbrella_terms.json,glossary.json,zip_centroids.csv}
models/all-MiniLM-L6-v2/
checkpoints/   (empty, or flan-t5-pdq-best/ with -IncludeModel)
```

## Option b: download at BUILD time from a Hugging Face repo you create

Upload the artifacts once to an HF dataset repo (`hf upload <user>/<repo> data/trials_ship.db data/trials.db --repo-type dataset`, same for the index, MiniLM, and optionally the checkpoint). In the Dockerfile, before `USER appuser`, add:
```
ARG HF_TOKEN
RUN pip install huggingface_hub && python -c "from huggingface_hub import snapshot_download; snapshot_download('<user>/<repo>', repo_type='dataset', local_dir='/app', token='${HF_TOKEN}')"
```
and remove the `COPY data/models/checkpoints` lines. This runs during `docker build` only; the running container has `HF_HUB_OFFLINE=1`, so requests never touch the network. Keep the token a build secret, never in the repo.

## Memory and CPU

Measured natively: the app (MiniLM + FAISS 43 MB + SQLite, no T5) idles at about 565 MB RSS.
T5-base adds about 1 GB for fp32 weights plus 0.3-0.5 GB while generating, so about 2 GB with live summaries.
HF Spaces free CPU tier: 2 vCPU, 16 GB RAM, so both fit. Render free (512 MB) does not fit even without T5; use a plan with at least 1 GB and pre-generated summaries only.

## "No model on host" switch

`STUDY_SEARCH_LIVE_SUMMARY` is read per request in `backend/app.py`:
- `0` (Dockerfile default): `/api/summary/{nct}` only reads the `summaries` table. A miss returns 404 and the UI says the summary is not ready. The T5 model is never loaded.
- `1` (local use, or a Space shipped with `-IncludeModel`; set it as a Space variable): on a miss, `summarize_trial` runs (faithfulness check included), the row is stored if the DB is writable, and it is returned. This stores a public-trial summary, never anything the user typed (rule 8).

## Image size estimate

Dry run of the Dockerfile's pip steps in a fresh venv (CPU torch + `deploy/requirements-deploy.txt`): 1.2 GB of site-packages (torch 543 MB, transformers 122, scipy 119). Estimate: about 1.2 GB Python layers + 0.15 GB base + 0.6 GB data = about 2.0 GB, or about 3.0 GB with the T5 checkpoint. Docker is not installed on the build machine, so `docker build` itself was never run; the Dockerfile was reviewed by hand and its install and uvicorn steps were run natively with the same offline env vars.

## Render fallback

New Web Service, Docker runtime, pointed at a GitHub repo containing the staged folder (with LFS). `PORT` is set by Render; health check path `/api/cancer-types`. Plan with at least 1 GB RAM.
