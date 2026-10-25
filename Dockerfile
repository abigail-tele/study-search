# Cancer Trials Plain-Language Finder - container for Hugging Face Spaces (Docker SDK) or Render.
# Build context must contain: backend/ frontend/ deploy/ data/ (db + faiss + json + csv) models/all-MiniLM-L6-v2
# and, only if you want live summaries, checkpoints/flan-t5-pdq-best. See docs/DEPLOY.md.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Rule 4: the running server must never contact Hugging Face.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Default for deployment: serve pre-generated summaries only; the T5 model is never loaded.
    STUDY_SEARCH_LIVE_SUMMARY=0 \
    PORT=7860

# curl is used only by the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000; do the same everywhere.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# CPU-only torch (~200 MB) instead of the default CUDA wheel (~2 GB+). Separate layer so it caches.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY deploy/requirements-deploy.txt /tmp/requirements-deploy.txt
RUN pip install -r /tmp/requirements-deploy.txt

COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser models ./models
# Optional (live summaries only). The stage script creates an empty checkpoints/ dir when the model is omitted.
COPY --chown=appuser:appuser checkpoints ./checkpoints

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/cancer-types" >/dev/null || exit 1

# --no-access-log: rule 8, request URLs (search text, ZIPs) must never be logged.
CMD ["sh", "-c", "exec python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
