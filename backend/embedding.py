"""Shared sentence-embedding helpers (MiniLM) used by the indexer and the live app.

Why a separate module: the offline embed script, search, and cancer-type
detection must all use the SAME model and the SAME text recipe, otherwise
query vectors and trial vectors would not live in the same space.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from backend import config

_encoder = None


def trial_text(row: Any) -> str:
    """Text we embed for a trial: title + conditions + keywords + brief summary.

    Why these four: they say what the trial is about in plain words. Eligibility
    lists and long descriptions are mostly lab values and would drown the topic.
    MiniLM reads ~256 tokens, so the summary is naturally truncated.
    """
    def lst(v: str | None) -> str:
        try:
            return "; ".join(json.loads(v or "[]"))
        except (TypeError, ValueError):
            return ""
    parts = [row["brief_title"] or "", lst(row["conditions"]), lst(row["keywords"]),
             row["brief_summary"] or ""]
    return ". ".join(p for p in parts if p)


def get_encoder() -> Callable[[list[str]], np.ndarray]:
    """Load MiniLM once from the local models/ folder only; return a function mapping texts -> normalized float32 vectors."""
    global _encoder
    if _encoder is None:
        if not (config.EMBED_MODEL_DIR / "config.json").exists():
            # Rule 4: never fall back to the Hugging Face Hub while serving a user.
            raise RuntimeError(f"Embedding model not found at {config.EMBED_MODEL_DIR}. "
                               "Run `python -m pipeline.fetch_models` once (offline setup step).")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(config.EMBED_MODEL_DIR), local_files_only=True)

        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(model.encode(texts, normalize_embeddings=True,
                                           show_progress_bar=False), dtype="float32")
        _encoder = encode
    return _encoder
