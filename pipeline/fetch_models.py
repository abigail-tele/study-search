"""OFFLINE setup: download the embedding model once so the live app never touches the network.

Why: rule 4 forbids network calls during user requests. Run this once (or at Docker build time):
`python -m pipeline.fetch_models`. It also checks the summarizer checkpoint is present.
"""
from __future__ import annotations

import sys

from backend import config


def fetch_embed_model() -> None:
    """Save MiniLM into config.EMBED_MODEL_DIR (skipped if already there)."""
    if (config.EMBED_MODEL_DIR / "config.json").exists():
        print(f"Embedding model already at {config.EMBED_MODEL_DIR}")
        return
    from sentence_transformers import SentenceTransformer
    config.EMBED_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(config.EMBED_MODEL_NAME).save(str(config.EMBED_MODEL_DIR))
    print(f"Saved embedding model to {config.EMBED_MODEL_DIR}")


def check_summarizer() -> bool:
    """The checkpoint is trained, not downloaded; just report whether it exists."""
    for d in (config.SUMMARY_MODEL_DIR, config.SUMMARY_FALLBACK_MODEL_DIR):
        if d.is_dir():
            print(f"Summarizer checkpoint found: {d}")
            return True
    print(f"WARNING: no summarizer checkpoint in {config.CHECKPOINT_DIR}", file=sys.stderr)
    return False


if __name__ == "__main__":
    fetch_embed_model()
    sys.exit(0 if check_summarizer() else 1)
