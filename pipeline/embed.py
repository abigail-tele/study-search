"""Build the FAISS meaning-search index from the trials database.

Run: python -m pipeline.embed
Why FAISS flat inner product: vectors are normalized, so inner product equals
cosine similarity, and ~30k vectors search in milliseconds without tuning.
"""
from __future__ import annotations

import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend import config, db
from backend.embedding import trial_text

BATCH_SIZE = 64


def build_index(conn, encode, batch_size: int = BATCH_SIZE, progress: bool = False):
    """Embed every trial; return (faiss index, row->nct list)."""
    rows = conn.execute("SELECT nct_id, brief_title, conditions, keywords, brief_summary "
                        "FROM trials ORDER BY nct_id").fetchall()
    ids = [r["nct_id"] for r in rows]
    texts = [trial_text(r) for r in rows]
    chunks = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        chunks.append(encode(texts[i:i + batch_size]))
        if progress and (i // batch_size) % 20 == 0:
            print(f"  {i + batch_size}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    vecs = np.vstack(chunks) if chunks else np.zeros((0, 384), dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, ids


def save(index, ids: list[str]) -> None:
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    config.FAISS_ROWMAP_PATH.write_text(json.dumps(ids), encoding="utf-8")


def main() -> None:
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    print(f"Embedding {total} trials with {config.EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(config.EMBED_MODEL_NAME)
    encode = lambda t: np.asarray(model.encode(t, normalize_embeddings=True,
                                               show_progress_bar=False), dtype="float32")
    t0 = time.time()
    index, ids = build_index(conn, encode, progress=True)
    save(index, ids)
    assert index.ntotal == len(ids) == total, "row map must match trials count"
    print(f"Done: {index.ntotal} vectors, {time.time() - t0:.0f}s, "
          f"index {config.FAISS_INDEX_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
