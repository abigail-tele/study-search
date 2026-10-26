"""Tiny taxonomy + deterministic fake encoder for semantic type-fallback tests.

Why fake: unit tests must stay offline and fast, so we never load MiniLM. The
hand-set vectors make "milk ducts" close to breast terms, "close pair" close to
breast and lung, and ordinary off-topic text orthogonal to every cancer type.
"""
from __future__ import annotations

import numpy as np

from backend import embedding, taxonomy

TYPES = [
    {"id": "breast", "label": "Breast cancer",
     "lay_terms": ["breast cancer"], "clinical_terms": ["ductal carcinoma"]},
    {"id": "lung", "label": "Lung cancer",
     "lay_terms": ["lung cancer"], "clinical_terms": ["pulmonary carcinoma"]},
    {"id": "prostate", "label": "Prostate cancer",
     "lay_terms": ["prostate cancer"], "clinical_terms": ["prostatic carcinoma"]},
    {"id": "melanoma", "label": "Melanoma",
     "lay_terms": ["melanoma"], "clinical_terms": ["malignant melanoma"]},
]

# Query -> hand-set vector. The encoder L2-normalizes every vector below.
_SPECIAL = {
    "milk ducts": [0.95, 0.0, 0.0, 0.0, 0.05],
    "close pair": [0.80, 0.78, 0.0, 0.0, 0.02],
    "clear winner": [0.80, 0.60, 0.0, 0.0, 0.0],
    "weak match": [0.50, 0.0, 0.0, 0.0, 0.866],
    "four types": [0.80, 0.79, 0.78, 0.77, 0.0],
    "knee pain": [0.0, 0.0, 0.0, 0.0, 1.0],
}


def fake_encode(texts: list[str]) -> np.ndarray:
    """Map known phrases to hand-set vectors; normalize so dot product is cosine."""
    rows: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        for phrase, vector in _SPECIAL.items():
            if phrase in lowered:
                rows.append(vector)
                break
        else:
            if any(w in lowered for w in ("breast", "duct", "mammary", "lobular")):
                rows.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif any(w in lowered for w in ("lung", "pulmonary", "nsclc", "sclc")):
                rows.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif any(w in lowered for w in ("prostate", "prostatic")):
                rows.append([0.0, 0.0, 1.0, 0.0, 0.0])
            elif any(w in lowered for w in ("melanoma", "mole")):
                rows.append([0.0, 0.0, 0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
    matrix = np.asarray(rows, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def install(monkeypatch) -> None:
    """Point taxonomy at the tiny data and fake encoder; clear cached state."""
    taxonomy.cancer_type_term_matrix.cache_clear()
    taxonomy.load_cancer_types.cache_clear()
    taxonomy.load_umbrella_terms.cache_clear()
    monkeypatch.setattr(taxonomy, "load_cancer_types", lambda: TYPES)
    monkeypatch.setattr(taxonomy, "load_umbrella_terms", lambda: {})
    monkeypatch.setattr(embedding, "get_encoder", lambda: fake_encode)
    # The human implementation may import get_encoder into taxonomy directly.
    monkeypatch.setattr(taxonomy, "get_encoder", lambda: fake_encode, raising=False)
