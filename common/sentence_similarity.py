"""Sentence-level similarity helpers (all-MiniLM-L6-v2 + cosine + 0.1 binning)."""

from typing import List, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model: SentenceTransformer | None = None


def _load() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_similarity_matrix(prompts: Sequence[str]) -> np.ndarray:
    embeddings = _load().encode(list(prompts))
    return cosine_similarity(embeddings)


def round_to_bin(similarity: float, step: float = 0.1) -> float:
    """Round a similarity score to the nearest ``step`` (e.g. 0.1)."""
    return float(round(similarity / step) * step)


def matrix_to_dataframe(prompts: Sequence[str], decimals: int = 4) -> pd.DataFrame:
    matrix = compute_similarity_matrix(prompts)
    labels = [f"prompt{i+1}" for i in range(len(prompts))]
    return pd.DataFrame(matrix, columns=labels, index=labels).round(decimals)


def pair_bins(prompts: Sequence[str]) -> List[List[float]]:
    """Return an N x N matrix of similarities rounded to the nearest 0.1.

    Used by the fitting code to bucket personal prompts when fitting one
    sigmoid curve per similarity bin.
    """
    matrix = compute_similarity_matrix(prompts)
    return [[round_to_bin(float(v)) for v in row] for row in matrix]
