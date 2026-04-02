"""
Semantic Retrieval with FAISS
==============================

Goal
----
Given a generated radiology report (the query), retrieve the top-k most
semantically similar reports from the indexed training corpus.

Unlike keyword matching, dense retrieval handles synonyms:
    "opacity" ↔ "consolidation"
    "enlarged heart" ↔ "cardiomegaly"

Pipeline
--------
    Query Text
        ↓
    SentenceTransformer Embedding  (dim=384)
        ↓
    L2 Normalise  (to match index encoding)
        ↓
    FAISS IndexFlatIP Search
        ↓
    Top-k Reports + Similarity Scores

Fixes Applied
-------------
- Query embedding must also be L2-normalised to match the normalised index.
  Without this, cosine similarity is NOT computed correctly.
- Index and reports are loaded lazily and cached as module-level singletons
  so they are not reloaded on every call (important for inference speed).
- Returns (docs, scores) tuple so callers can filter by minimum similarity.
"""

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ── Paths ──────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "retrieval/faiss_index.bin"
REPORTS_PATH     = "retrieval/reports.pkl"

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


# ── Lazy Singleton Cache ───────────────────────────────────────────────────────
_index:   faiss.Index | None = None
_reports: list[str]  | None = None
_model:   SentenceTransformer | None = None


def _load_resources() -> tuple:
    """Load index, reports, and embedding model once and cache them."""
    global _index, _reports, _model

    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)

    if _reports is None:
        with open(REPORTS_PATH, "rb") as f:
            _reports = pickle.load(f)

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _index, _reports, _model


# ── Public API ─────────────────────────────────────────────────────────────────
def retrieve_documents(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> tuple[list[str], list[float]]:
    """
    Retrieve semantically similar radiology reports.

    Parameters
    ----------
    query     : generated report text to use as the search query
    top_k     : number of candidates to retrieve
    min_score : minimum cosine similarity threshold (0–1);
                results below this score are filtered out

    Returns
    -------
    docs   : list of retrieved report strings
    scores : corresponding cosine similarity scores (higher = more similar)
    """
    index, reports, model = _load_resources()

    # Encode and normalise the query to match index encoding
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores_arr, indices = index.search(query_vec, top_k)

    docs:   list[str]   = []
    scores: list[float] = []

    for score, idx in zip(scores_arr[0], indices[0]):
        if idx < 0:           # FAISS returns -1 for unfilled slots
            continue
        if float(score) < min_score:
            continue
        docs.append(reports[idx])
        scores.append(float(score))

    return docs, scores