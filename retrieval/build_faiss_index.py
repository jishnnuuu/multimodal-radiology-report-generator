"""
Build FAISS Semantic Retrieval Index
======================================

Goal
----
Encode all radiology reports from the training set into dense vector
embeddings and store them in a FAISS index for fast similarity search
at inference time.

Why Cosine Similarity (not L2)?
--------------------------------
L2 distance is sensitive to vector magnitude.  Sentence embeddings from
MiniLM have varying norms, so two semantically similar sentences can have
large L2 distance simply because one has a larger norm.

Fix: L2-normalise all embeddings before indexing → inner product == cosine.
We use `IndexFlatIP` (inner product) on normalised vectors.

Pipeline
--------
    Training CSV
        ↓
    SentenceTransformer (all-MiniLM-L6-v2, dim=384)
        ↓
    L2-Normalise Embeddings
        ↓
    FAISS IndexFlatIP
        ↓
    retrieval/faiss_index.bin
    retrieval/reports.pkl

Fixes Applied
-------------
- Changed from IndexFlatL2 to IndexFlatIP on L2-normalised embeddings
  to achieve correct cosine similarity ranking.
- Only indexes the TRAINING split to prevent retrieval leakage on val/test.
- Added assertion to verify index size matches report count.
- Output directory created automatically if missing.
"""

import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_CSV        = "dataset/train.csv"
RETRIEVAL_DIR    = "retrieval"
FAISS_INDEX_PATH = os.path.join(RETRIEVAL_DIR, "faiss_index.bin")
REPORTS_PATH     = os.path.join(RETRIEVAL_DIR, "reports.pkl")


# ── Embedding Model ────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: fast, 384-dim, strong semantic similarity performance
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_faiss_index() -> None:
    os.makedirs(RETRIEVAL_DIR, exist_ok=True)

    # ── Load Training Reports ─────────────────────────────────────────────────
    df      = pd.read_csv(TRAIN_CSV)
    reports = df["report"].dropna().tolist()
    print(f"Indexing {len(reports)} training reports …")

    # ── Encode ────────────────────────────────────────────────────────────────
    model      = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        reports,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # ── L2 Normalise → cosine similarity via inner product ────────────────────
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    print(f"Embedding dimension : {dimension}")

    # ── Build Index ───────────────────────────────────────────────────────────
    index = faiss.IndexFlatIP(dimension)   # inner product on normalised = cosine
    index.add(embeddings)

    assert index.ntotal == len(reports), (
        f"Index size mismatch: {index.ntotal} vs {len(reports)} reports"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved  → {FAISS_INDEX_PATH}  ({index.ntotal} vectors)")

    with open(REPORTS_PATH, "wb") as f:
        pickle.dump(reports, f)
    print(f"Report texts saved → {REPORTS_PATH}")


if __name__ == "__main__":
    build_faiss_index()