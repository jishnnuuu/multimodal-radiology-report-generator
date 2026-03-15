"""
SEMANTIC RETRIEVAL WITH FAISS

Goal
----
Retrieve radiology reports based on semantic similarity.

Unlike BM25:
    "opacity"
    "consolidation"

can still match.

Pipeline
--------
    Query Text
        ↓
    Sentence Embedding
        ↓
    FAISS Similarity Search
        ↓
    Top-k Similar Reports
"""

import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


# paths
FAISS_INDEX_PATH = "retrieval/faiss_index.bin"
REPORTS_PATH = "retrieval/reports.pkl"


# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(REPORTS_PATH, "rb") as f:
        reports = pickle.load(f)
    return index, reports


def retrieve_documents(query, top_k=5):
    index, reports = load_index()
    
    # convert query to embedding
    query_embedding = model.encode([query])
    
    query_embedding = np.array(query_embedding).astype("float32")
    
    # search FAISS
    distances, indices = index.search(query_embedding, top_k)
    results = [reports[i] for i in indices[0]]
    return results