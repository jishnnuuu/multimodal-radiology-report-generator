"""
BUILD FAISS INDEX


Goal
----
Create an embedding-based retrieval system.

Instead of matching words (BM25),
we compare sentence embeddings.

Similar meaning → similar vector.

Example:

Query:
    "right lower lobe opacity"

Report:
    "consolidation in right lower lung"

Even though words differ,
their embeddings will be similar.


Pipeline
--------

Radiology Reports
        ↓
Sentence Embedding Model
        ↓
Vector Embeddings
        ↓
FAISS Index
        ↓
Fast Similarity Search


Output
------

retrieval/faiss_index.bin
retrieval/reports.pkl
"""

import pandas as pd
import numpy as np
import pickle

# FAISS for vector search
import faiss

# sentence embedding model
from sentence_transformers import SentenceTransformer


DATASET_PATH = "dataset/iu_xray_dataset.csv"
FAISS_INDEX_PATH = "retrieval/faiss_index.bin"
REPORTS_PATH = "retrieval/reports.pkl"


# loading the sentence embedding model
# all-MiniLM-L6-v2 is small and fast
# dimension = 384
model = SentenceTransformer("all-MiniLM-L6-v2")

# build the FAISS index
def build_faiss_index():
    # load dataset
    df = pd.read_csv(DATASET_PATH)
    
    # get report texts
    reports = df["report"].tolist()
    
    print("Generating embeddings...")
    
    # create embeddings for all reports
    embeddings = model.encode(reports, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    
    # create FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # add embeddings to index
    index.add(embeddings)
    
    # save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # save report texts
    with open(REPORTS_PATH, "wb") as f:
        pickle.dump(reports, f)
    print("FAISS index built successfully.")


if __name__ == "__main__":
    build_faiss_index()