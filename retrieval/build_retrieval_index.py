"""
Goal
-----
Create a searchable knowledge base from IU-Xray reports.

Later in the pipeline we will retrieve reports that are
similar to the generated report.

This enables Retrieval-Augmented Generation (RAG).

Example pipeline:

Generated Report
        ↓
Convert to query
        ↓
BM25 search
        ↓
Retrieve similar radiology reports
        ↓
Use them as supporting evidence


Why IU-Xray reports?
--------------------
IU-Xray dataset already contains thousands of expert-written
radiology reports. These reports act as a medical knowledge base.


Example dataset row
-------------------

Report:
"The lungs are clear without focal opacity.
No pleural effusion or pneumothorax."

These reports will be indexed for retrieval.


Output
------
This script produces a file:
    retrieval/bm25_index.pkl

which contains:
    BM25 search model
    All radiology reports


We run this script ONLY ONCE to build the index.
"""

import pandas as pd

# pickle → used to save the BM25 index to disk
import pickle

# BM25 implementation
from rank_bm25 import BM25Okapi

# Path to dataset containing image paths + reports
DATASET_PATH = "dataset/iu_xray_dataset.csv"

# File where the BM25 index will be stored
INDEX_PATH = "retrieval/bm25_index.pkl"


# tokenization function for BM25
def tokenize(text):
    """
    Convert text into tokens for BM25.
    BM25 works on tokenized documents.
    Example:
        "Right lower lobe opacity"
    becomes
        ["right", "lower", "lobe", "opacity"]
    """
    # convert text to lowercase and split by spaces
    return text.lower().split()


#build retrieval index using BM25
def build_index():
    # Load dataset CSV file
    df = pd.read_csv(DATASET_PATH)
    # Extract all radiology reports
    reports = df["report"].tolist()
    # Tokenize every report for BM25
    tokenized_corpus = [tokenize(r) for r in reports]
    # Create BM25 retrieval model
    bm25 = BM25Okapi(tokenized_corpus)
    # Save the index and reports together
    with open(INDEX_PATH, "wb") as f:
        # store both BM25 model and reports
        pickle.dump((bm25, reports), f)
    print("BM25 index built and saved.")

if __name__ == "__main__":
    build_index()