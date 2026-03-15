"""
Document Retrieval for Medical Knowledge Grounding
==============================================
Goal
-----
Given a query (for example a generated radiology report),
retrieve the most similar reports from the IU-Xray dataset.

These retrieved reports act as supporting medical knowledge.

Example
-------
Query:
"right lower lobe opacity pneumonia"

Retrieved reports:

1) "Right lower lobe consolidation suspicious for pneumonia."
2) "Patchy opacity in the lower lung field."
3) "Findings consistent with right lower lobe pneumonia."


These reports become:
    retrieved_docs

which are later used for:
    hallucination detection
    knowledge grounding
"""


import pickle

# path to saved BM25 index
INDEX_PATH = "retrieval/bm25_index.pkl"


# load retrieval index from disk
def load_index():
    """
    Load the BM25 model and report list from disk.
    """
    # open saved index file
    with open(INDEX_PATH, "rb") as f:
        # load stored objects
        bm25, reports = pickle.load(f)
    # return BM25 model + reports
    return bm25, reports


#retrieve documents based on query
def retrieve_documents(query, top_k=5):
    """
    Retrieve the top-k most similar reports.
    Parameters
    ----------
    query : str
        search query (generated report or keywords)

    top_k : int
        number of documents to retrieve
    """

    # load BM25 model and corpus
    bm25, reports = load_index()

    # tokenize query for BM25
    tokenized_query = query.lower().split()

    # compute similarity scores with all reports
    scores = bm25.get_scores(tokenized_query)

    # rank reports by score (highest first)
    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # select top-k reports
    results = [reports[i] for i in ranked[:top_k]]

    # return retrieved documents
    return results