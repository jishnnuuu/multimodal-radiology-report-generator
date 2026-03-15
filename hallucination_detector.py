import re
from medical_vocabulary import load_vocabulary

"""
Hallucination Detection for Radiology Report Generation
==============================================
Goal
-----
Detect when the generated radiology report contains medical claims
that are not supported by retrieved medical knowledge.

This prevents the model from hallucinating diseases.

Pipeline
--------
Generated Report
        ↓
Extract Medical Entities
        ↓
Retrieve Supporting Medical Documents (RAG)
        ↓
Extract Entities From Retrieved Docs
        ↓
Compare Entities
        ↓
Detect Unsupported Claims
        ↓
Compute Hallucination Rate


Example
====================================================================

1) Generated Radiology Report
----------------------------------------------------

"The left atrium appears enlarged.
Pleural effusion is present."


2) Retrieved Medical Documents
----------------------------------------------------

Doc 1:
"Enlargement of the heart atria may occur in cardiac disease."

Doc 2:
"The lungs appear clear with no signs of pneumonia."


3) Extract Entities Using Radiology Vocabulary
----------------------------------------------------

Generated entities (E_g):

    {"heart atria", "pleural effusion"}

Explanation:
    "left atrium" → normalized to "heart atria"


Retrieved entities (E_r):

    {"heart atria", "lung"}


4) Detect Unsupported Entities
----------------------------------------------------

Unsupported entities:

    U = E_g − E_r

    {"pleural effusion"}

Meaning:
    The generated report mentions pleural effusion,
    but no retrieved document supports it.


5) Compute Hallucination Rate
----------------------------------------------------

Hallucination Rate Formula:

    H = |U| / |E_g|

where

    E_g = generated entities
    U   = unsupported entities


Example calculation:

    E_g = {"heart atria", "pleural effusion"} → 2 entities
    U   = {"pleural effusion"}                → 1 entity


    H = 1 / 2 = 0.5


Interpretation
----------------------------------------------------

Hallucination Rate = 0.5

Meaning:
    50% of the medical claims in the report are unsupported.


6) Possible System Action
----------------------------------------------------

If hallucination rate is high:

    if H > 0.3:
        regenerate_report_with_more_context()

This forces the system to retrieve more medical knowledge
before generating the final report.


Why This Is Important
----------------------------------------------------

Medical AI must avoid unsupported diagnoses.

Hallucination detection helps ensure that the generated
radiology report is grounded in medical evidence.
"""


VOCAB_PATH = "dataset/radiology_vocabulary_final.xlsx"
term_map = load_vocabulary(VOCAB_PATH)

def extract_entities(text):
    text = text.lower()
    entities = set()
    for phrase, canonical in term_map.items():
        if phrase in text:
            entities.add(canonical)
    return entities


def compute_hallucination_rate(generated_report, retrieved_docs):
    generated_entities = extract_entities(generated_report)
    retrieved_entities = set()
    for doc in retrieved_docs:
        retrieved_entities |= extract_entities(doc)
    unsupported = generated_entities - retrieved_entities
    if len(generated_entities) == 0:
        return 0.0, unsupported
    hallucination_rate = len(unsupported) / len(generated_entities)
    return hallucination_rate, unsupported