"""
Hallucination Detection for Radiology Report Generation
========================================================

Goal
----
Detect medical claims in the generated report that are NOT supported by
any of the retrieved reference documents.

Algorithm
---------

    Generated Report  →  E_g  (set of canonical medical entities)
    Retrieved Docs    →  E_r  (union of entities across all docs)

    Unsupported entities : U = E_g − E_r
    Hallucination rate   : H = |U| / |E_g|   (0.0 = perfect, 1.0 = fully hallucinated)

Example
-------
    Generated : "The left atrium appears enlarged. Pleural effusion is present."

    E_g = {"cardiomegaly", "pleural effusion"}      (after synonym normalisation)
    E_r = {"cardiomegaly"}                          (from retrieved docs)

    U   = {"pleural effusion"}
    H   = 1/2 = 0.50

Fixes Applied
-------------
- Word-boundary matching: original code used `if phrase in text` which
  caused "lung" to match inside "lungs" or "prolonged".  Now uses
  `re.search(r'\bphrase\b')` with re.escape for safety.
- Zero-division: already guarded, but now also returns a typed dataclass
  rather than a bare tuple for cleaner downstream usage.
- `unsupported` set is sorted alphabetically in the result for reproducibility.
- Threshold constant centralised so callers can import it.
"""

import re
from dataclasses import dataclass, field
from medical_vocabulary import load_vocabulary


# ── Configuration ──────────────────────────────────────────────────────────────
VOCAB_PATH             = "dataset/radiology_vocabulary_final.xlsx"

# Reports with hallucination rate above this threshold are flagged
HALLUCINATION_THRESHOLD = 0.30


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass
class HallucinationResult:
    hallucination_rate:    float
    generated_entities:    set[str]
    retrieved_entities:    set[str]
    unsupported_entities:  set[str]
    is_flagged:            bool = field(init=False)

    def __post_init__(self):
        self.is_flagged = self.hallucination_rate > HALLUCINATION_THRESHOLD

    def summary(self) -> str:
        lines = [
            f"Hallucination rate    : {self.hallucination_rate:.2%}",
            f"Generated entities    : {sorted(self.generated_entities)}",
            f"Retrieved entities    : {sorted(self.retrieved_entities)}",
            f"Unsupported entities  : {sorted(self.unsupported_entities)}",
            f"Flagged               : {'YES ⚠' if self.is_flagged else 'NO  ✓'}",
        ]
        return "\n".join(lines)


# ── Vocabulary ─────────────────────────────────────────────────────────────────
_term_map: dict[str, str] | None = None

def _get_term_map() -> dict[str, str]:
    global _term_map
    if _term_map is None:
        _term_map = load_vocabulary(VOCAB_PATH)
    return _term_map


# ── Entity Extraction ──────────────────────────────────────────────────────────
def extract_entities(text: str) -> set[str]:
    """
    Extract canonical medical entities from text using the radiology vocabulary.

    Uses word-boundary regex matching to avoid partial matches:
        "lung" does NOT match inside "prolonged" or "lungs".
    """
    text     = text.lower()
    term_map = _get_term_map()
    entities: set[str] = set()

    for phrase, canonical in term_map.items():
        # re.escape handles multi-word phrases and special characters safely
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text):
            entities.add(canonical)

    return entities


# ── Hallucination Detection ────────────────────────────────────────────────────
def compute_hallucination_rate(
    generated_report: str,
    retrieved_docs: list[str],
) -> HallucinationResult:
    """
    Compute the hallucination rate for a generated radiology report.

    Parameters
    ----------
    generated_report : str
        The model-generated radiology report text.
    retrieved_docs : list[str]
        Radiology reports retrieved from the training corpus via FAISS.

    Returns
    -------
    HallucinationResult
        Dataclass with rate, entity sets, and a boolean flag.
    """
    generated_entities: set[str] = extract_entities(generated_report)

    retrieved_entities: set[str] = set()
    for doc in retrieved_docs:
        retrieved_entities |= extract_entities(doc)

    unsupported = generated_entities - retrieved_entities

    if not generated_entities:
        # No medical entities detected — cannot compute meaningful rate
        rate = 0.0
    else:
        rate = len(unsupported) / len(generated_entities)

    return HallucinationResult(
        hallucination_rate=rate,
        generated_entities=generated_entities,
        retrieved_entities=retrieved_entities,
        unsupported_entities=unsupported,
    )