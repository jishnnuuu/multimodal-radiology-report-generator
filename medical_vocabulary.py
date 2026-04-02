"""
Medical Vocabulary Loader
==========================

Loads a radiology terminology Excel file and builds a
synonym → canonical-term mapping used for entity extraction.

Excel Format Expected
---------------------
| Term          | Synonym_1         | Synonym_2   | … |
|---------------|-------------------|-------------|---|
| cardiomegaly  | enlarged heart    | cardiac …   |   |
| pleural effu… | fluid in pleura   | …           |   |

Fixes Applied
-------------
- Added file-existence check with a clear error message.
- Handles NaN values robustly (original relied on pd.isna which still
  needs the value to exist — added explicit str-cast guard).
- Vocabulary is cached after first load so repeated calls are O(1).
- Returns copy of cached dict to prevent external mutation.
"""

import logging
import os

import pandas as pd


logger = logging.getLogger(__name__)

# Module-level cache: path → term_map
_cache: dict[str, dict[str, str]] = {}


def load_vocabulary(path: str) -> dict[str, str]:
    """
    Build a synonym → canonical-term dictionary from an Excel file.

    Parameters
    ----------
    path : str
        Path to the vocabulary Excel file.

    Returns
    -------
    dict[str, str]
        Mapping of every surface form (including the canonical term itself)
        to its canonical form.  All keys and values are lowercase, stripped.

    Raises
    ------
    FileNotFoundError  if the Excel file does not exist.
    ValueError         if the file has no 'Term' column.
    """
    if path in _cache:
        return dict(_cache[path])   # return a copy

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Vocabulary file not found: {path}\n"
            "Download or place the radiology vocabulary Excel at the expected path."
        )

    df = pd.read_excel(path)

    if "Term" not in df.columns:
        raise ValueError(
            f"Vocabulary file '{path}' must contain a 'Term' column. "
            f"Found columns: {list(df.columns)}"
        )

    term_map: dict[str, str] = {}
    synonym_cols = [c for c in df.columns if "Synonym" in c]

    for _, row in df.iterrows():
        canonical = str(row["Term"]).strip().lower()
        if not canonical or canonical == "nan":
            continue

        # Map canonical term to itself
        term_map[canonical] = canonical

        # Map each synonym to canonical
        for col in synonym_cols:
            raw = row[col]
            if pd.isna(raw):
                continue
            for synonym in str(raw).split(";"):
                synonym = synonym.strip().lower()
                if synonym and synonym != "nan":
                    term_map[synonym] = canonical

    logger.info("Loaded %d vocabulary entries from %s", len(term_map), path)
    _cache[path] = term_map
    return dict(term_map)