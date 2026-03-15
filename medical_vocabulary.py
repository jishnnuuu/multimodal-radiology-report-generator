import pandas as pd

"""
Load radiology vocabulary from Excel
and build synonym → canonical term mapping.
"""

def load_vocabulary(path):
    df = pd.read_excel(path)
    term_map = {}
    for _, row in df.iterrows():
        canonical = str(row["Term"]).lower().strip()
        # add canonical term itself
        term_map[canonical] = canonical
        # iterate through synonym columns
        for col in df.columns:
            if "Synonym" in col:
                value = row[col]
                if pd.isna(value):
                    continue
                synonyms = str(value).split(";")
                for s in synonyms:
                    s = s.strip().lower()
                    if s:
                        term_map[s] = canonical
    return term_map