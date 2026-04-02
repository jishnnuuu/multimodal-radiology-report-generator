"""
Parse IU X-Ray XML Reports → CSV Dataset

Pipeline
--------
    XML Report Files
        ↓
    Extract Findings + Impression
        ↓
    Clean Text
        ↓
    Map Image IDs → Image Paths
        ↓
    Save as CSV (train / val / test splits)

Fixes Applied
-------------
- `clean_report` now handles None gracefully before calling .replace()
- Deduplicate rows by image_id to prevent training on identical samples
- Train/val/test split (80/10/10) saved as separate CSV files
- Skips rows where image file does not exist on disk
- Progress bar + summary statistics
"""

import os
import re
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ── Paths ──────────────────────────────────────────────────────────────────────
REPORTS_FOLDER = "dataset/ecgen-radiology"
IMAGES_FOLDER  = "dataset/NLMCXR_png"
OUTPUT_DIR     = "dataset"
OUTPUT_CSV     = os.path.join(OUTPUT_DIR, "iu_xray_dataset.csv")
TRAIN_CSV      = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV        = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV       = os.path.join(OUTPUT_DIR, "test.csv")


# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_report(text: str | None) -> str:
    """
    Sanitise a raw report section.

    Handles
    -------
    - None / missing fields (returns empty string)
    - Anonymisation tokens (XXXX)
    - Repeated whitespace / newlines
    """
    if not text:           # covers None, "", whitespace-only
        return ""
    text = text.replace("XXXX", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Single XML Parser ──────────────────────────────────────────────────────────
def parse_xml_file(xml_path: str) -> list[dict]:
    """
    Parse one XML report and return a list of row dicts.

    Each row corresponds to one image that appears in the report.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    findings   = ""
    impression = ""

    for abstract in root.iter("AbstractText"):
        label = abstract.attrib.get("Label", "")
        if label == "FINDINGS":
            findings = abstract.text
        elif label == "IMPRESSION":
            impression = abstract.text

    findings   = clean_report(findings)
    impression = clean_report(impression)

    # Combine sections; strip leading/trailing whitespace from join
    report = (findings + " " + impression).strip()

    # Skip records with no usable text
    if not report:
        return []

    image_ids = [
        img.attrib.get("id")
        for img in root.iter("parentImage")
        if img.attrib.get("id")
    ]

    rows = []
    for img_id in image_ids:
        image_path = os.path.join(IMAGES_FOLDER, img_id + ".png")
        # Only include rows whose image actually exists on disk
        if not os.path.exists(image_path):
            continue
        rows.append({
            "image_id":   img_id,
            "image_path": image_path,
            "report":     report,
        })

    return rows


# ── Full Dataset Builder ───────────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    xml_files = [
        f for f in os.listdir(REPORTS_FOLDER) if f.endswith(".xml")
    ]
    print(f"Found {len(xml_files)} XML reports")

    all_rows: list[dict] = []
    for fname in tqdm(xml_files, desc="Parsing XML"):
        xml_path = os.path.join(REPORTS_FOLDER, fname)
        all_rows.extend(parse_xml_file(xml_path))

    df = pd.DataFrame(all_rows)

    # Deduplicate: same image should not appear twice
    before = len(df)
    df = df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate image entries.")

    return df


# ── Train / Val / Test Split ───────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> None:
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,   index=False)
    test_df.to_csv(TEST_CSV,  index=False)

    print(f"\nSplit sizes  →  train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = build_dataset()

    print(f"\nTotal usable samples : {len(df)}")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))

    # Save full dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull dataset  → {OUTPUT_CSV}")

    # Save splits
    split_and_save(df)
    print("Splits saved.")


if __name__ == "__main__":
    main()