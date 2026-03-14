import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

REPORTS_FOLDER = "dataset/ecgen-radiology"
IMAGES_FOLDER = "dataset/NLMCXR_png"
OUTPUT_CSV = "dataset/iu_xray_dataset.csv"

def clean_report(text):
    if text is None:
        return ""
    # remove anonymization tokens
    text = text.replace("XXXX", "")
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Parse Single XML File
def parse_xml_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    findings = ""
    impression = ""
    
    # extract report sections
    for abstract in root.iter("AbstractText"):
        label = abstract.attrib.get("Label")
        
        if label == "FINDINGS":
            findings = abstract.text
        elif label == "IMPRESSION":
            impression = abstract.text
    findings = clean_report(findings)
    impression = clean_report(impression)
    
    report = findings + " " + impression
    
    # extract image ids
    image_ids = []
    for img in root.iter("parentImage"):
        image_id = img.attrib.get("id")
        image_ids.append(image_id)
        
    # create dataset rows
    rows = []
    for img_id in image_ids:
        image_file = img_id + ".png"
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        rows.append({
            "image_id": img_id,
            "image_path": image_path,
            "report": report
        })
    return rows

#process entire dataset
def build_dataset():
    all_rows = []
    xml_files = [
        f for f in os.listdir(REPORTS_FOLDER)
        if f.endswith(".xml")
    ]
    print(f"Found {len(xml_files)} XML reports")
    for file in tqdm(xml_files):
        xml_path = os.path.join(REPORTS_FOLDER, file)
        rows = parse_xml_file(xml_path)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    return df

def main():
    os.makedirs("dataset", exist_ok=True)
    df = build_dataset()
    print("\nDataset size:", len(df))
    print("\nSample rows:")
    print(df.head())
    df.to_csv(OUTPUT_CSV, index=False)
    print("\nSaved dataset to:", OUTPUT_CSV)

if __name__ == "__main__":
    main()