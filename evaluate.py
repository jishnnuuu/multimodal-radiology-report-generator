"""
EVALUATION PIPELINE — MULTIMODAL RADIOLOGY REPORT GENERATION

Goal
-----
Evaluate model performance using:

    ✔ BLEU Score        → n-gram overlap
    ✔ ROUGE-L           → sequence similarity
    ✔ BERTScore         → semantic similarity
    ✔ Hallucination Rate→ factual consistency

Pipeline
--------
Image → Generate Report → Compare with Ground Truth → Compute Metrics

Output
------
Average metrics over dataset:
    BLEU-4
    ROUGE-L
    BERTScore (F1)
    Hallucination Rate
"""

import torch
import pandas as pd
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from dataset_loader import CSV_PATH
from inference import load_model, preprocess_image, generate_report
from hallucination_detector import compute_hallucination_rate

from bert_score import BERTScorer

from retrieval.retrieve_faiss import retrieve_documents


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MAX_SAMPLES = 500   # limit for faster evaluation (set None for full dataset)


# ─────────────────────────────────────────────────────────────
# METRIC INITIALIZATION
# ─────────────────────────────────────────────────────────────
smooth_fn = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────
def evaluate():

    # Load dataset
    df = pd.read_csv(CSV_PATH)

    if MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42)

    # Load model
    model, device = load_model()

    bleu_scores = []
    rouge_scores = []
    bert_scores = []
    hallucination_rates = []

    print("\nStarting Evaluation...\n")
    
    scorer = BERTScorer(lang="en", device=device)

    for _, row in tqdm(df.iterrows(), total=len(df)):

        image_path = row["image_path"]
        ground_truth = row["report"]

        try:
            # Preprocess image
            image = preprocess_image(image_path, device)

            # Generate report
            generated = generate_report(model, image, device)

            # Remove prompt if present
            generated = generated.replace("Generate a radiology report:", "").strip()

            # ───────────── BLEU ─────────────
            bleu = sentence_bleu(
                [ground_truth.split()],
                generated.split(),
                smoothing_function=smooth_fn
            )
            bleu_scores.append(bleu)

            # ───────────── ROUGE-L ─────────────
            rouge_score_val = rouge.score(ground_truth, generated)["rougeL"].fmeasure
            rouge_scores.append(rouge_score_val)

            # ───────────── BERTScore ─────────────
            P, R, F1 = scorer.score([generated], [ground_truth])
            bert_scores.append(F1.item())

            # ───────────── Hallucination ─────────────
            docs, _ = retrieve_documents(generated, top_k=5)

            result = compute_hallucination_rate(
                generated,
                docs
            )
            hallucination_rates.append(result.hallucination_rate)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # ─────────────────────────────────────────────
    # FINAL METRICS
    # ─────────────────────────────────────────────
    print("\n================ FINAL METRICS ================\n")

    print(f"BLEU-4 Score        : {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"ROUGE-L Score       : {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"BERTScore (F1)      : {sum(bert_scores)/len(bert_scores):.4f}")
    print(f"Hallucination Rate  : {sum(hallucination_rates)/len(hallucination_rates):.4f}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate()