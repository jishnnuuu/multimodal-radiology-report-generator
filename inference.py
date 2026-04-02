"""
Inference Pipeline — Multimodal Radiology Report Generator
===========================================================

Pipeline
--------
    Chest X-Ray Image (PNG / JPG)
        ↓
    Preprocessing  (resize → tensor → normalise)
        ↓
    MultimodalReportGenerator.generate()
        ↓  beam search, repetition penalty
    Generated Radiology Report (text)
        ↓
    FAISS Semantic Retrieval
        ↓  top-5 similar training reports
    Hallucination Detection
        ↓  entity comparison
    Structured Inference Result

Output Fields
-------------
    generated_report  : str
    retrieved_docs    : list[str]
    retrieval_scores  : list[float]
    hallucination     : HallucinationResult

Fixes Applied
-------------
- Original code manually reconstructed the multimodal embedding inside
  inference.py, duplicating model internals and risking divergence.
  Now calls `model.generate()` which lives in multimodal_model.py.
- Added beam search (num_beams=4) and repetition penalty (1.5) to avoid
  degenerate repetitive outputs — a known failure mode of T5 on short
  training corpora.
- Added no_repeat_ngram_size=3 to prevent 3-gram repetition.
- Model loaded from BEST checkpoint by default (not last epoch).
- `load_model` is separated from `run_inference` for testability.
- Returns a dataclass rather than printing raw strings; caller decides display.
"""

import os
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.multimodal_model import MultimodalReportGenerator
from retrieval.retrieve_faiss import retrieve_documents
from hallucination_detector import (
    compute_hallucination_rate,
    HallucinationResult,
    HALLUCINATION_THRESHOLD,
)


# ── Configuration ──────────────────────────────────────────────────────────────
BEST_CHECKPOINT  = "checkpoints/best_model.pt"
TOKENIZER_NAME   = "google/flan-t5-base"
IMAGE_SIZE       = 224

# Generation hyper-parameters
NUM_BEAMS             = 4
MAX_NEW_TOKENS        = 256
NO_REPEAT_NGRAM_SIZE  = 3
REPETITION_PENALTY    = 1.5
LENGTH_PENALTY        = 1.0    # > 1 encourages longer output

# Retrieval
RETRIEVAL_TOP_K = 5

# Prompt prefix
PROMPT = "generate a detailed chest x-ray radiology report:"


# ── Result Container ───────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    generated_report: str
    retrieved_docs:   list[str]
    retrieval_scores: list[float]
    hallucination:    HallucinationResult

    def display(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("GENERATED REPORT")
        print(sep)
        print(self.generated_report)

        print(f"\n{sep}")
        print(f"RETRIEVED REPORTS  (top {len(self.retrieved_docs)})")
        print(sep)
        for i, (doc, score) in enumerate(
            zip(self.retrieved_docs, self.retrieval_scores), 1
        ):
            print(f"\n[{i}]  similarity={score:.3f}")
            print(doc[:300] + ("…" if len(doc) > 300 else ""))

        print(f"\n{sep}")
        print("HALLUCINATION ANALYSIS")
        print(sep)
        print(self.hallucination.summary())

        if self.hallucination.is_flagged:
            print(
                f"\n⚠  Hallucination rate ({self.hallucination.hallucination_rate:.1%}) "
                f"exceeds threshold ({HALLUCINATION_THRESHOLD:.0%})."
                "\n   Consider re-running with a higher retrieval top-k or "
                "refining the generation prompt."
            )


# ── Model Loading ──────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str = BEST_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[MultimodalReportGenerator, torch.device]:
    """
    Load the trained MultimodalReportGenerator from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
    device          : torch.device  (auto-detected if None)

    Returns
    -------
    (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with: python train.py"
        )

    model = MultimodalReportGenerator().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded from  : {checkpoint_path}")
    print(f"Device             : {device}")
    if "val_loss" in ckpt:
        print(f"Checkpoint val loss: {ckpt['val_loss']:.4f}")

    return model, device


# ── Image Preprocessing ────────────────────────────────────────────────────────
_eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """Load, preprocess, and add batch dimension to a chest X-ray image."""
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return _eval_transform(image).unsqueeze(0).to(device)


# ── Report Generation ──────────────────────────────────────────────────────────
def generate_report(
    model: MultimodalReportGenerator,
    image: torch.Tensor,
    device: torch.device,
) -> str:
    """
    Run beam-search generation on a single preprocessed image.

    Parameters
    ----------
    model  : loaded MultimodalReportGenerator
    image  : [1, 3, 224, 224] preprocessed image tensor
    device : target device

    Returns
    -------
    str : decoded radiology report
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokens = tokenizer(
        PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=64,           # prompt is short — no need for 256
        truncation=True,
    )
    input_ids      = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    output_ids = model.generate(
        images=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        # ── Generation quality controls ────────────────────────────────────
        num_beams=NUM_BEAMS,
        max_new_tokens=MAX_NEW_TOKENS,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=REPETITION_PENALTY,
        length_penalty=LENGTH_PENALTY,
        early_stopping=True,
    )

    report = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return report


# ── Full Inference Pipeline ────────────────────────────────────────────────────
def run_inference(
    image_path: str,
    checkpoint_path: str = BEST_CHECKPOINT,
) -> InferenceResult:
    """
    End-to-end inference: image → report → retrieval → hallucination check.

    Parameters
    ----------
    image_path      : path to chest X-ray PNG or JPG
    checkpoint_path : path to model checkpoint

    Returns
    -------
    InferenceResult
    """
    model, device = load_model(checkpoint_path)

    image  = preprocess_image(image_path, device)
    report = generate_report(model, image, device)

    docs, scores = retrieve_documents(report, top_k=RETRIEVAL_TOP_K)

    hallucination = compute_hallucination_rate(report, docs)

    return InferenceResult(
        generated_report=report,
        retrieved_docs=docs,
        retrieval_scores=scores,
        hallucination=hallucination,
    )


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run radiology report inference")
    parser.add_argument("image_path", type=str, help="Path to chest X-ray image")
    parser.add_argument(
        "--checkpoint", type=str, default=BEST_CHECKPOINT,
        help=f"Model checkpoint path (default: {BEST_CHECKPOINT})"
    )
    args = parser.parse_args()

    result = run_inference(args.image_path, checkpoint_path=args.checkpoint)
    result.display()