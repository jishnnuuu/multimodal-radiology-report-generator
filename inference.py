"""
Multimodal Radiology Report Generation — Inference Pipeline

Goal
-----
Run the trained multimodal model to generate a radiology report
from a chest X-ray image and verify the report using retrieval
and hallucination detection.

Pipeline
--------

    Input Image
        ↓
    CLIP Vision Encoder
        ↓
    Visual Patch Tokens
        ↓
    Projection Layer
        ↓
    FLAN-T5 Language Model
        ↓
    Generated Radiology Report
        ↓
    FAISS Semantic Retrieval
        ↓
    Retrieve Similar Radiology Reports
        ↓
    Hallucination Detection
        ↓
    Final Report Verification


Output
------

Generated Radiology Report
Retrieved Supporting Reports
Hallucination Rate
Unsupported Medical Entities


Why Retrieval?
--------------

LLMs can hallucinate medical findings.

We retrieve similar radiology reports from IU-Xray dataset
to verify whether the generated findings are supported.


Example Flow
------------

Generated Report:

    "Opacity in right lower lobe consistent with pneumonia."

Retrieved Reports:

    "Right lower lobe consolidation suspicious for pneumonia."
    "Patchy opacity in lower lung field."

Hallucination Detector:

    Generated entities: {pneumonia, opacity}
    Retrieved entities: {pneumonia, opacity}

Hallucination Rate: 0.0

This means the generated report is supported by retrieved evidence.
"""

# ------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------

# PyTorch
import torch

# Image loading
from PIL import Image

# Image preprocessing
from torchvision import transforms

# Tokenizer for FLAN-T5
from transformers import AutoTokenizer

# Multimodal model
from models.multimodal_model import MultimodalReportGenerator

# FAISS semantic retrieval
from retrieval.retrieve_faiss import retrieve_documents

# Hallucination detection module
from hallucination_detector import compute_hallucination_rate




# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Path to trained model checkpoint
MODEL_PATH = "checkpoints/model_epoch_5.pt"

# Load tokenizer only once to avoid repeated initialization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


#load the trained model from disk
def load_model():
    """
    Load the trained multimodal model.
    """
    # Initialize model architecture
    model = MultimodalReportGenerator().to(device)
    # Load checkpoint file
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Load trained parameters
    model.load_state_dict(checkpoint["model_state_dict"])
    # Set model to evaluation mode
    model.eval()
    return model



# generate radiology report from image
def generate_report(model, image):
    """
    Generate radiology report from a chest X-ray image.
    
    Steps
    -----
    1) Extract visual tokens from CLIP
    2) Project visual tokens to language embedding space
    3) Create text embeddings for prompt
    4) Concatenate visual + text tokens
    5) Run FLAN-T5 generation
    """
    
    # Prompt to guide language generation
    prompt = "generate radiology report"
    
    # Convert prompt into token IDs
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=256
    )
    
    # Move tokens to device
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    # Disable gradient tracking during inference
    with torch.no_grad():
        
        # extract visual features using CLIP vision encoder
        vision_outputs = model.vision_encoder(image)
        
        # Patch embeddings
        visual_features = vision_outputs.last_hidden_state
        
        #project visual features to language embedding space
        visual_tokens = model.visual_projection(visual_features)
        
        #convert context prompt into text embeddings
        text_embeddings = model.language_model.shared(input_ids).to(device)
        
        # concatenate visual tokens and text embeddings to create multimodal input
        multimodal_embeddings = torch.cat(
            [visual_tokens, text_embeddings],
            dim=1
        )
        
        # create multimodal attention mask
        batch_size = attention_mask.shape[0]
        
        num_visual_tokens = visual_tokens.shape[1]
        
        visual_mask = torch.ones(
            (batch_size, num_visual_tokens),
            dtype=attention_mask.dtype,
            device=device
        )
        
        multimodal_mask = torch.cat(
            [visual_mask, attention_mask],
            dim=1
        )
        
        # generate report using FLAN-T5 language model
        outputs = model.language_model.generate(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            max_length=256
        )
        
    # Convert tokens back to readable text
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return report


# full inference pipeline
def run_inference(image_path):
    """
    Run the complete inference pipeline.
    """
    
    # Load trained model
    model = load_model()
    
    # image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    # Load chest X-ray image
    image = Image.open(image_path).convert("RGB")
    
    # Apply preprocessing
    image = transform(image).unsqueeze(0).to(device)
    
    #generate radiology report from image
    report = generate_report(model, image)
    
    # retrieve similar reports from IU-Xray dataset using FAISS
    retrieved_docs = retrieve_documents(report)
    
    # hallucination detection: compare generated report with retrieved reports
    rate, unsupported = compute_hallucination_rate(
        report,
        retrieved_docs
    )
    
    # display results
    print("\n================ GENERATED REPORT ================\n")
    print(report)
    print("\n================ RETRIEVED REPORTS ================\n")
    for doc in retrieved_docs:
        print("-", doc)
    print("\n================ HALLUCINATION ANALYSIS ================\n")
    print("Hallucination Rate:", rate)
    print("Unsupported Entities:", unsupported)


if __name__ == "__main__":

    # Replace with your test X-ray image
    run_inference("sample_xray.png")