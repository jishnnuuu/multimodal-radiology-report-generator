"""
INFERENCE PIPELINE FOR MULTIMODAL RADIOLOGY REPORT GENERATION

Goal
-----
Run the trained multimodal model to generate a report
for a new chest X-ray image.

Pipeline
--------

    Input Image
        ↓
    CLIP Vision Encoder
        ↓
    Projection Layer
        ↓
    FLAN-T5 Language Model
        ↓
    Generated Radiology Report
        ↓
    Retrieve Supporting Documents
        ↓
    Hallucination Detection


Output
------

Generated Report
Retrieved Supporting Reports
Hallucination Rate
Unsupported Medical Entities
"""

import torch

# image processing
from PIL import Image

# image transformations
from torchvision import transforms

# tokenizer for T5
from transformers import AutoTokenizer


# import multimodal model
from models.multimodal_model import MultimodalReportGenerator

# import retrieval system
from retrieval.retrieve_docs import retrieve_documents

# import hallucination detection
from hallucination_detector import compute_hallucination_rate



# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#path to trained model checkpoint
MODEL_PATH = "checkpoints/model_epoch_5.pt"


# load the trained model
def load_model():
    # initialize model architecture
    model = MultimodalReportGenerator().to(device)
    
    # load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # switch to evaluation mode
    model.eval()
    
    return model


# generate reports from images
def generate_report(model, image):
    
    # load tokenizer used during training
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # prompt to guide the language model
    prompt = "generate radiology report"
    
    # tokenize prompt
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=256
    )
    
    # move tensors to device
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    # disable gradients for inference
    with torch.no_grad():
        # generate text using T5 decoder
        outputs = model.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256
        )
        
    # convert tokens back to text
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return report


# -------------------------------------------------------------
# Full inference pipeline
# -------------------------------------------------------------

def run_inference(image_path):
    # load trained model
    model = load_model()
    
    # image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    ])
    
    # load image
    image = Image.open(image_path).convert("RGB")
    
    # apply transformations
    image = transform(image).unsqueeze(0).to(device)
    
    # generate radiology report
    report = generate_report(model, image)
    
    # retrieve supporting reports
    retrieved_docs = retrieve_documents(report)
    
    # compute hallucination rate
    rate, unsupported = compute_hallucination_rate(
        report,
        retrieved_docs
    )
    
    # generated reports
    print("\nGenerated Report:\n")
    print(report)
    
    print("\nRetrieved Supporting Reports:\n")
    
    for doc in retrieved_docs:
        print("-", doc)
        
    print("\nHallucination Rate:", rate)
    
    print("Unsupported Entities:", unsupported)


# run inference
if __name__ == "__main__":
    # run pipeline for a sample image
    run_inference("sample_xray.png")