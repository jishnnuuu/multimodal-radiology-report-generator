import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.multimodal_model import MultimodalReportGenerator


"""
Quick sanity test before full training.

Goal:
    Verify the multimodal pipeline works correctly.

Steps:
    Load single X-ray
    Pass through model
    Check loss and logits
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = MultimodalReportGenerator().to(device)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])


IMAGE_PATH = "dataset/NLMCXR_png/CXR1_1_IM-0001-3001.png"  # change to any image
print("Loading image:", IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

prompt = "generate radiology report"

tokens = tokenizer(
    prompt,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100
labels = labels.to(device)

print("Running forward pass...")

with torch.no_grad():

    outputs = model(
        images=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

print("Loss:", outputs.loss.item())
print("Logits shape:", outputs.logits.shape)


print("Sanity test complete.")