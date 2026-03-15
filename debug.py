import torch

from dataset_loader import create_dataloader
from models.multimodal_model import MultimodalReportGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
loader = create_dataloader()

batch = next(iter(loader))

images = batch["image"].to(device)
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["labels"].to(device)

print("Image shape:", images.shape)
print("Token shape:", input_ids.shape)

print("Loading model...")
model = MultimodalReportGenerator().to(device)

print("Running forward pass...")

outputs = model(
    images=images,
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

print("Loss:", outputs.loss.item())
print("Logits shape:", outputs.logits.shape)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print("Trainable params:", trainable)
print("Total params:", total)

print("Debug successful!")