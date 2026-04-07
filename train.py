import os
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dataset_loader import create_dataloader
from models.multimodal_model import MultimodalReportGenerator


"""
Train the multimodal radiology report generation model.

Pipeline
------------------------------------------------
Dataset → DataLoader → Model → Loss → Backpropagation

    Image
    ↓
    Vision Encoder (CLIP)
    ↓
    Patch Tokens
    ↓
    Projection Layer
    ↓
    FLAN-T5
    ↓
    Generate Radiology Report
------------------------------------------------


During Training, the system performs
    Load Dataset
        ↓
    Create DataLoader
        ↓
    Initialize Model
        ↓
    Move model to GPU
        ↓
    Create Optimizer
        ↓
    Enable Mixed Precision
        ↓
    Training Loop
        ↓
    Forward Pass
        ↓
    Compute Loss
        ↓
    Backward Pass
        ↓
    Update Parameters
        ↓
    Save Checkpoints
"""

#training configurations
EPOCHS = 20
LEARNING_RATE = 1e-5
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)


#load dataset
print("Loading dataset...")
dataloader = create_dataloader()
print("Dataset loaded.")


#initialize model
print("Loading model...")
model = MultimodalReportGenerator().to(device)
print("Model loaded.")


#optimizer
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


#mixed precision scaler
scaler = GradScaler(device=device.type)

# Initialize best_loss with infinity
best_loss = float('inf')

# training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    model.vision_encoder.eval() 
    total_loss = 0
    progress_bar = tqdm(dataloader)
    
    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        if torch.isnan(loss):
            print("NaN loss detected — check LR, labels, or input data")
            raise ValueError("NaN loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(progress_bar.n+1))
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # --- SAVE LOGIC ---
    
    # 1. Prepare the checkpoint data
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }

    # 2. Save the regular epoch checkpoint
    epoch_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
    torch.save(checkpoint, epoch_path)
    print(f"Checkpoint saved: {epoch_path}")

    # 3. Check if this is the BEST model so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"New best model found! Saved to: {best_path}")

print("\nTraining Complete.")