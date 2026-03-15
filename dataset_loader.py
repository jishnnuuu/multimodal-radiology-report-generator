import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoTokenizer

"""
    Image --> Tensor
    Text --> Token IDs
    
    csv dataset --> load image file --> tranform image --> Tennsor
                                    --> tokenize report --> token ids
                                    --> return image tensor + token ids
                                    
                                    
    final sample returned to model
        {
            image : [3 x 224 x 224]
            input_ids : [256]
            attention_mask : [256]
            labels : [256]
        }
    
    dataloader --> batch of samples
        batch["image"] → [8 x 3 x 224 x 224]
        batch["input_ids"] → [8 x 256]
        
        
    Load CSV dataset
        ↓
    Create tokenizer
        ↓
    Create Dataset class
        ↓
    Dataset loads image + report
        ↓
    Image → preprocessing
    Report → tokenization
        ↓
    Return tensors
        ↓
    DataLoader groups them into batches
        ↓
    Training loop will consume batches
"""



CSV_PATH = "dataset/iu_xray_dataset.csv"

IMAGE_SIZE = 224        #224 x 224
MAX_TEXT_LENGTH = 256   #reports are truncated/padded to 256 tokens
BATCH_SIZE = 8          #number of samples per batch

#tokenizer
TOKENIZER_NAME = "google/flan-t5-base"

#image transformations
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), #converts PIL image to tensor and scales pixel values to [0, 1]
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ) #converts pixel values from [0, 1] to [-1, 1]
]) 


#custom dataset class
class IUXrayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform=None):
        """
        dataframe → CSV data
        tokenizer → text tokenizer
        transform → image preprocessing
        """
        
        self.df = dataframe
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        report = row["report"]
        
        # Load image
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        
        #applies image transformations
        if self.transform:
            image = self.transform(image)
        
        # Tokenize report
        tokens = self.tokenizer(
            report,
            padding="max_length", #short reports get padded to 256 tokens
            truncation=True,      #long reports get truncated to 256 tokens
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"   #returns pytorch tensors
        )
        
        input_ids = tokens["input_ids"].squeeze(0)              #remove batch dimension --> [1 x 256] → [256]
        attention_mask = tokens["attention_mask"].squeeze(0)    #remove batch dimension --> [1 x 256] → [256]
        ## mask indicates: 1 → real token, 0 → padding
        
        labels = input_ids.clone()
        """
        when the report is padded,
        we do not want the model to learn to predict <PAD>
        so we set those padding token labels to -100, and pytorch cross-entropy loss will ignore them during loss calculation.
        """
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

#create batch generator
def create_dataloader():
    df = pd.read_csv(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    dataset = IUXrayDataset(
        df,
        tokenizer,
        transform=image_transform
    )
    
    pin_memory = torch.cuda.is_available()  #if using GPU, pin memory for faster transfer
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory  # enables faster GPU transfer by allocating page-locked memory
    )
    
    return loader

# just a sanity check to see if dataloader is working correctly
def test_loader():
    loader = create_dataloader()
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("Image tensor shape:", batch["image"].shape)
    print("Token shape:", batch["input_ids"].shape)


if __name__ == "__main__":
    test_loader()