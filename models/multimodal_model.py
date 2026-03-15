"""
Multimodal model that combines a vision encoder and a text decoder for medical report generation.

The vision encoder processes the input chest X-ray image and extracts visual features
that describe the medical content of the image (lungs, heart, abnormalities, etc).

The language model (FLAN-T5) then generates the corresponding medical report
based on those extracted visual features.

The model learns a mapping between:
    visual patterns → medical language

Goal
----------------------------------------------------
Image → Radiology Report

Chest X-ray
    ↓
Vision Encoder (CLIP ViT)
    ↓
Visual Patch Embeddings
    ↓
Projection Layer
    ↓
Language Model (FLAN-T5)
    ↓
Generated Radiology Report
----------------------------------------------------

Why we need a PROJECTION LAYER
----------------------------------------------------
The vision encoder and language model operate in different embedding spaces.

Example:
    CLIP embedding dimension  = 768
    T5 embedding dimension    = 768

Even if the dimensions match, the **semantic spaces are different**.

The projection layer learns how to map:

    visual features → language embedding space

This allows the language model to interpret visual information correctly.

Training Flow
----------------------------------------------------
    Image
    ↓
    CLIP Vision Encoder
    ↓
    Visual Patch Tokens (each token = image region)
    ↓
    Projection Layer
    ↓
    Concatenate with Text Tokens
    ↓
    FLAN-T5 Transformer
    ↓
    Predict Next Token
----------------------------------------------------

Important Design Choice
----------------------------------------------------
Instead of compressing the image into ONE vector,
we keep ALL patch tokens.

This allows the model to attend to specific image regions.

Example:
    patch 12 → lung opacity
    patch 35 → enlarged heart
    patch 48 → pleural effusion

This significantly improves report generation quality.
"""

import torch
import torch.nn as nn

from transformers import CLIPVisionModel
from transformers import T5ForConditionalGeneration


class MultimodalReportGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        """
        ----------------------------------------------------
        1. Vision Encoder
        ----------------------------------------------------
        We use CLIP's Vision Transformer to extract features
        from the chest X-ray image.
        
        CLIP is pretrained on massive image-text datasets,
        so it already understands many visual patterns.
        
        Because our dataset (IU-Xray) is small (~7k images),
        we FREEZE the vision encoder to prevent overfitting.
        """

        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        vision_dim = self.vision_encoder.config.hidden_size

        # Freeze the vision encoder parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False


        """
        ----------------------------------------------------
        2. Language Model (FLAN-T5)
        ----------------------------------------------------
        This model generates the radiology report.

        It receives both:
            visual tokens
            text tokens

        and learns how to generate medical descriptions.
        """

        self.language_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base"
        )

        text_dim = self.language_model.config.d_model


        """
        ----------------------------------------------------
        3. Projection Layer
        ----------------------------------------------------
        Maps visual embeddings → language embedding space.

        This is the bridge between:
            vision model
            language model
        """

        self.visual_projection = nn.Linear(vision_dim, text_dim)


    def forward(self, images, input_ids, attention_mask, labels=None):

        """
        ----------------------------------------------------
        Step 1 — Extract Visual Features
        ----------------------------------------------------
        Pass the image through CLIP Vision Transformer.
        """
        with torch.no_grad():
            vision_outputs = self.vision_encoder(images)
        visual_features = vision_outputs.last_hidden_state
        """
        visual_features shape example:
            [batch_size, num_patches, vision_dim]
        Example:
            [8, 50, 768]
        Meaning:
            8 images
            50 patch tokens per image [49 patch tokens + 1 CLS token = 50]
            768-dimensional feature vector
        Each patch token represents a region of the X-ray.
        """
        
        
        """
        ----------------------------------------------------
        Step 2 — Project Visual Tokens
        ----------------------------------------------------
        Convert vision embeddings into language model space.
        """
        visual_tokens = self.visual_projection(visual_features)
        """
        shape:
            [8, 50, 768]
        Now these embeddings live in the same space
        as the T5 token embeddings.
        """
        
        
        """
        ----------------------------------------------------
        Step 3 — Convert Text Tokens → Embeddings
        ----------------------------------------------------
        input_ids are token indices.
        The embedding layer converts them into vectors.
        """
        # Convert text tokens → embeddings
        text_embeddings = self.language_model.shared(input_ids)
        text_embeddings = text_embeddings.to(visual_tokens.device) #ensure same device
        """
        shared because it's used for both input and output token embeddings in T5.
        Example shape:
            input_ids      → [8, 256]
            text_embeddings→ [8, 256, 768]
        """
        
        
        
        """
        ----------------------------------------------------
        Step 4 — Combine Vision + Text Tokens
        ----------------------------------------------------
        We concatenate visual tokens BEFORE text tokens.
        Final sequence becomes:
            [patch1 patch2 ... patch50 word1 word2 ... word256]
        """
        multimodal_embeddings = torch.cat(
            [visual_tokens, text_embeddings],
            dim=1
        )
        
        
        
        """
        ----------------------------------------------------
        Step 5 — Update Attention Mask
        ----------------------------------------------------
        The transformer must know which tokens are valid.
        We create an attention mask for the visual tokens.
        """
        batch_size = attention_mask.shape[0]
        num_visual_tokens = visual_tokens.shape[1]
        
        visual_mask = torch.ones(
            (batch_size, num_visual_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        
        multimodal_mask = torch.cat(
            [visual_mask, attention_mask],
            dim=1
        )
        
        outputs = self.language_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            labels=labels
        )
        return outputs