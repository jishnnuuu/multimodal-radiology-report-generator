"""
Multimodal Radiology Report Generator
======================================

Architecture
------------
    Chest X-Ray Image
        ↓
    CLIP ViT Vision Encoder  (frozen)
        ↓  [B, N_patches, vision_dim]
    Linear Projection Layer  (trainable)
        ↓  [B, N_patches, text_dim]
    ┌── concatenate ──────────────────────────────┐
    │  Visual Tokens  [B, N_patches, text_dim]    │
    │  Text Embeddings[B, seq_len,   text_dim]    │
    └─────────────────────────────────────────────┘
        ↓  [B, N_patches + seq_len, text_dim]
    FLAN-T5 Encoder-Decoder
        ↓
    Generated Radiology Report

Design Choices
--------------
1. CLIP vision encoder is FROZEN — IU-Xray has ~7k images; fine-tuning a ViT
   on such a small corpus causes severe overfitting.

2. All patch tokens are retained (not pooled into one CLS vector) so that
   the cross-attention in T5's decoder can attend to specific lung regions.

3. A single nn.Linear projection bridges the vision and language embedding
   spaces even when their nominal dimensions match — the semantic spaces are
   different and must be explicitly aligned.

4. Dropout is added after projection for regularisation.

Fixes Applied
-------------
- `vision_encoder` was called inside `torch.no_grad()` in `forward()` but the
  grad context was inconsistently applied.  Now handled cleanly with a context
  manager that respects the frozen parameter state.
- Device consistency: `text_embeddings` is moved to the same device as
  `visual_tokens` rather than relying on the caller.
- `visual_mask` dtype and device now always match `attention_mask`.
- Added `dropout` after projection for regularisation.
- `model_dim` property exposed for external use (e.g. beam search).
"""
"""
Multimodal Radiology Report Generator
=====================================

Goal
-----
Generate a radiology report directly from a chest X-ray image.

    Image → Report

------------------------------------------------------------

🚨 CRITICAL DESIGN FIX (IMPORTANT)

Earlier approach (WRONG):
--------------------------------
    Image + Ground Truth Report → Predict Report

Problem:
    - Model learns to copy text instead of using image
    - Loss becomes artificially very low
    - Generation becomes meaningless at inference

Correct approach (CURRENT):
--------------------------------
    Image → Predict Report

This forces the model to:
    - Actually learn visual understanding
    - Map visual features → medical language

------------------------------------------------------------

Architecture
------------

    Chest X-ray Image
        ↓
    CLIP Vision Encoder (FROZEN)
        ↓
    Visual Patch Embeddings
        ↓
    Projection Layer
        ↓
    FLAN-T5 Language Model
        ↓
    Generated Radiology Report

------------------------------------------------------------

Key Design Choices
------------------

1. CLIP Vision Encoder is FROZEN
   - Dataset is small (~7k images)
   - Prevents overfitting
   - Uses strong pretrained visual features

2. Patch Tokens (NOT pooled)
   - Each token = different region of image
   - Helps model attend to:
        lungs, heart, ribs, abnormalities

3. Projection Layer
   - Maps CLIP space → T5 embedding space
   - Even if dims match, semantics differ

4. NO TEXT INPUT during training
   - Prevents shortcut learning
   - Forces true multimodal learning

------------------------------------------------------------
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration


class MultimodalReportGenerator(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # ----------------------------------------------------
        # 1. Vision Encoder (CLIP)
        # ----------------------------------------------------
        # Extracts visual features from X-ray
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Freeze all parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size  # usually 768

        # ----------------------------------------------------
        # 2. Language Model (FLAN-T5)
        # ----------------------------------------------------
        # Generates radiology report
        self.language_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base"
        )

        text_dim = self.language_model.config.d_model  # usually 768

        # ----------------------------------------------------
        # 3. Projection Layer
        # ----------------------------------------------------
        # Aligns visual embeddings → language space
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.Dropout(dropout)
        )

    # --------------------------------------------------------
    # FORWARD (Training)
    # --------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,           # [B, 3, 224, 224]
        input_ids: torch.Tensor,        # (NOT USED, kept for compatibility)
        attention_mask: torch.Tensor,   # (NOT USED)
        labels: torch.Tensor | None = None,
    ):
        """
        Training Flow:
        ----------------
            Image → Visual Tokens → T5 → Predict Report

        labels:
            Ground truth report tokens
            - Padding tokens should be -100 (ignored in loss)
        """

        # Step 1: Extract visual features
        with torch.no_grad():  # CLIP is frozen
            vision_outputs = self.vision_encoder(pixel_values=images)

        visual_features = vision_outputs.last_hidden_state
        # Shape: [B, N_patches, vision_dim]

        # Step 2: Project to T5 space
        visual_tokens = self.visual_projection(visual_features)
        # Shape: [B, N_patches, text_dim]

        # Step 3: Pass ONLY visual tokens to T5
        outputs = self.language_model(
            inputs_embeds=visual_tokens,
            labels=labels
        )

        return outputs


    # --------------------------------------------------------
    # GENERATION (Inference)
    # --------------------------------------------------------
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,        # (NOT USED)
        attention_mask: torch.Tensor,   # (NOT USED)
        **generation_kwargs,
    ):
        """
        Inference Flow:
        ----------------
            Image → Visual Tokens → Generate Report

        generation_kwargs:
            num_beams, max_new_tokens, etc.
        """

        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)

        visual_features = vision_outputs.last_hidden_state

        # Project to language space
        visual_tokens = self.visual_projection(visual_features)

        # Generate report
        output_ids = self.language_model.generate(
            inputs_embeds=visual_tokens,
            **generation_kwargs
        )

        return output_ids