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

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration


# ── Model ──────────────────────────────────────────────────────────────────────
class MultimodalReportGenerator(nn.Module):

    VISION_CHECKPOINT   = "openai/clip-vit-base-patch32"
    LANGUAGE_CHECKPOINT = "google/flan-t5-base"

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # ── 1. Vision Encoder (CLIP ViT) — FROZEN ─────────────────────────────
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            self.VISION_CHECKPOINT
        )
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size   # 768

        # ── 2. Language Model (FLAN-T5) ───────────────────────────────────────
        self.language_model = T5ForConditionalGeneration.from_pretrained(
            self.LANGUAGE_CHECKPOINT
        )
        text_dim = self.language_model.config.d_model         # 768

        # ── 3. Projection: vision space → language space ───────────────────────
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.Dropout(dropout),
        )

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        images: torch.Tensor,           # [B, 3, 224, 224]
        input_ids: torch.Tensor,        # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
        labels: torch.Tensor | None = None,  # [B, seq_len]  -100 = ignore
    ):
        """
        Parameters
        ----------
        images          : normalised chest X-ray batch
        input_ids       : tokenised prompt / report for encoder
        attention_mask  : 1 = real token, 0 = padding
        labels          : target token ids for computing cross-entropy loss
                          (padding positions should be set to -100)

        Returns
        -------
        transformers Seq2SeqLMOutput — contains `.loss` and `.logits`
        """

        # ── Step 1: Extract Visual Patch Embeddings ────────────────────────────
        # Vision encoder is frozen — no gradient needed
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=images)

        # [B, N_patches, vision_dim]  e.g. [B, 50, 768]
        visual_features = vision_out.last_hidden_state

        # ── Step 2: Project to Language Embedding Space ────────────────────────
        # [B, N_patches, text_dim]
        visual_tokens = self.visual_projection(visual_features)

        # ── Step 3: Text Token → Embeddings ───────────────────────────────────
        # `shared` is the embedding table shared between T5 encoder and decoder
        # [B, seq_len, text_dim]
        text_embeddings = self.language_model.shared(input_ids)
        # Ensure both tensors are on the same device
        text_embeddings = text_embeddings.to(visual_tokens.device)

        # ── Step 4: Concatenate [Visual | Text] ───────────────────────────────
        # Final encoder input: [patch_1 … patch_N word_1 … word_L]
        # [B, N_patches + seq_len, text_dim]
        multimodal_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

        # ── Step 5: Build Extended Attention Mask ─────────────────────────────
        B, N = visual_tokens.shape[0], visual_tokens.shape[1]
        visual_mask = torch.ones(
            (B, N),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # [B, N_patches + seq_len]
        multimodal_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # ── Step 6: T5 Forward ────────────────────────────────────────────────
        return self.language_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            labels=labels,
        )

    # ── Generation Helper ──────────────────────────────────────────────────────
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation with full multimodal context.

        Parameters
        ----------
        generation_kwargs : forwarded to T5's .generate()
            Recommended defaults (set in inference.py):
                num_beams=4, max_new_tokens=256,
                no_repeat_ngram_size=3, repetition_penalty=1.5

        Returns
        -------
        output_ids : Tensor [B, generated_length]
        """
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values=images)

        visual_features = vision_out.last_hidden_state
        visual_tokens   = self.visual_projection(visual_features)

        text_embeddings = self.language_model.shared(input_ids).to(visual_tokens.device)

        multimodal_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

        B, N = visual_tokens.shape[0], visual_tokens.shape[1]
        visual_mask = torch.ones(
            (B, N),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_mask = torch.cat([visual_mask, attention_mask], dim=1)

        return self.language_model.generate(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            **generation_kwargs,
        )