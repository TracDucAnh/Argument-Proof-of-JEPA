"""
tjepa_architecture.py
─────────────────────────────────────────────────────────────────────────────
Standalone Text-JEPA architecture.

Design faithful to I-JEPA (arXiv 2301.08243) adapted for text:
  - Context encoder : BERT-base OR BERT-large, sees span-masked sentence
  - Target encoder  : EMA copy of context encoder — no gradients
  - Predictor       : narrow BERT  D→d (bottleneck)→D
  - Loss            : token-level L2 on span positions (not mean-pooled first)

Compatible with tjepa_dataloader.py — forward() accepts the 7-key batch dict:
    masked_input_ids / masked_attention_mask / masked_token_type_ids
    clean_input_ids  / clean_attention_mask  / clean_token_type_ids
    span_mask   [B, L]  binary, 1 at span token positions

Changes vs original
───────────────────
* Added build_bert_large_config()  (hidden_size=1024, 24 layers, 16 heads)
* Added _ENCODER_CONFIGS registry  {"bert_base": ..., "bert_large": ...}
* TextJEPA now accepts model_name  (default "bert_base")
* Removed over-strict predictor_dim == hidden_dim // 2 constraint
  → replaced with a soft warning; any valid predictor_dim is accepted
* predictor_layers < encoder_layers check now uses the actual encoder depth
"""

from __future__ import annotations

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Encoder configs
# ══════════════════════════════════════════════════════════════════════════════

def build_bert_base_config(max_length: int = 256) -> BertConfig:
    """Standard BERT-base hyper-parameters (hidden_size=768, 12 layers)."""
    return BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,      # 4 × D
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
    )


def build_bert_large_config(max_length: int = 256) -> BertConfig:
    """Standard BERT-large hyper-parameters (hidden_size=1024, 24 layers)."""
    return BertConfig(
        vocab_size=30522,
        hidden_size=1024,            # D_large
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,      # 4 × D
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
    )


# Registry: model_name → config builder
_ENCODER_CONFIGS: dict[str, callable] = {
    "bert_base":  build_bert_base_config,
    "bert_large": build_bert_large_config,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Predictor
# ══════════════════════════════════════════════════════════════════════════════

class SmallBertPredictor(nn.Module):
    """
    Narrow BERT predictor that mirrors the I-JEPA predictor design.

    Data flow
    ─────────
    [B, L, D]
      → input_proj   Linear(D → d)        project to bottleneck
      → BERT layers  (hidden_size = d)    contextualise
      → output_proj  Linear(d → D)        project back to encoder space
    [B, L, D]

    The D→d→D bottleneck forces the predictor to compress context
    information, preventing it from trivially copying encoder output.
    Input and output both live in encoder space D so the loss is computed
    directly against the (unprojected) target encoder output.
    """

    def __init__(
        self,
        input_dim: int = 768,       # D — encoder hidden size
        predictor_dim: int = 384,   # d — internal bottleneck
        num_heads: int = 6,
        num_layers: int = 4,
        ffn_dim: int = 1536,
        max_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # ── projection in / out ───────────────────────────────────────────────
        self.input_proj  = nn.Linear(input_dim, predictor_dim)   # D → d
        self.output_proj = nn.Linear(predictor_dim, input_dim)   # d → D

        # ── narrow BERT ───────────────────────────────────────────────────────
        predictor_config = BertConfig(
            vocab_size=1,                          # no token embeddings used
            hidden_size=predictor_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ffn_dim,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_length,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
        )
        self.bert = BertModel(predictor_config, add_pooling_layer=False)

    def forward(
        self,
        hidden: torch.Tensor,           # [B, L, D]
        attention_mask: torch.Tensor,   # [B, L]
        token_type_ids: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:                  # [B, L, D]
        x = self.input_proj(hidden)     # [B, L, d]
        x = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state             # [B, L, d]
        return self.output_proj(x)      # [B, L, D]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TextJEPA  (main model)
# ══════════════════════════════════════════════════════════════════════════════

class TextJEPA(nn.Module):
    """
    Text JEPA: context encoder + target encoder (EMA) + predictor.

    Parameters
    ──────────
    model_name       : str  — "bert_base" (768-d, 12L) | "bert_large" (1024-d, 24L)
    hidden_dim       : int  — D, encoder hidden size  (must match model_name)
    predictor_dim    : int  — d, predictor bottleneck (any value < D)
    predictor_layers : int  — predictor transformer depth (< encoder num_layers)
    predictor_heads  : int  — predictor attention heads  (predictor_dim % heads == 0)
    predictor_ffn_dim: int  — predictor FFN hidden dim
    max_length       : int  — sequence length, must match dataloader (default 256)
    """

    def __init__(
        self,
        model_name: str = "bert_base",
        hidden_dim: int = 768,
        predictor_dim: int = 384,
        predictor_layers: int = 4,
        predictor_heads: int = 6,
        predictor_ffn_dim: int = 1536,
        max_length: int = 256,
    ):
        super().__init__()

        # ── resolve encoder config ────────────────────────────────────────────
        if model_name not in _ENCODER_CONFIGS:
            raise ValueError(
                f"model_name '{model_name}' not recognised. "
                f"Choose from: {list(_ENCODER_CONFIGS.keys())}"
            )
        encoder_config = _ENCODER_CONFIGS[model_name](max_length=max_length)

        # ── sanity checks ─────────────────────────────────────────────────────
        if hidden_dim != encoder_config.hidden_size:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal encoder hidden_size "
                f"({encoder_config.hidden_size}) for model_name='{model_name}'.\n"
                f"  bert_base  → hidden_dim=768\n"
                f"  bert_large → hidden_dim=1024"
            )

        # Soft warning instead of hard error: predictor_dim == D/2 is a good
        # default but not a strict architectural requirement.
        recommended = hidden_dim // 2
        if predictor_dim != recommended:
            warnings.warn(
                f"predictor_dim={predictor_dim} differs from the recommended "
                f"D/2={recommended}. This is allowed but may affect training dynamics.",
                UserWarning,
                stacklevel=2,
            )

        if predictor_layers >= encoder_config.num_hidden_layers:
            raise ValueError(
                f"predictor_layers ({predictor_layers}) must be fewer than "
                f"encoder layers ({encoder_config.num_hidden_layers}) for "
                f"model_name='{model_name}'."
            )

        # ── encoders ─────────────────────────────────────────────────────────
        self.context_encoder = BertModel(encoder_config, add_pooling_layer=False)
        self.target_encoder  = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        # ── predictor ────────────────────────────────────────────────────────
        self.predictor = SmallBertPredictor(
            input_dim=hidden_dim,
            predictor_dim=predictor_dim,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
            ffn_dim=predictor_ffn_dim,
            max_length=max_length,
        )

    # ── target encoder management ────────────────────────────────────────────

    def _freeze_target_encoder(self) -> None:
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay: float = 0.996) -> None:
        """EMA update: target ← decay·target + (1−decay)·context."""
        for ctx, tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt.data.mul_(decay).add_(ctx.data, alpha=1.0 - decay)

    # ── encoder helper ───────────────────────────────────────────────────────

    def _encode(
        self,
        encoder: BertModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder, return last hidden state [B, L, D]."""
        return encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state

    # ── loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def span_jepa_loss(
        pred: torch.Tensor,       # [B, L, D]
        target: torch.Tensor,     # [B, L, D]
        span_mask: torch.Tensor,  # [B, L]  binary, 1 at span positions
    ) -> torch.Tensor:
        """
        Token-level L2 loss over span positions, averaged over span tokens.

        Mirrors I-JEPA paper eq:
            (1/M) * Σ_i Σ_{j ∈ B_i} ‖pred_j − target_j‖²
        """
        l2_per_token  = ((pred - target) ** 2).sum(dim=-1)   # [B, L]
        masked         = l2_per_token * span_mask.float()     # zero non-span
        n_span_tokens  = span_mask.float().sum().clamp(min=1.0)
        return masked.sum() / n_span_tokens

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Accepts the 7-key batch dict from tjepa_dataloader.JEPASpanMaskCollator.

        Keys consumed
        ─────────────
        masked_input_ids / masked_attention_mask / masked_token_type_ids
            → context encoder input  (spans replaced with [MASK])
        clean_input_ids  / clean_attention_mask  / clean_token_type_ids
            → target encoder input   (original sentence)
        span_mask  [B, L]
            → selects which token positions contribute to the loss

        Returns
        ───────
        dict with:
            predicted_hidden  [B, L, D]  — predictor output
            target_hidden     [B, L, D]  — target encoder output (no grad)
            span_loss         scalar     — training objective
        """
        # ── context encoder (backprop flows through here) ─────────────────────
        context_hidden = self._encode(
            self.context_encoder,
            batch["masked_input_ids"],
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )  # [B, L, D]

        # ── target encoder (no gradient, no projection) ───────────────────────
        with torch.no_grad():
            target_hidden = self._encode(
                self.target_encoder,
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["clean_token_type_ids"],
            )  # [B, L, D]

        # ── predictor: D → d (bottleneck) → D ────────────────────────────────
        predicted_hidden = self.predictor(
            context_hidden,
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )  # [B, L, D]

        # ── loss on span positions only ───────────────────────────────────────
        span_loss = self.span_jepa_loss(
            predicted_hidden,
            target_hidden.detach(),
            batch["span_mask"],
        )

        return {
            "predicted_hidden": predicted_hidden,   # [B, L, D]
            "target_hidden":    target_hidden,       # [B, L, D]
            "span_loss":        span_loss,           # scalar
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Run one forward pass for both bert_base and bert_large."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    VOCAB_SIZE = 30522

    for model_name, D in [("bert_base", 768), ("bert_large", 1024)]:
        B, L = 2, 128

        print("=" * 60)
        print(f"  Smoke test: {model_name}  (D={D})")
        print("=" * 60)

        predictor_dim = 384          # intentionally not D//2 for large
        model = TextJEPA(
            model_name=model_name,
            hidden_dim=D,
            predictor_dim=predictor_dim,
            predictor_layers=4,
            predictor_heads=16 if model_name == "bert_large" else 6,
            predictor_ffn_dim=predictor_dim * 4,
            max_length=L,
        )
        model.eval()

        ctx_params  = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
        pred_params = sum(p.numel() for p in model.predictor.parameters())       / 1e6
        print(f"  context_encoder : {ctx_params:.1f} M")
        print(f"  predictor       : {pred_params:.1f} M")
        print(f"  trainable total : {ctx_params + pred_params:.1f} M")

        clean_ids  = torch.randint(1000, VOCAB_SIZE, (B, L))
        attn_mask  = torch.ones(B, L, dtype=torch.long)
        ttype_ids  = torch.zeros(B, L, dtype=torch.long)
        masked_ids = clean_ids.clone()
        span_mask  = torch.zeros(B, L, dtype=torch.long)
        for i in range(B):
            masked_ids[i, 10:15] = 103
            span_mask[i, 10:15]  = 1

        batch = dict(
            masked_input_ids=masked_ids, masked_attention_mask=attn_mask,
            masked_token_type_ids=ttype_ids, clean_input_ids=clean_ids,
            clean_attention_mask=attn_mask, clean_token_type_ids=ttype_ids,
            span_mask=span_mask,
        )

        with torch.no_grad():
            out = model(batch)

        assert out["predicted_hidden"].shape == (B, L, D)
        assert out["target_hidden"].shape    == (B, L, D)
        assert out["span_loss"].ndim         == 0
        print(f"  span_loss = {out['span_loss'].item():.6f}  ✓")
        print(f"  All assertions passed  ✓\n")


if __name__ == "__main__":
    _smoke_test()