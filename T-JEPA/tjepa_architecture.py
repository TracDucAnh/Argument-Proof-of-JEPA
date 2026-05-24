"""
tjepa_architecture.py
─────────────────────────────────────────────────────────────────────────────
Standalone Text-JEPA architecture.

Design faithful to I-JEPA (arXiv 2301.08243) adapted for text:
  - Context encoder : BERT-base, sees span-masked sentence
  - Target encoder  : BERT-base (EMA copy), sees clean sentence — no gradients
  - Predictor       : narrow BERT  D→d (bottleneck)→D
  - Loss            : token-level L2 on span positions (not mean-pooled first)

Compatible with tjepa_dataloader.py — forward() accepts the 7-key batch dict:
    masked_input_ids / masked_attention_mask / masked_token_type_ids
    clean_input_ids  / clean_attention_mask  / clean_token_type_ids
    span_mask   [B, L]  binary, 1 at span token positions
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BERT-base config  (inlined — no dependency on pretrain/common.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_bert_base_config(max_length: int = 256) -> BertConfig:
    """
    Standard BERT-base hyper-parameters.
    max_position_embeddings is set to max_length so the encoder
    matches whatever sequence length the dataloader uses.
    """
    return BertConfig(
        vocab_size=30522,            # bert-base-uncased vocab
        hidden_size=768,             # D
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

    Default settings (matching tjepa_training.py defaults)
    ───────────────────────────────────────────────────────
    input_dim      = 768   (D, must equal encoder hidden_size)
    predictor_dim  = 384   (d = D/2)
    num_layers     = 4
    num_heads      = 6
    ffn_dim        = 1536  (4 × d)
    max_length     = 256
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
    hidden_dim       : int  — D, encoder hidden size          (default 768)
    predictor_dim    : int  — d, predictor bottleneck         (default 384 = D/2)
    predictor_layers : int  — predictor transformer depth     (default 4)
    predictor_heads  : int  — predictor attention heads       (default 6)
    predictor_ffn_dim: int  — predictor FFN hidden dim        (default 1536 = 4d)
    max_length       : int  — sequence length, must match dataloader (default 256)

    Constraints (enforced in __init__)
    ───────────────────────────────────
    hidden_dim       == encoder hidden_size   (768 for BERT-base)
    predictor_dim    == hidden_dim // 2       (384)
    predictor_layers <  encoder num_layers    (< 12)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        predictor_dim: int = 384,
        predictor_layers: int = 4,
        predictor_heads: int = 6,
        predictor_ffn_dim: int = 1536,
        max_length: int = 256,
    ):
        super().__init__()

        encoder_config = build_bert_base_config(max_length=max_length)

        # sanity checks
        if hidden_dim != encoder_config.hidden_size:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal encoder hidden_size "
                f"({encoder_config.hidden_size})."
            )
        if predictor_dim != hidden_dim // 2:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be D/2 = {hidden_dim // 2}."
            )
        if predictor_layers >= encoder_config.num_hidden_layers:
            raise ValueError(
                f"predictor_layers ({predictor_layers}) must be fewer than "
                f"encoder layers ({encoder_config.num_hidden_layers})."
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

        where M·|B_i| ≈ total number of span tokens (sum of span_mask).

        Tokens outside spans are zeroed out — the model is only penalised
        for predicting the masked (target) positions.
        We do NOT mean-pool tokens before computing loss; that would destroy
        per-token signal and deviate from the original formulation.
        """
        l2_per_token = ((pred - target) ** 2).sum(dim=-1)   # [B, L]
        masked       = l2_per_token * span_mask.float()      # zero non-span
        n_span_tokens = span_mask.float().sum().clamp(min=1.0)
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
    """
    Instantiate TextJEPA with default settings and run one synthetic
    forward pass that mimics a batch from tjepa_dataloader.py.

    Batch keys and shapes match JEPASpanMaskCollator output exactly.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B          = 4      # batch size
    L          = 256    # max_length
    VOCAB_SIZE = 30522  # bert-base-uncased
    D          = 768
    EMA_DECAY  = 0.996

    print("=" * 60)
    print("  T-JEPA Architecture — Smoke Test")
    print("=" * 60)

    # ── instantiate ──────────────────────────────────────────────────────────
    model = TextJEPA(
        hidden_dim=768,
        predictor_dim=384,
        predictor_layers=4,
        predictor_heads=6,
        predictor_ffn_dim=1536,
        max_length=L,
    )
    model.eval()

    ctx_params  = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
    tgt_params  = sum(p.numel() for p in model.target_encoder.parameters())  / 1e6
    pred_params = sum(p.numel() for p in model.predictor.parameters())       / 1e6
    total       = ctx_params + pred_params   # target shares arch, not trained
    print(f"  context_encoder params : {ctx_params:.1f} M")
    print(f"  target_encoder  params : {tgt_params:.1f} M  (EMA, no grad)")
    print(f"  predictor       params : {pred_params:.1f} M")
    print(f"  trainable total        : {total:.1f} M  (context_encoder + predictor)")

    # ── synthetic batch mimicking JEPASpanMaskCollator output ────────────────
    clean_ids   = torch.randint(1000, VOCAB_SIZE, (B, L))
    attn_mask   = torch.ones(B, L, dtype=torch.long)
    ttype_ids   = torch.zeros(B, L, dtype=torch.long)

    # mask 5 spans × 5 tokens each (same as smoke test defaults)
    masked_ids  = clean_ids.clone()
    span_mask   = torch.zeros(B, L, dtype=torch.long)
    for i in range(B):
        for s in range(5):
            start = 10 + s * 12
            masked_ids[i, start:start + 5] = 103   # [MASK]
            span_mask[i, start:start + 5]  = 1

    batch = {
        "masked_input_ids":       masked_ids,
        "masked_attention_mask":  attn_mask,
        "masked_token_type_ids":  ttype_ids,
        "clean_input_ids":        clean_ids,
        "clean_attention_mask":   attn_mask,
        "clean_token_type_ids":   ttype_ids,
        "span_mask":              span_mask,
    }

    print(f"\n  Input shapes:")
    print(f"    masked_input_ids      : {tuple(masked_ids.shape)}")
    print(f"    clean_input_ids       : {tuple(clean_ids.shape)}")
    print(f"    span_mask             : {tuple(span_mask.shape)}  "
          f"(tokens masked per sample: {span_mask.sum(dim=1).tolist()})")

    # ── forward ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model(batch)

    print(f"\n  Output shapes:")
    print(f"    predicted_hidden : {tuple(out['predicted_hidden'].shape)}  ✓")
    print(f"    target_hidden    : {tuple(out['target_hidden'].shape)}  ✓")
    print(f"    span_loss        : {out['span_loss'].item():.6f}  ✓")

    assert out["predicted_hidden"].shape == (B, L, D)
    assert out["target_hidden"].shape    == (B, L, D)
    assert out["span_loss"].ndim         == 0

    # ── EMA update check ─────────────────────────────────────────────────────
    # record one target param before update
    p_before = next(model.target_encoder.parameters()).data.clone()
    model.update_target_encoder(decay=EMA_DECAY)
    p_after  = next(model.target_encoder.parameters()).data
    assert not torch.equal(p_before, p_after), "EMA update had no effect!"
    print(f"\n  EMA update (decay={EMA_DECAY}) : target params changed  ✓")

    # ── gradient check ───────────────────────────────────────────────────────
    for name, param in model.target_encoder.named_parameters():
        assert not param.requires_grad, f"target_encoder.{name} has grad!"
    print(f"  target_encoder frozen (no grad)  ✓")

    print("\n  All assertions passed — architecture is compatible with")
    print("  tjepa_dataloader.py  ✓")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()