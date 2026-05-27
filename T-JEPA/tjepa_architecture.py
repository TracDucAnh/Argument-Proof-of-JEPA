# tjepa_architecture.py  (fair-comparison edition)
# ─────────────────────────────────────────────────────────────────────────────
# Standalone Text-JEPA architecture.
# Changes vs. original:
#   • compute_effective_rank() returns normalized_rank + participation_ratio
#     using FULL SEQUENCE embeddings (not just span tokens) for fair comparison
#   • Added encode_full_sequence() helper — mirrors I-JEPA's forward_all_patches()
#   • Both bert_base and bert_large supported via _ENCODER_CONFIGS registry
#   • Soft warning on predictor_dim != D//2  (no hard error)
# ─────────────────────────────────────────────────────────────────────────────

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
    return BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length, type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


def build_bert_large_config(max_length: int = 256) -> BertConfig:
    return BertConfig(
        vocab_size=30522, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length, type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


_ENCODER_CONFIGS: dict[str, callable] = {
    "bert_base":  build_bert_base_config,
    "bert_large": build_bert_large_config,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Effective rank  — FAIR COMPARISON VERSION  (mirrors ijepa_architecture.py)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_effective_rank(z: torch.Tensor, embed_dim: int | None = None) -> dict:
    """
    Compute effective rank and normalized effective rank.

    Parameters
    ----------
    z         : Tensor [N, D]  — token embeddings (full sequence recommended)
    embed_dim : int or None    — if provided, also computes rank / embed_dim

    Returns
    -------
    dict with keys:
        effective_rank        : float  — exp(Shannon entropy of eigenvalue spectrum)
        normalized_rank       : float  — effective_rank / embed_dim  (0–1)
        participation_ratio   : float  — (Σλ)² / Σλ²  (alternative, robust measure)
    """
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)

    try:
        eigvals = torch.linalg.eigvalsh(cov)
    except Exception:
        nan = float("nan")
        return dict(effective_rank=nan, normalized_rank=nan, participation_ratio=nan)

    eigvals = eigvals.clamp(min=0.0)
    total   = eigvals.sum()

    if total < 1e-12:
        D = embed_dim or z.shape[1]
        return dict(effective_rank=1.0, normalized_rank=1.0 / D, participation_ratio=1.0)

    p = eigvals / total

    # Effective rank via entropy
    mask     = p > 1e-12
    H        = -(p[mask] * torch.log(p[mask])).sum()
    eff_rank = float(torch.exp(H).item())

    # Participation ratio
    pr = float((total ** 2 / (eigvals ** 2).sum()).item())

    D = embed_dim or z.shape[1]
    return dict(
        effective_rank      = eff_rank,
        normalized_rank     = eff_rank / D,
        participation_ratio = pr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Predictor
# ══════════════════════════════════════════════════════════════════════════════

class SmallBertPredictor(nn.Module):
    """
    Narrow BERT predictor: D → d (bottleneck) → D.
    Input/output live in encoder space D; loss computed against target encoder.
    """

    def __init__(self, input_dim=768, predictor_dim=384, num_heads=6,
                 num_layers=4, ffn_dim=1536, max_length=256, dropout=0.1):
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by "
                f"num_heads ({num_heads}).")

        self.input_proj  = nn.Linear(input_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, input_dim)

        predictor_config = BertConfig(
            vocab_size=1, hidden_size=predictor_dim, num_hidden_layers=num_layers,
            num_attention_heads=num_heads, intermediate_size=ffn_dim,
            hidden_act="gelu", hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_length, type_vocab_size=2,
            initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
            position_embedding_type="absolute",
        )
        self.bert = BertModel(predictor_config, add_pooling_layer=False)

    def forward(self, hidden, attention_mask, token_type_ids):
        x = self.input_proj(hidden)
        x = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TextJEPA  (main model)
# ══════════════════════════════════════════════════════════════════════════════

class TextJEPA(nn.Module):
    """
    Text JEPA: context encoder + EMA target encoder + predictor.

    Parameters
    ──────────
    model_name       : "bert_base" (768-d, 12L) | "bert_large" (1024-d, 24L)
    hidden_dim       : must match model_name (768 or 1024)
    predictor_dim    : bottleneck dim d < D
    predictor_layers : transformer depth (< encoder num_layers)
    predictor_heads  : attention heads (predictor_dim % heads == 0)
    predictor_ffn_dim: FFN hidden dim
    max_length       : sequence length, must match dataloader
    """

    def __init__(self, model_name="bert_base", hidden_dim=768, predictor_dim=384,
                 predictor_layers=4, predictor_heads=6, predictor_ffn_dim=1536,
                 max_length=256):
        super().__init__()

        if model_name not in _ENCODER_CONFIGS:
            raise ValueError(f"model_name '{model_name}' not recognised. "
                             f"Choose from: {list(_ENCODER_CONFIGS.keys())}")
        encoder_config = _ENCODER_CONFIGS[model_name](max_length=max_length)

        if hidden_dim != encoder_config.hidden_size:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal encoder hidden_size "
                f"({encoder_config.hidden_size}) for model_name='{model_name}'.")

        recommended = hidden_dim // 2
        if predictor_dim != recommended:
            warnings.warn(
                f"predictor_dim={predictor_dim} differs from recommended D/2={recommended}. "
                "This is allowed but may affect training dynamics.", UserWarning, stacklevel=2)

        if predictor_layers >= encoder_config.num_hidden_layers:
            raise ValueError(
                f"predictor_layers ({predictor_layers}) must be fewer than "
                f"encoder layers ({encoder_config.num_hidden_layers}).")

        self.hidden_dim      = hidden_dim
        self.context_encoder = BertModel(encoder_config, add_pooling_layer=False)
        self.target_encoder  = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.predictor = SmallBertPredictor(
            input_dim=hidden_dim, predictor_dim=predictor_dim,
            num_heads=predictor_heads, num_layers=predictor_layers,
            ffn_dim=predictor_ffn_dim, max_length=max_length,
        )

    # ── target encoder management ─────────────────────────────────────────────

    def _freeze_target_encoder(self):
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay=0.996):
        """EMA update: target ← decay·target + (1−decay)·context."""
        for ctx, tgt in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            tgt.data.mul_(decay).add_(ctx.data, alpha=1.0 - decay)

    # ── encoder helpers ───────────────────────────────────────────────────────

    def _encode(self, encoder, input_ids, attention_mask, token_type_ids):
        """Run encoder, return last_hidden_state [B, L, D]."""
        return encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, return_dict=True,
        ).last_hidden_state

    def encode_full_sequence(self, batch: dict, use_target: bool = False) -> torch.Tensor:
        """
        Encode the CLEAN (unmasked) sentence through context or target encoder.
        Returns all token embeddings [B, L, D] — mirrors I-JEPA's forward_all_patches().
        Used by training script for fair effective-rank computation.

        Parameters
        ----------
        batch      : 7-key batch dict from tjepa_dataloader
        use_target : if True, use target encoder (no grad); else context encoder
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        return self._encode(
            encoder,
            batch["clean_input_ids"],
            batch["clean_attention_mask"],
            batch["clean_token_type_ids"],
        )

    # ── loss ──────────────────────────────────────────────────────────────────

    @staticmethod
    def span_jepa_loss(pred, target, span_mask):
        """
        Token-level L2 loss over span positions only, averaged over span tokens.
        Mirrors I-JEPA eq: (1/M) Σ_i Σ_{j∈B_i} ‖pred_j − target_j‖²
        """
        l2_per_token = ((pred - target) ** 2).sum(dim=-1)
        masked        = l2_per_token * span_mask.float()
        n_span_tokens = span_mask.float().sum().clamp(min=1.0)
        return masked.sum() / n_span_tokens

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Accepts 7-key batch dict from tjepa_dataloader.JEPASpanMaskCollator.

        Returns
        ───────
        dict:
            predicted_hidden  [B, L, D]
            target_hidden     [B, L, D]
            span_loss         scalar
        """
        context_hidden = self._encode(
            self.context_encoder,
            batch["masked_input_ids"],
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )

        with torch.no_grad():
            target_hidden = self._encode(
                self.target_encoder,
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["clean_token_type_ids"],
            )

        predicted_hidden = self.predictor(
            context_hidden,
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )

        span_loss = self.span_jepa_loss(
            predicted_hidden, target_hidden.detach(), batch["span_mask"])

        return dict(
            predicted_hidden = predicted_hidden,
            target_hidden    = target_hidden,
            span_loss        = span_loss,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    VOCAB_SIZE = 30522

    for model_name, D in [("bert_base", 768), ("bert_large", 1024)]:
        B, L = 2, 128
        print("=" * 60)
        print(f"  Smoke test: {model_name}  (D={D})")
        print("=" * 60)

        model = TextJEPA(
            model_name=model_name, hidden_dim=D, predictor_dim=384,
            predictor_layers=4,
            predictor_heads=16 if model_name == "bert_large" else 6,
            predictor_ffn_dim=384 * 4, max_length=L,
        )
        model.eval()

        ctx_p  = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
        pred_p = sum(p.numel() for p in model.predictor.parameters())       / 1e6
        print(f"  context_encoder : {ctx_p:.1f} M")
        print(f"  predictor       : {pred_p:.1f} M")

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
            z_full = model.encode_full_sequence(batch)        # [B, L, D]
            z_flat = z_full.reshape(-1, D)                    # [B*L, D]
            rank_info = compute_effective_rank(z_flat, embed_dim=D)

        assert out["predicted_hidden"].shape == (B, L, D)
        assert out["span_loss"].ndim == 0
        assert z_full.shape == (B, L, D)

        print(f"  span_loss          = {out['span_loss'].item():.6f}  ✓")
        print(f"  effective_rank     = {rank_info['effective_rank']:.2f}")
        print(f"  normalized_rank    = {rank_info['normalized_rank']:.4f}  (0–1)")
        print(f"  participation_ratio= {rank_info['participation_ratio']:.2f}")
        print(f"  All assertions passed  ✓\n")


if __name__ == "__main__":
    _smoke_test()