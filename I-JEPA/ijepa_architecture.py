# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Standalone version — inlines trunc_normal_, repeat_interleave_batch,
# and apply_masks so the file works without cloning the full Meta I-JEPA repo.
# Default hyper-parameters match the ViT-B/16 config from the I-JEPA paper.
# Compatible with ijepa_dataloader.py (masks shape: [B, n_masks, keep]).

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Inlined utilities  (from src/utils/tensors.py  &  src/masks/utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated-normal fill (in-place, no gradient)."""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fill *tensor* with values drawn from a truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    """
    Repeat each sample in x `repeat` times, interleaved.

    Example: x has shape [B*n, ...] and we want [B*n*repeat, ...]
    where each of the B groups of n tokens is repeated `repeat` times.
    """
    N = len(x) // B
    return torch.cat([
        torch.cat([x[i * N:(i + 1) * N] for _ in range(repeat)], dim=0)
        for i in range(B)
    ], dim=0)


def apply_masks(x, masks):
    """
    Select patch tokens according to mask indices.

    Parameters
    ----------
    x     : Tensor  [B, N, D]  — full sequence of patch embeddings
    masks : list of LongTensor, each [B, keep]  (or a single tensor)

    Returns
    -------
    Tensor  [B*len(masks), keep, D]
    """
    if not isinstance(masks, (list, tuple)):
        masks = [masks]

    all_x = []
    for m in masks:
        # m: [B, keep]
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, keep, D]
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Positional embeddings
# ══════════════════════════════════════════════════════════════════════════════

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size : int — height == width of the patch grid
    returns   : ndarray [grid_size*grid_size, embed_dim]
                or [1+grid_size*grid_size, embed_dim] when cls_token=True
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid   = np.meshgrid(grid_w, grid_h)   # w first
    grid   = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)                       # (H*W, D)


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid      = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega  = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)          # (D/2,)

    pos = pos.reshape(-1)                    # (M,)
    out = np.einsum("m,d->md", pos, omega)   # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)   # (M, D)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob     = 1 - drop_prob
    shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic Depth (drop path) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = MLP(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Patch embedding
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Image → flat sequence of patch embeddings via a strided Conv2d."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches      = (img_size // patch_size) ** 2
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = num_patches
        self.proj        = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


class ConvEmbed(nn.Module):
    """3×3 convolution stems for ViT-C (ViTC) style models."""

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        stem     = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3,
                               stride=strides[i], padding=1,
                               bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i + 1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1],
                           kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        stride_prod      = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod) ** 2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Predictor
# ══════════════════════════════════════════════════════════════════════════════

class VisionTransformerPredictor(nn.Module):
    """
    Narrow transformer that maps context tokens → predicted target tokens.

    Input  : context embeddings  x       [B, N_ctx, encoder_dim]
             context mask indices masks_x [list of LongTensor [B, keep_ctx]]
             target  mask indices masks   [list of LongTensor [B, keep_tgt]]
    Output : predicted embeddings         [B*len(masks), keep_tgt, encoder_dim]
    """

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # fixed sincos positional embedding (no cls token)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            predictor_embed_dim, int(num_patches ** 0.5), cls_token=False)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path_rate=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    # ── init ─────────────────────────────────────────────────────────────────

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data,   layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, masks_x, masks):
        """
        x       : [B, N_ctx, encoder_dim]  — encoder output (context tokens only)
        masks_x : LongTensor [B, keep_ctx] or list thereof
        masks   : LongTensor [B, keep_tgt] or list thereof
        """
        assert masks is not None and masks_x is not None, \
            "Cannot run predictor without mask indices"

        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks   = [masks]

        B = len(x) // len(masks_x)

        # encoder dim → predictor dim
        x = self.predictor_embed(x)

        # add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x           = x + apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # build mask tokens for target positions
        pos_embs  = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs  = apply_masks(pos_embs, masks)
        pos_embs  = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.expand(pos_embs.shape[0], pos_embs.shape[1], -1).clone()
        pred_tokens = pred_tokens + pos_embs

        # concatenate context + mask tokens
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # transformer forward
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # return only predictions for the target (masked) positions
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Encoder
# ══════════════════════════════════════════════════════════════════════════════

class VisionTransformer(nn.Module):
    """
    Standard Vision Transformer encoder (no CLS token, sincos pos-embed).

    When masks is provided the forward pass returns only the unmasked
    (context) patch tokens — matching I-JEPA's encoder behaviour.
    """

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads    = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # fixed sincos positional embedding (no cls token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches ** 0.5), cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path_rate=dpr[i],
                  norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    # ── init ─────────────────────────────────────────────────────────────────

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data,   layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ── positional encoding interpolation ────────────────────────────────────

    def interpolate_pos_encoding(self, x, pos_embed):
        """
        Bilinear interpolation of sincos pos embed when the number of
        patches at runtime differs from the one used at construction.

        NOTE: this encoder has NO cls token, so indices are not shifted.
        """
        npatch = x.shape[1]          # actual number of patches
        N      = pos_embed.shape[1]  # number of patches the embed was built for

        if npatch == N:
            return pos_embed

        dim       = x.shape[-1]
        grid_orig = int(N      ** 0.5)
        grid_new  = int(npatch ** 0.5)

        pos_embed = F.interpolate(
            pos_embed.reshape(1, grid_orig, grid_orig, dim).permute(0, 3, 1, 2),
            size=(grid_new, grid_new),
            mode="bicubic",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos_embed

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, masks=None):
        """
        x     : FloatTensor [B, C, H, W]
        masks : LongTensor [B, keep] or list thereof, or None
                When provided, only the indexed patch tokens are returned.

        Returns
        -------
        Tensor [B, N, D]   (N = keep when masks given, num_patches otherwise)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # patchify
        x = self.patch_embed(x)           # [B, N, D]

        # add positional embedding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # apply context mask (keep only unmasked tokens)
        if masks is not None:
            x = apply_masks(x, masks)

        # transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Factory functions  (default settings from I-JEPA paper)
# ══════════════════════════════════════════════════════════════════════════════

def vit_predictor(**kwargs):
    """Predictor head — depth=6, predictor_embed_dim=384 (paper default)."""
    return VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_base(patch_size=16, **kwargs):
    """ViT-B/16 — primary backbone in the I-JEPA paper."""
    return VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_huge(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_giant(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16,
        mlp_ratio=int(48 / 11), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


VIT_EMBED_DIMS = {
    "vit_tiny":  192,
    "vit_small": 384,
    "vit_base":  768,
    "vit_large": 1024,
    "vit_huge":  1280,
    "vit_giant": 1408,
}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Smoke test — verifies compatibility with ijepa_dataloader.py
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """
    Instantiate encoder + predictor with ViT-B/16 defaults and run one
    synthetic forward pass that mimics what ijepa_dataloader returns.

    Mask shapes expected from MaskCollator (default settings):
        masks_enc  : [B, nenc=1,  keep_enc]   — context (encoder) patches
        masks_pred : [B, npred=4, keep_pred]  — target  (predictor) patches
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    BATCH      = 4
    IMG_SIZE   = 224
    PATCH_SIZE = 16
    N_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
    ENC_DIM    = 768    # ViT-B
    PRED_DIM   = 384

    print("=" * 60)
    print("  I-JEPA Architecture — Smoke Test (ViT-B/16)")
    print("=" * 60)

    # ── instantiate ──────────────────────────────────────────────────────
    encoder = vit_base(patch_size=PATCH_SIZE, img_size=[IMG_SIZE])
    predictor = vit_predictor(
        num_patches=N_PATCHES,
        embed_dim=ENC_DIM,
        predictor_embed_dim=PRED_DIM,
        depth=6,
        num_heads=12,
    )
    encoder.eval()
    predictor.eval()

    enc_params  = sum(p.numel() for p in encoder.parameters())  / 1e6
    pred_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"  encoder   params : {enc_params:.1f} M")
    print(f"  predictor params : {pred_params:.1f} M")

    # ── synthetic batch mimicking MaskCollator output ────────────────────
    imgs = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)

    # context mask: nenc=1, keep=120 patches
    # target  mask: npred=4, keep=25 patches each
    keep_enc  = 120
    keep_pred = 25
    masks_enc  = torch.stack([
        torch.randperm(N_PATCHES)[:keep_enc] for _ in range(BATCH)
    ]).unsqueeze(1)   # [B, 1, keep_enc]
    masks_pred = torch.stack([
        torch.stack([torch.randperm(N_PATCHES)[:keep_pred] for _ in range(4)])
        for _ in range(BATCH)
    ])                # [B, 4, keep_pred]

    print(f"\n  imgs       : {tuple(imgs.shape)}")
    print(f"  masks_enc  : {tuple(masks_enc.shape)}  (B, nenc, keep_enc)")
    print(f"  masks_pred : {tuple(masks_pred.shape)}  (B, npred, keep_pred)")

    # ── encoder forward (context patches only) ───────────────────────────
    # masks_enc[:, 0, :] → [B, keep_enc]
    with torch.no_grad():
        h = encoder(imgs, masks=masks_enc[:, 0, :])   # [B, keep_enc, ENC_DIM]
    print(f"\n  encoder output   : {tuple(h.shape)}  ✓")
    assert h.shape == (BATCH, keep_enc, ENC_DIM)

    # ── predictor forward ─────────────────────────────────────────────────
    # pass each target mask separately (list of [B, keep_pred])
    masks_pred_list = [masks_pred[:, k, :] for k in range(masks_pred.shape[1])]
    with torch.no_grad():
        preds = predictor(h, masks_x=masks_enc[:, 0, :], masks=masks_pred_list)
    # expected: [B*npred, keep_pred, ENC_DIM]
    print(f"  predictor output : {tuple(preds.shape)}  ✓")
    assert preds.shape == (BATCH * 4, keep_pred, ENC_DIM)

    print("\n  All assertions passed — architecture is compatible with")
    print("  ijepa_dataloader.py  ✓")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()