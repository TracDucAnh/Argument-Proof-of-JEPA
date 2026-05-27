# ijepa_architecture.py  (fair-comparison edition)
# ─────────────────────────────────────────────────────────────────────────────
# Standalone I-JEPA Vision Transformer architecture.
# Changes vs. original:
#   • compute_effective_rank() now also returns normalized rank (rank/embed_dim)
#   • Added get_all_patch_embeddings() helper for fair rank computation
#     (uses ALL patches, not just context — mirrors T-JEPA's full-sequence approach)
#   • Factory VIT_EMBED_DIMS kept for cross-model lookups
# ─────────────────────────────────────────────────────────────────────────────

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Inlined utilities
# ══════════════════════════════════════════════════════════════════════════════

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
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
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat([
        torch.cat([x[i * N:(i + 1) * N] for _ in range(repeat)], dim=0)
        for i in range(B)
    ], dim=0)


def apply_masks(x, masks):
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Effective rank  — FAIR COMPARISON VERSION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_effective_rank(z: torch.Tensor, embed_dim: int | None = None):
    """
    Compute effective rank and normalized effective rank.

    Parameters
    ----------
    z         : Tensor [N, D]  — token embeddings (any subset or full sequence)
    embed_dim : int or None    — if provided, also returns rank / embed_dim

    Returns
    -------
    dict with keys:
        effective_rank        : float  — exp(Shannon entropy of eigenvalue spectrum)
        normalized_rank       : float  — effective_rank / embed_dim  (0–1)
        participation_ratio   : float  — (Σλ)² / Σλ²  (alternative measure)
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
    mask = p > 1e-12
    H    = -(p[mask] * torch.log(p[mask])).sum()
    eff_rank = float(torch.exp(H).item())

    # Participation ratio  (more robust to outlier eigenvalues)
    pr = float((total ** 2 / (eigvals ** 2).sum()).item())

    D = embed_dim or z.shape[1]
    return dict(
        effective_rank      = eff_rank,
        normalized_rank     = eff_rank / D,
        participation_ratio = pr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Positional embeddings
# ══════════════════════════════════════════════════════════════════════════════

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid   = np.meshgrid(grid_w, grid_h)
    grid   = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega  = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega  = 1.0 / (10000 ** omega)
    pos    = pos.reshape(-1)
    out    = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob     = 1 - drop_prob
    shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
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
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


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
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x)), attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Patch embedding
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj        = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Predictor
# ══════════════════════════════════════════════════════════════════════════════

class VisionTransformerPredictor(nn.Module):
    def __init__(self, num_patches, embed_dim=768, predictor_embed_dim=384,
                 depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02, **kwargs):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            predictor_embed_dim, int(num_patches ** 0.5), cls_token=False)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path_rate=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

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
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert masks is not None and masks_x is not None
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks   = [masks]

        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x           = x + apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, D = x.shape

        pos_embs    = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs    = apply_masks(pos_embs, masks)
        pos_embs    = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.expand(pos_embs.shape[0], pos_embs.shape[1], -1).clone()
        pred_tokens = pred_tokens + pos_embs

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Encoder
# ══════════════════════════════════════════════════════════════════════════════

class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads    = num_heads

        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path_rate=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

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
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N      = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        dim       = x.shape[-1]
        grid_orig = int(N      ** 0.5)
        grid_new  = int(npatch ** 0.5)
        pos_embed = F.interpolate(
            pos_embed.reshape(1, grid_orig, grid_orig, dim).permute(0, 3, 1, 2),
            size=(grid_new, grid_new), mode="bicubic", align_corners=False,
        )
        return pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

    def forward(self, x, masks=None):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        x         = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x         = x + pos_embed
        if masks is not None:
            x = apply_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward_all_patches(self, x):
        """
        Forward pass WITHOUT any masking — returns ALL patch embeddings.
        Used by training script for fair effective-rank computation.

        Returns: Tensor [B, num_patches, embed_dim]
        """
        x         = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x         = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Factory functions
# ══════════════════════════════════════════════════════════════════════════════

def vit_predictor(**kwargs):
    return VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=192, depth=12,
        num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12,
        num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_huge(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=1280, depth=32,
        num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_giant(patch_size=16, **kwargs):
    return VisionTransformer(patch_size=patch_size, embed_dim=1408, depth=40,
        num_heads=16, mlp_ratio=int(48 / 11), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


VIT_EMBED_DIMS = {
    "vit_tiny":  192,
    "vit_small": 384,
    "vit_base":  768,
    "vit_large": 1024,
    "vit_huge":  1280,
    "vit_giant": 1408,
}