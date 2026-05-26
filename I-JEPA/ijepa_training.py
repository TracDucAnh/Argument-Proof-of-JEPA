# ijepa_training.py
# Trains I-JEPA for 10 epochs on local ImageNet-1K subset.
# Logs JEPA loss + effective rank every 10 iters → ../Arg-I/I-JEPA.json
# Saves dual-axis plot (live update every 10 iters) → ../Arg-I/I-JEPA.png
#
# Usage (run from I-JEPA/ directory):
#   python ijepa_training.py
#
# Requires: ijepa_architecture.py, ijepa_dataloader.py, data/ populated
# ─────────────────────────────────────────────────────────────────────────────

import copy
import json
import math
import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../I-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = ARG_I_DIR / "I-JEPA.json"
PNG_PATH  = ARG_I_DIR / "I-JEPA.png"

# ── make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from ijepa_architecture import vit_base, vit_predictor
from ijepa_dataloader   import make_imagenet1k_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 256,
    num_workers     = 4,
    crop_size       = 224,
    # masking
    patch_size      = 16,
    enc_mask_scale  = (0.85, 1.0),
    pred_mask_scale = (0.15, 0.2),
    aspect_ratio    = (0.75, 1.5),
    num_enc_masks   = 1,
    num_pred_masks  = 4,
    min_keep        = 10,
    # model
    embed_dim       = 768,   # ViT-B
    pred_embed_dim  = 384,
    pred_depth      = 6,
    pred_heads      = 12,
    # optimiser
    lr              = 1e-3,
    weight_decay    = 0.05,
    # EMA
    ema_tau         = 0.996,
    # training
    epochs          = 10,
    log_every       = 10,    # iterations between records
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_effective_rank(z: torch.Tensor) -> float:
    """
    Effective rank of the representation covariance.

    effective_rank = exp( H( λ_i / Σλ_i ) )
    where λ_i are eigenvalues of the empirical covariance of z.

    Parameters
    ----------
    z : Tensor [N, D]   — batch of representation vectors
    """
    z = z.float()
    # centre
    z = z - z.mean(dim=0, keepdim=True)
    # covariance  [D, D]
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    # eigenvalues (symmetric, so use eigh for speed + stability)
    try:
        eigvals = torch.linalg.eigvalsh(cov)          # [D], ascending
    except Exception:
        return float("nan")
    eigvals = eigvals.clamp(min=0.0)
    total   = eigvals.sum()
    if total < 1e-12:
        return 1.0
    p    = eigvals / total                             # probability distribution
    # entropy H = -Σ p log p  (skip zeros)
    mask = p > 1e-12
    H    = -(p[mask] * torch.log(p[mask])).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    """In-place EMA: θ_target ← τ·θ_target + (1-τ)·θ_online"""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


def save_plot(records: list[dict]):
    """
    Dual-axis line chart: left = loss, right = effective rank.
    Được gọi sau mỗi log_every iteration để cập nhật PNG liên tục (live update).
    """
    steps  = [r["global_step"] for r in records]
    losses = [r["loss"]           for r in records]
    ranks  = [r["effective_rank"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = "#378ADD"
    color_rank = "#D85A30"

    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("MSE Loss", color=color_loss, fontsize=12)
    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="I-JEPA loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=color_rank, fontsize=12)
    ax2.plot(steps, ranks, color=color_rank, linewidth=1.8,
             linestyle="--", label="I-JEPA eff. rank")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(bottom=0)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    # Hiển thị step hiện tại trên tiêu đề
    current_step = steps[-1] if steps else 0
    plt.title(
        f"I-JEPA Training Dynamics (Loss & Effective Rank)  [step {current_step}]",
        fontsize=13,
    )
    fig.tight_layout()

    # Ghi vào file tạm rồi rename → tránh đọc file lúc đang ghi dở
    tmp_path = PNG_PATH.with_suffix(".tmp.png")
    fig.savefig(str(tmp_path), dpi=150)
    plt.close(fig)
    tmp_path.replace(PNG_PATH)

    log.info(f"[live] Plot updated → {PNG_PATH}  (step {current_step})")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device(CFG["device"])
    log.info(f"Device: {device}")

    # ── dataloader ────────────────────────────────────────────────────────
    loader, _ = make_imagenet1k_dataloader(
        data_dir        = CFG["data_dir"],
        split           = "train",
        batch_size      = CFG["batch_size"],
        num_workers     = CFG["num_workers"],
        pin_mem         = (device.type == "cuda"),
        crop_size       = CFG["crop_size"],
        patch_size      = CFG["patch_size"],
        enc_mask_scale  = CFG["enc_mask_scale"],
        pred_mask_scale = CFG["pred_mask_scale"],
        aspect_ratio    = CFG["aspect_ratio"],
        num_enc_masks   = CFG["num_enc_masks"],
        num_pred_masks  = CFG["num_pred_masks"],
        min_keep        = CFG["min_keep"],
        allow_overlap   = False,
        use_masking     = True,
        persistent_workers = False,
    )
    log.info(f"Dataset: {len(loader.dataset):,} images, "
             f"{len(loader):,} batches/epoch, bs={CFG['batch_size']}")

    N_PATCHES = (CFG["crop_size"] // CFG["patch_size"]) ** 2   # 196

    # ── models ────────────────────────────────────────────────────────────
    context_encoder = vit_base(
        patch_size = CFG["patch_size"],
        img_size   = [CFG["crop_size"]],
    ).to(device)

    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    predictor = vit_predictor(
        num_patches         = N_PATCHES,
        embed_dim           = CFG["embed_dim"],
        predictor_embed_dim = CFG["pred_embed_dim"],
        depth               = CFG["pred_depth"],
        num_heads           = CFG["pred_heads"],
    ).to(device)

    log.info(
        f"Params — encoder: "
        f"{sum(p.numel() for p in context_encoder.parameters())/1e6:.1f}M  "
        f"predictor: "
        f"{sum(p.numel() for p in predictor.parameters())/1e6:.1f}M"
    )

    # ── optimiser ─────────────────────────────────────────────────────────
    params = list(context_encoder.parameters()) + list(predictor.parameters())
    optimiser = torch.optim.AdamW(
        params, lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )

    total_steps = CFG["epochs"] * len(loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps, eta_min=1e-5
    )

    # ── training loop ─────────────────────────────────────────────────────
    records      = []
    global_step  = 0

    for epoch in range(1, CFG["epochs"] + 1):
        context_encoder.train()
        predictor.train()
        epoch_losses = []

        for it, (imgs, masks_enc, masks_pred) in enumerate(loader, start=1):
            imgs       = imgs.to(device, non_blocking=True)
            masks_enc  = masks_enc.to(device,  non_blocking=True)   # [B, 1, keep_enc]
            masks_pred = masks_pred.to(device, non_blocking=True)   # [B, 4, keep_pred]

            B = imgs.shape[0]

            # context mask: shape [B, keep_enc]
            masks_enc_flat  = masks_enc[:, 0, :]
            # target  masks: list of [B, keep_pred]
            masks_pred_list = [masks_pred[:, k, :] for k in range(masks_pred.shape[1])]

            # ── target representations (no grad) ──────────────────────────
            with torch.no_grad():
                z_target_full = target_encoder(imgs)   # [B, N, D] — all patches
                # gather target patches for each pred mask
                z_targets = []
                for m in masks_pred_list:
                    idx  = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                    z_targets.append(torch.gather(z_target_full, 1, idx))
                # [B*npred, keep_pred, D]
                z_target_cat = torch.cat(z_targets, dim=0)

            # ── context encoder ───────────────────────────────────────────
            z_ctx = context_encoder(imgs, masks=masks_enc_flat)   # [B, keep_enc, D]

            # ── predictor ─────────────────────────────────────────────────
            z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)
            # z_pred: [B*npred, keep_pred, D]

            # ── JEPA loss (MSE in latent space) ───────────────────────────
            loss = F.mse_loss(z_pred, z_target_cat)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            scheduler.step()

            # ── EMA target encoder update ─────────────────────────────────
            ema_update(context_encoder, target_encoder, CFG["ema_tau"])

            epoch_losses.append(loss.item())
            global_step += 1

            # ── logging + live plot update ────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                # effective rank on current batch context representations
                with torch.no_grad():
                    # flatten keep dim: [B*keep_enc, D]
                    z_flat = z_ctx.detach().reshape(-1, CFG["embed_dim"])
                    eff_rank = compute_effective_rank(z_flat)

                record = dict(
                    global_step    = global_step,
                    epoch          = epoch,
                    iter           = it,
                    loss           = round(loss.item(), 6),
                    effective_rank = round(eff_rank, 4),
                )
                records.append(record)

                log.info(
                    f"[ep {epoch:02d}|it {it:04d}|step {global_step:06d}]  "
                    f"loss={loss.item():.4f}  eff_rank={eff_rank:.2f}"
                )

                # ── incremental JSON save ──────────────────────────────────
                with open(JSON_PATH, "w") as f:
                    json.dump(records, f, indent=2)

                # ── live PNG update ────────────────────────────────────────
                save_plot(records)

        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {epoch:02d} complete  mean_loss={mean_ep_loss:.4f}")

    # ── final JSON flush ──────────────────────────────────────────────────
    with open(JSON_PATH, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Records saved → {JSON_PATH}  ({len(records)} entries)")

    log.info("Training complete.")
    return records


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()