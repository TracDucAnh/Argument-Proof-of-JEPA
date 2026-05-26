# ijepa_training.py
# Trains I-JEPA for 300 epochs on local ImageNet-1K subset.
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
from tqdm import tqdm

# ── resolve paths (GIỮ NGUYÊN GỐC 100%) ───────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../I-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = ARG_I_DIR / "I-JEPA.json"
PNG_PATH  = ARG_I_DIR / "I-JEPA.png"

# ── make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from ijepa_architecture import vit_huge, vit_predictor
from ijepa_dataloader   import make_imagenet1k_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config - Cập nhật theo file thông số mới
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data", # Giữ nguyên cấu trúc thư mục data gốc
    batch_size      = 128,
    num_workers     = 10,
    crop_size       = 224,
    pin_mem         = True,
    # masking
    patch_size      = 14,
    enc_mask_scale  = (0.85, 1.0),
    pred_mask_scale = (0.15, 0.2),
    aspect_ratio    = (0.75, 1.5),
    num_enc_masks   = 1,
    num_pred_masks  = 4,
    min_keep        = 10,
    allow_overlap   = False,
    # model (Cấu hình vit_huge)
    model_name      = "vit_huge",
    embed_dim       = 1280,   # ViT-H
    pred_embed_dim  = 384,
    pred_depth      = 12,
    pred_heads      = 16,
    use_bfloat16    = True,
    # optimiser & schedules
    epochs          = 20,
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 40,     # epochs warmup
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 1.0),
    # training
    log_every       = 10,    
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schedulers (LR & WD & EMA)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_wd_ema_schedulers(total_steps, steps_per_epoch):
    warmup_steps = CFG["warmup"] * steps_per_epoch
    
    lr_schedule = np.zeros(total_steps)
    wd_schedule = np.zeros(total_steps)
    ema_schedule = np.zeros(total_steps)
    
    for step in range(total_steps):
        # LR Schedule (Warmup + Cosine)
        if step < warmup_steps:
            lr_schedule[step] = CFG["start_lr"] + step * (CFG["lr"] - CFG["start_lr"]) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = CFG["final_lr"] + 0.5 * (CFG["lr"] - CFG["final_lr"]) * (1 + math.cos(math.pi * progress))
            
        # Weight Decay Schedule (Cosine)
        progress = step / total_steps
        wd_schedule[step] = CFG["weight_decay"] + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"]) * (1 - math.cos(math.pi * progress))
        
        # EMA Schedule (Cosine tăng dần từ 0.996 -> 1.0)
        ema_schedule[step] = CFG["ema_range"][1] - 0.5 * (CFG["ema_range"][1] - CFG["ema_range"][0]) * (1 + math.cos(math.pi * progress))
        
    return lr_schedule, wd_schedule, ema_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (Giữ nguyên gốc)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_effective_rank(z: torch.Tensor) -> float:
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov)          
    except Exception:
        return float("nan")
    eigvals = eigvals.clamp(min=0.0)
    total   = eigvals.sum()
    if total < 1e-12:
        return 1.0
    p    = eigvals / total                             
    mask = p > 1e-12
    H    = -(p[mask] * torch.log(p[mask])).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


def save_plot(records: list[dict]):
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
    ax2.plot(steps, ranks, color=color_rank, linewidth=1.8, linestyle="--", label="I-JEPA eff. rank")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    current_step = steps[-1] if steps else 0
    plt.title(f"I-JEPA Training Dynamics (Loss & Effective Rank)  [step {current_step}]", fontsize=13)
    fig.tight_layout()

    tmp_path = PNG_PATH.with_suffix(".tmp.png")
    fig.savefig(str(tmp_path), dpi=150)
    plt.close(fig)
    tmp_path.replace(PNG_PATH)


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
        pin_mem         = CFG["pin_mem"],
        crop_size       = CFG["crop_size"],
        patch_size      = CFG["patch_size"],
        enc_mask_scale  = CFG["enc_mask_scale"],
        pred_mask_scale = CFG["pred_mask_scale"],
        aspect_ratio    = CFG["aspect_ratio"],
        num_enc_masks   = CFG["num_enc_masks"],
        num_pred_masks  = CFG["num_pred_masks"],
        min_keep        = CFG["min_keep"],
        allow_overlap   = CFG["allow_overlap"],
        use_masking     = True,
        persistent_workers = True if CFG["num_workers"] > 0 else False,
    )
    log.info(f"Dataset: {len(loader.dataset):,} images, {len(loader):,} batches/epoch")

    N_PATCHES = (CFG["crop_size"] // CFG["patch_size"]) ** 2   # (224 // 14)^2 = 256

    # ── models (Đổi sang vit_huge) ─────────────────────────────────────────
    context_encoder = vit_huge(
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
        f"Params — encoder: {sum(p.numel() for p in context_encoder.parameters())/1e6:.1f}M  "
        f"predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M"
    )

    # ── optimiser & schedules ─────────────────────────────────────────────
    params = list(context_encoder.parameters()) + list(predictor.parameters())
    optimiser = torch.optim.AdamW(
        params, lr=CFG["start_lr"], weight_decay=CFG["weight_decay"]
    )

    steps_per_epoch = len(loader)
    total_steps = CFG["epochs"] * steps_per_epoch
    
    # Sinh mảng giá trị lr, wd, ema động từng step
    lr_sched, wd_sched, ema_sched = get_lr_wd_ema_schedulers(total_steps, steps_per_epoch)

    # ── training loop ─────────────────────────────────────────────────────
    records      = []
    global_step  = 0

    epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Epochs", unit="ep", position=0)

    for epoch in epoch_bar:
        context_encoder.train()
        predictor.train()
        epoch_losses = []

        iter_bar = tqdm(
            enumerate(loader, start=1),
            total         = len(loader),
            desc          = f"Ep {epoch:03d}",
            unit          = "it",
            position      = 1,
            leave         = False,
            dynamic_ncols = True,
        )

        for it, (imgs, masks_enc, masks_pred) in iter_bar:
            # Gán lr và weight decay động theo đúng step hiện hành
            current_lr = lr_sched[global_step]
            current_wd = wd_sched[global_step]
            current_ema = ema_sched[global_step]
            
            for param_group in optimiser.param_groups:
                param_group["lr"] = current_lr
                param_group["weight_decay"] = current_wd

            imgs       = imgs.to(device, non_blocking=True)
            masks_enc  = masks_enc.to(device,  non_blocking=True)   
            masks_pred = masks_pred.to(device, non_blocking=True)   

            masks_enc_flat  = masks_enc[:, 0, :]
            masks_pred_list = [masks_pred[:, k, :] for k in range(masks_pred.shape[1])]

            # Tính toán amp autocast dạng bfloat16 nếu cấu hình bật
            with torch.amp.autocast(device_type="cuda", enabled=CFG["use_bfloat16"] and device.type == "cuda", dtype=torch.bfloat16):
                # ── target representations (no grad) ──────────────────────────
                with torch.no_grad():
                    z_target_full = target_encoder(imgs)   
                    z_targets = []
                    for m in masks_pred_list:
                        idx  = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                        z_targets.append(torch.gather(z_target_full, 1, idx))
                    z_target_cat = torch.cat(z_targets, dim=0)

                # ── context encoder ───────────────────────────────────────────
                z_ctx = context_encoder(imgs, masks=masks_enc_flat)   

                # ── predictor ─────────────────────────────────────────────────
                z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)

                # ── JEPA loss ─────────────────────────────────────────────────
                loss = F.mse_loss(z_pred, z_target_cat)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()

            # ── EMA target encoder update ─────────────────────────────────
            ema_update(context_encoder, target_encoder, current_ema)

            epoch_losses.append(loss.item())
            
            iter_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.5f}", step=global_step)

            # ── logging + live plot update ────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
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
                    f"[ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                    f"loss={loss.item():.4f}  eff_rank={eff_rank:.2f}  lr={current_lr:.6f}"
                )

                with open(JSON_PATH, "w") as f:
                    json.dump(records, f, indent=2)

                save_plot(records)
                
            global_step += 1

        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {epoch:03d} complete  mean_loss={mean_ep_loss:.4f}")
        epoch_bar.set_postfix(mean_loss=f"{mean_ep_loss:.4f}")

    with open(JSON_PATH, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Final records saved → {JSON_PATH}")
    return records


if __name__ == "__main__":
    train()