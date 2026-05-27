# tjepa_training.py
# Trains T-JEPA for 300 epochs on local C4-subset (BERT-Large Settings).
# Logs JEPA loss + effective rank every 10 iters → ../Arg-I/T-JEPA.json
# Saves dual-axis plot (live update every 10 iters) → ../Arg-I/T-JEPA.png
#
# Usage (run from T-JEPA/ directory):
#   python tjepa_training.py
#
# Requires: tjepa_architecture.py, tjepa_dataloader.py, data/ populated
# ─────────────────────────────────────────────────────────────────────────────

import copy
import json
import math
import os
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

# ── resolve paths (GIỮ NGUYÊN GỐC 100%) ───────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../T-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = ARG_I_DIR / "T-JEPA.json"
PNG_PATH  = ARG_I_DIR / "T-JEPA.png"

# ── make sure sibling modules are importable ──────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from tjepa_architecture import TextJEPA
from tjepa_dataloader   import make_c4_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config - Đồng bộ hóa setting tương đồng với cấu hình mới của I-JEPA
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 32,             # Đồng bộ với cấu hình chịu tải lớn
    num_workers     = 10,              # Đồng bộ hóa worker xử lý dữ liệu
    max_length      = 256,
    pin_mem         = True,
    # masking (span)
    max_span_length = 5,
    max_num_spans   = 5,
    min_num_spans   = 1,
    allow_overlap   = False,
    # model (Nâng cấp cấu hình tương đương BERT-Large)
    model_name      = "bert_large",
    hidden_dim      = 1024,            # BERT-Large D
    predictor_dim   = 384,             # Giữ nguyên pred dim theo config
    predictor_layers= 12,              # Đồng bộ độ sâu predictor (depth=12)
    predictor_heads = 16,              # Đồng bộ số heads (heads=16)
    predictor_ffn_dim = 1536,
    use_bfloat16    = True,            # Bật bfloat16 cho mô hình lớn
    # optimiser & schedules động từ I-JEPA
    epochs          = 128,             # Nâng lên 300 epochs giống I-JEPA
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 40,              # 40 epochs khởi động (warmup)
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 1.0),    # EMA tăng dần từ 0.996 -> 1.0
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
# Schedulers (Đồng bộ thuật toán sinh LR, WD, EMA động từng step)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_wd_ema_schedulers(total_steps, steps_per_epoch):
    warmup_steps = CFG["warmup"] * steps_per_epoch
    
    lr_schedule = np.zeros(total_steps)
    wd_schedule = np.zeros(total_steps)
    ema_schedule = np.zeros(total_steps)
    
    for step in range(total_steps):
        # 1. Learning Rate Schedule (Warmup tuyến tính + Cosine Decay)
        if step < warmup_steps:
            lr_schedule[step] = CFG["start_lr"] + step * (CFG["lr"] - CFG["start_lr"]) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = CFG["final_lr"] + 0.5 * (CFG["lr"] - CFG["final_lr"]) * (1 + math.cos(math.pi * progress))
            
        # 2. Weight Decay Schedule (Cosine Annealing)
        progress = step / total_steps
        wd_schedule[step] = CFG["weight_decay"] + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"]) * (1 - math.cos(math.pi * progress))
        
        # 3. EMA Schedule (Cosine tăng dần từ 0.996 đến 1.0)
        ema_schedule[step] = CFG["ema_range"][1] - 0.5 * (CFG["ema_range"][1] - CFG["ema_range"][0]) * (1 + math.cos(math.pi * progress))
        
    return lr_schedule, wd_schedule, ema_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def save_plot(records: list[dict]):
    steps  = [r["global_step"]    for r in records]
    losses = [r["loss"]           for r in records]
    ranks  = [r["effective_rank"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = "#D85A30"
    color_rank = "#8B2500"

    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("MSE Loss (log scale)", color=color_loss, fontsize=12)
    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="T-JEPA loss")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=color_rank, fontsize=12)
    ax2.plot(steps, ranks, color=color_rank, linewidth=1.8, linestyle="--", label="T-JEPA eff. rank")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    current_step = steps[-1] if steps else 0
    plt.title(f"T-JEPA Training Dynamics (Loss & Effective Rank)  [step {current_step}]", fontsize=13)
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
    loader, _ = make_c4_dataloader(
        data_dir        = CFG["data_dir"],
        split           = "train",
        batch_size      = CFG["batch_size"],
        num_workers     = CFG["num_workers"],
        pin_mem         = CFG["pin_mem"],
        max_length      = CFG["max_length"],
        max_span_length = CFG["max_span_length"],
        max_num_spans   = CFG["max_num_spans"],
        min_num_spans   = CFG["min_num_spans"],
        seed            = 42,          
        drop_last       = True,
        persistent_workers = (CFG["num_workers"] > 0),
    )
    log.info(f"Dataset: {len(loader.dataset):,} sentences, {len(loader):,} batches/epoch")

    # ── model (Sử dụng cấu hình lớn) ──────────────────────────────────────
    model = TextJEPA(
        model_name        = CFG["model_name"],
        hidden_dim        = CFG["hidden_dim"],
        predictor_dim     = CFG["predictor_dim"],
        predictor_layers  = CFG["predictor_layers"],
        predictor_heads   = CFG["predictor_heads"],
        predictor_ffn_dim = CFG["predictor_ffn_dim"],
        max_length        = CFG["max_length"],
    ).to(device)

    ctx_params  = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
    pred_params = sum(p.numel() for p in model.predictor.parameters())       / 1e6
    log.info(
        f"Params — context_encoder: {ctx_params:.1f}M  "
        f"predictor: {pred_params:.1f}M  "
        f"(target_encoder: EMA copy, no grad)"
    )

    # ── optimiser & schedules ─────────────────────────────────────────────
    trainable_params = (
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters())
    )
    optimiser = torch.optim.AdamW(
        trainable_params,
        lr           = CFG["start_lr"],
        weight_decay = CFG["weight_decay"],
    )

    steps_per_epoch = len(loader)
    total_steps = CFG["epochs"] * steps_per_epoch
    
    # Tạo các lịch trình cập nhật động
    lr_sched, wd_sched, ema_sched = get_lr_wd_ema_schedulers(total_steps, steps_per_epoch)

    # ── training loop ─────────────────────────────────────────────────────
    records     = []
    global_step = 0

    epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Epochs", unit="ep", position=0)

    for epoch in epoch_bar:
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()   
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

        for it, batch in iter_bar:
            # Gán lr và weight decay biến thiên theo từng step huấn luyện
            current_lr = lr_sched[global_step]
            current_wd = wd_sched[global_step]
            current_ema = ema_sched[global_step]
            
            for param_group in optimiser.param_groups:
                param_group["lr"] = current_lr
                param_group["weight_decay"] = current_wd

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Kích hoạt bfloat16 autocast tương đồng I-JEPA
            with torch.amp.autocast(device_type="cuda", enabled=CFG["use_bfloat16"] and device.type == "cuda", dtype=torch.bfloat16):
                # ── forward ───────────────────────────────────────────────────
                out  = model(batch)
                loss = out["span_loss"]

            # ── backward ──────────────────────────────────────────────────
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()

            # ── EMA update of target encoder ──────────────────────────────
            model.update_target_encoder(decay=current_ema)

            epoch_losses.append(loss.item())

            # ── update iter progress bar ───────────────────────────────────
            iter_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.5f}", step=global_step)

            # ── logging + live plot update ────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    ctx_hidden = model._encode(
                        model.context_encoder,
                        batch["masked_input_ids"],
                        batch["masked_attention_mask"],
                        batch["masked_token_type_ids"],
                    )  

                    span_mask  = batch["span_mask"].bool()          
                    z_flat = ctx_hidden[span_mask]                  
                    if z_flat.shape[0] < 2:
                        z_flat = ctx_hidden.mean(dim=1)             

                    eff_rank = compute_effective_rank(z_flat.detach())

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
    log.info(f"Records saved → {JSON_PATH}  ({len(records)} entries)")

    log.info("Training complete.")
    return records


if __name__ == "__main__":
    train()