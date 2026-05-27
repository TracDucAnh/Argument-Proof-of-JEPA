# tjepa_training.py  (fair-comparison edition)
# ─────────────────────────────────────────────────────────────────────────────
# Trains T-JEPA for N epochs on local C4-subset (BERT-Large settings).
# Logs JEPA loss + FAIR effective rank metrics every log_every iters.
#
# FAIR COMPARISON CHANGES vs. original:
#   • Effective rank computed from ALL tokens via encode_full_sequence()
#     (not just span tokens) → apples-to-apples with I-JEPA's all-patch approach
#   • Logs 3 rank metrics: effective_rank, normalized_rank, participation_ratio
#   • JSON schema extended with all 3 metrics for cross-model plotting
#   • Dual-axis plot shows normalized_rank (0–1) → directly comparable with I-JEPA
#   • Output files: ../Arg-I/T-JEPA.json  and  ../Arg-I/T-JEPA.png  (unchanged)
#
# Usage (run from T-JEPA/ directory):
#   python tjepa_training.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../T-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = ARG_I_DIR / "T-JEPA.json"
PNG_PATH  = ARG_I_DIR / "T-JEPA.png"

sys.path.insert(0, str(SCRIPT_DIR))
from tjepa_architecture import TextJEPA, compute_effective_rank
from tjepa_dataloader   import make_c4_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 32,
    num_workers     = 10,
    max_length      = 256,
    pin_mem         = True,
    # masking (span)
    max_span_length = 5,
    max_num_spans   = 5,
    min_num_spans   = 1,
    allow_overlap   = False,
    # model (BERT-Large)
    model_name      = "bert_large",
    hidden_dim      = 1024,
    predictor_dim   = 384,
    predictor_layers= 12,
    predictor_heads = 16,
    predictor_ffn_dim = 1536,
    use_bfloat16    = True,
    # optimiser & schedules
    epochs          = 128,
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 40,
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
# Schedulers
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_wd_ema_schedulers(total_steps, steps_per_epoch):
    warmup_steps = CFG["warmup"] * steps_per_epoch
    lr_schedule  = np.zeros(total_steps)
    wd_schedule  = np.zeros(total_steps)
    ema_schedule = np.zeros(total_steps)

    for step in range(total_steps):
        # LR: linear warmup → cosine decay
        if step < warmup_steps:
            lr_schedule[step] = (CFG["start_lr"]
                + step * (CFG["lr"] - CFG["start_lr"]) / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = (CFG["final_lr"]
                + 0.5 * (CFG["lr"] - CFG["final_lr"]) * (1 + math.cos(math.pi * progress)))

        # WD: cosine annealing
        progress = step / total_steps
        wd_schedule[step] = (CFG["weight_decay"]
            + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"])
            * (1 - math.cos(math.pi * progress)))

        # EMA: cosine increase 0.996 → 1.0
        ema_schedule[step] = (CFG["ema_range"][1]
            - 0.5 * (CFG["ema_range"][1] - CFG["ema_range"][0])
            * (1 + math.cos(math.pi * progress)))

    return lr_schedule, wd_schedule, ema_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Plot — normalized_rank (0–1) on right axis, directly comparable to I-JEPA
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(records: list[dict]):
    steps  = [r["global_step"]    for r in records]
    losses = [r["loss"]           for r in records]
    nranks = [r["normalized_rank"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = "#D85A30"
    color_rank = "#8B2500"

    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("MSE Loss (log scale)", color=color_loss, fontsize=12)
    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="T-JEPA loss")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Normalized Effective Rank  (rank / embed_dim)", color=color_rank, fontsize=12)
    ax2.plot(steps, nranks, color=color_rank, linewidth=1.8, linestyle="--",
             label="T-JEPA norm. eff. rank")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(0, 1)   # fixed 0–1 range → directly comparable with I-JEPA plot

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    current_step = steps[-1] if steps else 0
    plt.title(
        f"T-JEPA Training Dynamics  [step {current_step}]\n"
        f"(right axis: normalized rank = eff_rank / {CFG['hidden_dim']}  →  0–1)",
        fontsize=12)
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

    # ── model ─────────────────────────────────────────────────────────────
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
    trainable_params = (list(model.context_encoder.parameters()) +
                        list(model.predictor.parameters()))
    optimiser = torch.optim.AdamW(trainable_params, lr=CFG["start_lr"],
                                  weight_decay=CFG["weight_decay"])

    steps_per_epoch = len(loader)
    total_steps     = CFG["epochs"] * steps_per_epoch
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
            total=len(loader), desc=f"Ep {epoch:03d}", unit="it",
            position=1, leave=False, dynamic_ncols=True,
        )

        for it, batch in iter_bar:
            current_lr  = lr_sched[global_step]
            current_wd  = wd_sched[global_step]
            current_ema = ema_sched[global_step]

            for pg in optimiser.param_groups:
                pg["lr"]           = current_lr
                pg["weight_decay"] = current_wd

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda",
                                    enabled=CFG["use_bfloat16"] and device.type == "cuda",
                                    dtype=torch.bfloat16):
                out  = model(batch)
                loss = out["span_loss"]

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            model.update_target_encoder(decay=current_ema)

            epoch_losses.append(loss.item())
            iter_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.5f}",
                                 step=global_step)

            # ── logging ────────────────────────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    # FAIR: encode FULL clean sequence (all tokens, no span filter)
                    # mirrors I-JEPA's forward_all_patches() approach
                    z_full = model.encode_full_sequence(batch, use_target=False)
                    # [B, L, D] → [B*L, D]
                    z_flat = z_full.detach().reshape(-1, CFG["hidden_dim"])
                    rank_info = compute_effective_rank(z_flat, embed_dim=CFG["hidden_dim"])

                record = dict(
                    global_step        = global_step,
                    epoch              = epoch,
                    iter               = it,
                    loss               = round(loss.item(), 6),
                    # ── 3 rank metrics (all logged for cross-model comparison) ──
                    effective_rank     = round(rank_info["effective_rank"],      4),
                    normalized_rank    = round(rank_info["normalized_rank"],     6),
                    participation_ratio= round(rank_info["participation_ratio"], 4),
                    embed_dim          = CFG["hidden_dim"],
                    model              = "T-JEPA",
                )
                records.append(record)

                log.info(
                    f"[ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                    f"loss={loss.item():.4f}  "
                    f"eff_rank={rank_info['effective_rank']:.2f}  "
                    f"norm_rank={rank_info['normalized_rank']:.4f}  "
                    f"pr={rank_info['participation_ratio']:.2f}  "
                    f"lr={current_lr:.6f}"
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