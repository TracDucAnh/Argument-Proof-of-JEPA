# tjepa_training.py
# Trains T-JEPA for 10 epochs on local C4-subset.
# Logs JEPA loss + effective rank every 10 iters → ../Arg-I/T-JEPA.json
# Saves dual-axis plot (live update every 10 iters) → ../Arg-I/T-JEPA.png
#
# Usage (run from T-JEPA/ directory):
#   python tjepa_training.py
#
# Requires: tjepa_architecture.py, tjepa_dataloader.py, data/ populated
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── resolve paths ─────────────────────────────────────────────────────────────
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
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 256,
    num_workers     = 4,
    max_length      = 256,
    # masking (span)
    max_span_length = 5,
    max_num_spans   = 5,
    min_num_spans   = 1,
    # model
    hidden_dim         = 768,   # BERT-base D
    predictor_dim      = 384,   # D/2
    predictor_layers   = 4,
    predictor_heads    = 6,
    predictor_ffn_dim  = 1536,
    # optimiser
    lr              = 1e-3,
    weight_decay    = 0.05,
    # EMA
    ema_decay       = 0.996,
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
# Helpers  (identical logic to ijepa_training.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_effective_rank(z: torch.Tensor) -> float:
    """
    Effective rank of the representation covariance.

    effective_rank = exp( H( λ_i / Σλ_i ) )
    where λ_i are eigenvalues of the empirical covariance of z.

    Parameters
    ----------
    z : Tensor [N, D]  — batch of representation vectors (any token positions)
    """
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)                      # centre
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)                # [D, D]
    try:
        eigvals = torch.linalg.eigvalsh(cov)                 # [D], ascending
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
    """
    Dual-axis line chart: left axis = loss, right axis = effective rank.
    Được gọi sau mỗi log_every iteration để cập nhật PNG liên tục (live update).
    """
    steps  = [r["global_step"]    for r in records]
    losses = [r["loss"]           for r in records]
    ranks  = [r["effective_rank"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = "#D85A30"   # T-JEPA orange-red (consistent with paper palette)
    color_rank = "#8B2500"   # darker shade for rank

    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("MSE Loss", color=color_loss, fontsize=12)
    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="T-JEPA loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Rank", color=color_rank, fontsize=12)
    ax2.plot(steps, ranks, color=color_rank, linewidth=1.8,
             linestyle="--", label="T-JEPA eff. rank")
    ax2.tick_params(axis="y", labelcolor=color_rank)
    ax2.set_ylim(bottom=0)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    # Hiển thị step hiện tại trên tiêu đề
    current_step = steps[-1] if steps else 0
    plt.title(
        f"T-JEPA Training Dynamics (Loss & Effective Rank)  [step {current_step}]",
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
    loader, _ = make_c4_dataloader(
        data_dir        = CFG["data_dir"],
        split           = "train",
        batch_size      = CFG["batch_size"],
        num_workers     = CFG["num_workers"],
        pin_mem         = (device.type == "cuda"),
        max_length      = CFG["max_length"],
        max_span_length = CFG["max_span_length"],
        max_num_spans   = CFG["max_num_spans"],
        min_num_spans   = CFG["min_num_spans"],
        seed            = None,          # fully random per epoch (recommended)
        drop_last       = True,
        persistent_workers = (CFG["num_workers"] > 0),
    )
    log.info(
        f"Dataset: {len(loader.dataset):,} sentences, "
        f"{len(loader):,} batches/epoch, bs={CFG['batch_size']}"
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = TextJEPA(
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

    # ── optimiser ─────────────────────────────────────────────────────────
    # Only context_encoder + predictor are trained (target_encoder is EMA)
    trainable_params = (
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters())
    )
    optimiser = torch.optim.AdamW(
        trainable_params,
        lr           = CFG["lr"],
        weight_decay = CFG["weight_decay"],
    )

    total_steps = CFG["epochs"] * len(loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps, eta_min=1e-5
    )

    # ── training loop ─────────────────────────────────────────────────────
    records     = []
    global_step = 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()   # target encoder: always eval mode
        epoch_losses = []

        for it, batch in enumerate(loader, start=1):
            # move every tensor in the batch dict to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ── forward ───────────────────────────────────────────────────
            out  = model(batch)
            loss = out["span_loss"]

            # ── backward ──────────────────────────────────────────────────
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            scheduler.step()

            # ── EMA update of target encoder ──────────────────────────────
            model.update_target_encoder(decay=CFG["ema_decay"])

            epoch_losses.append(loss.item())
            global_step += 1

            # ── logging + live plot update ────────────────────────────────
            if global_step % CFG["log_every"] == 0:
                # Effective rank: flatten [B, L, D] context hidden over span positions
                with torch.no_grad():
                    # out["predicted_hidden"] already computed; use context
                    # Re-use the target hidden (detached) from the forward pass.
                    # For rank we want the context encoder representations:
                    # run a quick no-grad pass on context encoder output.
                    ctx_hidden = model._encode(
                        model.context_encoder,
                        batch["masked_input_ids"],
                        batch["masked_attention_mask"],
                        batch["masked_token_type_ids"],
                    )  # [B, L, D]

                    # Pool over span positions per sample, then flatten to [N, D]
                    span_mask  = batch["span_mask"].bool()          # [B, L]
                    # gather all span token representations → [total_span_tokens, D]
                    z_flat = ctx_hidden[span_mask]                  # [M, D]
                    if z_flat.shape[0] < 2:
                        # fallback: use mean-pooled per sample
                        z_flat = ctx_hidden.mean(dim=1)             # [B, D]

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