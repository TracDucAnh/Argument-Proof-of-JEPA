# tjepa_training.py  (fair-comparison edition + Argument I-II metrics + MoCo Queue + Held-Out Eval)
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
#   • Output files: ../Arg-I/T-JEPA.json  and  ../Arg-I/T-JEPA.png
#
# ARGUMENT I METRICS:
#   • residual_var — Var(ẑ_T − z_T) computed over SPAN positions only
#
# ARGUMENT II METRICS:
#   • lambda_min, lambda_min_ratio, cosine_sim_mean/std/p95/hist
#
# MOCO QUEUE:
#   • moco_queue_size = 2048  — circular FIFO queue of target span representations
#
# HELD-OUT EVALUATION (new):
#   • Every eval_every steps a full pass over the val DataLoader is run.
#   • The model and target encoder are set to eval() during this pass;
#     NO gradients, NO parameter updates, NO batch-norm stat changes.
#   • ALL metrics computed on held-out data: loss, effective_rank, normalized_rank,
#     participation_ratio, residual_var, lambda_min, lambda_min_ratio,
#     cosine_sim_mean, cosine_sim_std, cosine_sim_p95, cosine_sim_hist.
#   • Records saved to a SEPARATE JSON: ../Arg-I/T-JEPA_val.json
#   • Plots show BOTH train (solid) and val (dashed) curves on the same axes.
#   • Why held-out is the right choice:
#       – Training metrics are computed on batches that just updated the gradient,
#         creating a confirmation bias — the model has already "seen" those tokens.
#       – Held-out metrics reflect genuine generalisation of the representation,
#         which is the quantity of interest for Argument I/II claims.
#       – Reviewer standard: any empirical claim about representation quality
#         (rank, collapse, irreducible variance) must be supported by out-of-
#         distribution evidence, not in-sample statistics.
#
# I/O ORDERING GUARANTEE:
#   All disk writes go through save_records() → save_checkpoint() in strict order:
#     1. Train JSON  (atomic)
#     2. Val   JSON  (atomic)
#     3. PNG         (atomic)
#     4. Checkpoint  (atomic, epoch-end only)
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
import torch.nn.functional as F

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../T-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH     = ARG_I_DIR / "T-JEPA.json"
JSON_VAL_PATH = ARG_I_DIR / "T-JEPA_val.json"
PNG_PATH      = ARG_I_DIR / "T-JEPA.png"
CKPT_PATH     = ARG_I_DIR / "T-JEPA_latest.pt"

sys.path.insert(0, str(SCRIPT_DIR))
from tjepa_architecture import TextJEPA, compute_effective_rank
from tjepa_dataloader   import make_c4_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 64,
    num_workers     = 4,
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
    epochs          = 20,
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 40,
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 0.999),
    # training
    log_every       = 10,
    # ── HELD-OUT EVAL ─────────────────────────────────────────────────────
    # Every eval_every steps: full val-split pass, all metrics, separate JSON
    eval_every      = 100,
    eval_max_batches = None,   # None = full val set; set int to cap (faster debug)
    # ── ARGUMENT I metrics ────────────────────────────────────────────────
    arg1_every       = 10,
    # ── ARGUMENT II metrics ───────────────────────────────────────────────
    arg2_every       = 10,
    arg2_sample_size = 2048,
    # ── MOCO QUEUE ────────────────────────────────────────────────────────
    moco_queue_size  = 2048,
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MoCo Queue
# ─────────────────────────────────────────────────────────────────────────────

class MoCoQueue:
    """
    Circular FIFO queue that stores L2-normalised target-encoder span keys.

    Design:
        • Fixed-size buffer of shape [queue_size, embed_dim] on `device`.
        • ptr tracks the next write position (wraps around modulo queue_size).
        • Keys are stored as float32 regardless of training dtype.
        • No gradients flow through the queue (all ops are @torch.no_grad).
    """

    def __init__(self, queue_size: int, embed_dim: int, device: torch.device):
        self.queue_size = queue_size
        self.embed_dim  = embed_dim
        self.device     = device

        buf = torch.randn(queue_size, embed_dim, device=device)
        self.buffer = F.normalize(buf, p=2, dim=1)
        self.ptr    = 0
        self.full   = False

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        keys = F.normalize(keys.detach().float(), p=2, dim=1)
        B    = keys.shape[0]

        end_ptr = self.ptr + B
        if end_ptr <= self.queue_size:
            self.buffer[self.ptr:end_ptr] = keys
        else:
            first  = self.queue_size - self.ptr
            second = B - first
            self.buffer[self.ptr:] = keys[:first]
            self.buffer[:second]   = keys[first:]
            self.full = True

        self.ptr = end_ptr % self.queue_size
        if end_ptr >= self.queue_size:
            self.full = True

    @torch.no_grad()
    def get_keys(self) -> torch.Tensor:
        if self.full:
            return self.buffer.clone()
        else:
            return self.buffer[:self.ptr].clone()

    def __len__(self) -> int:
        return self.queue_size if self.full else self.ptr


# ─────────────────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_wd_ema_schedulers(total_steps, steps_per_epoch):
    warmup_steps = CFG["warmup"] * steps_per_epoch
    lr_schedule  = np.zeros(total_steps)
    wd_schedule  = np.zeros(total_steps)
    ema_schedule = np.zeros(total_steps)

    for step in range(total_steps):
        if step < warmup_steps:
            lr_schedule[step] = (CFG["start_lr"]
                + step * (CFG["lr"] - CFG["start_lr"]) / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = (CFG["final_lr"]
                + 0.5 * (CFG["lr"] - CFG["final_lr"]) * (1 + math.cos(math.pi * progress)))

        progress = step / total_steps
        wd_schedule[step] = (CFG["weight_decay"]
            + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"])
            * (1 - math.cos(math.pi * progress)))

        ema_schedule[step] = (CFG["ema_range"][1]
            - 0.5 * (CFG["ema_range"][1] - CFG["ema_range"][0])
            * (1 + math.cos(math.pi * progress)))

    return lr_schedule, wd_schedule, ema_schedule


# ─────────────────────────────────────────────────────────────────────────────
# Argument I metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_residual_variance(
    z_pred: torch.Tensor,
    z_tgt:  torch.Tensor,
) -> float:
    """
    Residual variance = Var(ẑ_T − z_T) computed over span positions.

    Proxy for the irreducible variance term in Proposition 4.12.
    For text: lexical ambiguity creates a floor → residual_var plateaus.
    For images: predictor improves → residual_var decreases.

    Args:
        z_pred: [N_span, D] predictor output ẑ_T  (detached, span tokens only)
        z_tgt:  [N_span, D] target encoder output z_T (detached, span tokens only)

    Returns:
        float: scalar total variance of the residual. nan if N_span == 0.
    """
    if z_pred.shape[0] == 0:
        return float("nan")

    residual  = (z_pred.float() - z_tgt.float())
    mean_res  = residual.mean(dim=0, keepdim=True)
    centered  = residual - mean_res
    var       = (centered ** 2).mean().item()
    return round(var, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Argument II metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_arg2_metrics(z_flat: torch.Tensor, embed_dim: int) -> dict:
    """
    Compute Argument II collapse metrics from a flat [N, D] representation matrix.

    Returns dict with:
        lambda_min        — smallest eigenvalue of Σ_z
        lambda_min_ratio  — lambda_min / lambda_max
        cosine_sim_mean   — mean pairwise cosine similarity
        cosine_sim_std    — std  pairwise cosine similarity
        cosine_sim_p95    — 95th percentile
        cosine_sim_hist   — 10-bin histogram counts over [-1, 1]
    """
    z = z_flat.float()

    # ── λ_min of Σ_z ──────────────────────────────────────────────────────
    mean_z     = z.mean(dim=0, keepdim=True)
    z_centered = z - mean_z
    cov        = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)
    try:
        eigvals          = torch.linalg.eigvalsh(cov)
        lambda_min       = eigvals[0].item()
        lambda_max       = eigvals[-1].item()
        lambda_min_ratio = (lambda_min / lambda_max) if abs(lambda_max) > 1e-12 else 0.0
    except Exception:
        lambda_min       = float("nan")
        lambda_min_ratio = float("nan")

    # ── Pairwise cosine similarity (subsampled) ────────────────────────────
    N = z.shape[0]
    sample_size = min(N, CFG["arg2_sample_size"])
    if sample_size < N:
        idx      = torch.randperm(N, device=z.device)[:sample_size]
        z_sample = z[idx]
    else:
        z_sample = z

    z_norm  = F.normalize(z_sample, p=2, dim=1)
    sim_mat = z_norm @ z_norm.T
    S       = z_norm.shape[0]
    triu_idx = torch.triu_indices(S, S, offset=1, device=z.device)
    sim_vals = sim_mat[triu_idx[0], triu_idx[1]]

    sim_cpu = sim_vals.cpu().float().numpy()
    cosine_sim_mean = float(np.mean(sim_cpu))
    cosine_sim_std  = float(np.std(sim_cpu))
    cosine_sim_p95  = float(np.percentile(sim_cpu, 95))

    hist_counts, _ = np.histogram(sim_cpu, bins=10, range=(-1.0, 1.0))
    cosine_sim_hist = hist_counts.tolist()

    return dict(
        lambda_min        = round(lambda_min,       8),
        lambda_min_ratio  = round(lambda_min_ratio, 8),
        cosine_sim_mean   = round(cosine_sim_mean,  6),
        cosine_sim_std    = round(cosine_sim_std,   6),
        cosine_sim_p95    = round(cosine_sim_p95,   6),
        cosine_sim_hist   = cosine_sim_hist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Held-out evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_held_out_eval(
    model: "TextJEPA",
    val_loader,
    device: torch.device,
    global_step: int,
    epoch: int,
) -> dict:
    """
    Run a full pass over the validation DataLoader and compute ALL metrics.

    Why this is the correct approach:
        • No gradient update has touched these samples → zero confirmation bias.
        • Aggregating over the entire val split (or eval_max_batches batches)
          gives low-variance estimates suitable for Argument I/II claims.
        • The representation quality we care about — whether T-JEPA collapses,
          whether residual_var plateaus — should hold on unseen data, not just
          on the batch that just trained the model.

    Metrics computed (identical formulae to the training-time versions):
        loss, effective_rank, normalized_rank, participation_ratio,
        residual_var, lambda_min, lambda_min_ratio,
        cosine_sim_mean, cosine_sim_std, cosine_sim_p95, cosine_sim_hist

    All intermediate tensors are accumulated across batches before the final
    eigenspectrum / cosine-sim computation, so the result reflects the full
    distribution of val representations — not just one mini-batch.

    Args:
        model:       TextJEPA (context + target encoder + predictor)
        val_loader:  DataLoader over the val split
        device:      torch.device
        global_step: current training step (for record labelling)
        epoch:       current epoch

    Returns:
        dict with all metrics plus global_step, epoch, split="val"
    """
    model.context_encoder.eval()
    model.predictor.eval()
    model.target_encoder.eval()

    total_loss    = 0.0
    n_batches     = 0

    # Accumulate representations for rank / Arg-II across all val batches
    z_flat_list   = []   # [B*L, D] per batch → cat for rank & arg2
    # Accumulate span reps for residual_var
    z_pred_span_list = []
    z_tgt_span_list  = []

    max_batches = CFG["eval_max_batches"]

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast(
            device_type="cuda",
            enabled=CFG["use_bfloat16"] and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            out  = model(batch)
            loss = out["span_loss"]

        total_loss += loss.item()
        n_batches  += 1

        # Full-sequence representations for rank / Arg-II
        z_full = model.encode_full_sequence(batch, use_target=False)
        z_flat_list.append(
            z_full.detach().reshape(-1, CFG["hidden_dim"]).float().cpu()
        )

        # Span representations for residual_var (Arg I)
        span_mask_bool = batch["span_mask"].bool()
        z_pred_span = out["predicted_hidden"].detach()[span_mask_bool]
        z_tgt_span  = out["target_hidden"].detach()[span_mask_bool]
        z_pred_span_list.append(z_pred_span.float().cpu())
        z_tgt_span_list.append(z_tgt_span.float().cpu())

    # ── Aggregate ──────────────────────────────────────────────────────────
    mean_loss = total_loss / max(n_batches, 1)

    # Rank metrics — computed on the full accumulated z_flat
    z_flat_all = torch.cat(z_flat_list, dim=0).to(device)   # [N_total, D]
    # Subsample if too large to avoid OOM on eigendecomposition
    if z_flat_all.shape[0] > 32768:
        idx        = torch.randperm(z_flat_all.shape[0], device=device)[:32768]
        z_flat_all = z_flat_all[idx]

    rank_info = compute_effective_rank(z_flat_all, embed_dim=CFG["hidden_dim"])

    # Residual variance — span tokens only
    z_pred_all = torch.cat(z_pred_span_list, dim=0).to(device)
    z_tgt_all  = torch.cat(z_tgt_span_list,  dim=0).to(device)
    if z_pred_all.shape[0] > 32768:
        idx        = torch.randperm(z_pred_all.shape[0], device=device)[:32768]
        z_pred_all = z_pred_all[idx]
        z_tgt_all  = z_tgt_all[idx]
    res_var = compute_residual_variance(z_pred_all, z_tgt_all)

    # Arg II collapse metrics
    arg2 = compute_arg2_metrics(z_flat_all, embed_dim=CFG["hidden_dim"])

    val_record = dict(
        global_step         = global_step,
        epoch               = epoch,
        split               = "val",
        loss                = round(mean_loss,                            6),
        effective_rank      = round(rank_info["effective_rank"],          4),
        normalized_rank     = round(rank_info["normalized_rank"],         6),
        participation_ratio = round(rank_info["participation_ratio"],     4),
        embed_dim           = CFG["hidden_dim"],
        model               = "T-JEPA",
        residual_var        = res_var,
        **arg2,
    )

    log.info(
        f"  [VAL step {global_step:06d}]  "
        f"loss={mean_loss:.4f}  "
        f"eff_rank={rank_info['effective_rank']:.2f}  "
        f"norm_rank={rank_info['normalized_rank']:.4f}  "
        f"res_var={res_var:.6f}  "
        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
        f"cos_p95={arg2['cosine_sim_p95']:.4f}  "
        f"n_batches={n_batches}"
    )

    # Restore training mode
    model.context_encoder.train()
    model.predictor.train()
    # target_encoder stays eval (no grad, EMA only)

    return val_record


# ─────────────────────────────────────────────────────────────────────────────
# Atomic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json_atomic(records: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_plot_atomic(
    train_records: list[dict],
    val_records:   list[dict],
) -> None:
    """
    4-panel plot with train (solid) and val (dashed) curves overlaid.

    Panel 0 — Loss + Normalized Rank
    Panel 1 — Arg I: Residual Variance
    Panel 2 — Arg II: Cosine Sim Mean + λ_min ratio
    Panel 3 — (removed MoCo queue) → replaced with Participation Ratio comparison
    """

    def _extract(records, key):
        return (
            [r["global_step"] for r in records if key in r],
            [r[key]           for r in records if key in r],
        )

    # ── train traces ──────────────────────────────────────────────────────
    tr_steps,  tr_loss   = _extract(train_records, "loss")
    _,         tr_nrank  = _extract(train_records, "normalized_rank")
    tr_s1,     tr_rv     = _extract(train_records, "residual_var")
    tr_s2,     tr_cos    = _extract(train_records, "cosine_sim_mean")
    _,         tr_lam    = _extract(train_records, "lambda_min_ratio")
    tr_s3,     tr_prat   = _extract(train_records, "participation_ratio")

    # ── val traces ────────────────────────────────────────────────────────
    vl_steps,  vl_loss   = _extract(val_records, "loss")
    _,         vl_nrank  = _extract(val_records, "normalized_rank")
    vl_s1,     vl_rv     = _extract(val_records, "residual_var")
    vl_s2,     vl_cos    = _extract(val_records, "cosine_sim_mean")
    _,         vl_lam    = _extract(val_records, "lambda_min_ratio")
    vl_s3,     vl_prat   = _extract(val_records, "participation_ratio")

    fig, axes = plt.subplots(1, 4, figsize=(28, 5))

    # ── Panel 0: Loss + Normalized Rank ───────────────────────────────────
    ax0 = axes[0]
    c_loss = "#D85A30"
    c_rank = "#8B2500"

    ax0.set_xlabel("Training steps", fontsize=11)
    ax0.set_ylabel("MSE Loss (log scale)", color=c_loss, fontsize=11)
    if tr_steps:
        ax0.plot(tr_steps, tr_loss, color=c_loss, linewidth=1.8,
                 label="Train loss")
    if vl_steps:
        ax0.plot(vl_steps, vl_loss, color=c_loss, linewidth=1.8,
                 linestyle="--", label="Val loss")
    ax0.set_yscale("log")
    ax0.tick_params(axis="y", labelcolor=c_loss)

    ax0r = ax0.twinx()
    ax0r.set_ylabel("Normalized Effective Rank", color=c_rank, fontsize=10)
    if tr_steps:
        ax0r.plot(tr_steps, tr_nrank, color=c_rank, linewidth=1.8,
                  linestyle="-", label="Train norm.rank")
    if vl_steps:
        ax0r.plot(vl_steps, vl_nrank, color=c_rank, linewidth=1.8,
                  linestyle=":", label="Val norm.rank")
    ax0r.tick_params(axis="y", labelcolor=c_rank)
    ax0r.set_ylim(0, 1)

    lines0, lbls0   = ax0.get_legend_handles_labels()
    lines0r, lbls0r = ax0r.get_legend_handles_labels()
    ax0.legend(lines0 + lines0r, lbls0 + lbls0r, loc="upper right", fontsize=8)
    ax0.set_title("Loss & Effective Rank", fontsize=11)

    # ── Panel 1: Arg I — Residual Variance ────────────────────────────────
    ax1 = axes[1]
    c_rv_tr = "#FF7F0E"
    c_rv_vl = "#9B4C00"

    ax1.set_xlabel("Training steps", fontsize=11)
    ax1.set_ylabel("Residual Var  Var(ẑ_T − z_T)  [span tokens]",
                   color=c_rv_tr, fontsize=10)
    if tr_s1:
        ax1.plot(tr_s1, tr_rv, color=c_rv_tr, linewidth=1.8,
                 label="Train residual_var")
        ax1.axhline(y=tr_rv[-1], color=c_rv_tr, linewidth=0.8, linestyle=":",
                    alpha=0.55, label=f"train last={tr_rv[-1]:.4f}")
    if vl_s1:
        ax1.plot(vl_s1, vl_rv, color=c_rv_vl, linewidth=1.8,
                 linestyle="--", label="Val residual_var")
        ax1.axhline(y=vl_rv[-1], color=c_rv_vl, linewidth=0.8, linestyle="-.",
                    alpha=0.55, label=f"val last={vl_rv[-1]:.4f}")
    ax1.tick_params(axis="y", labelcolor=c_rv_tr)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Arg I — Residual Variance (text floor)", fontsize=11)

    # ── Panel 2: Arg II — Cosine Sim + λ_min ratio ────────────────────────
    ax2 = axes[2]
    color_cos = "#2CA02C"
    color_lam = "#9467BD"

    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylabel("Mean Pairwise Cosine Similarity", color=color_cos, fontsize=10)
    if tr_s2:
        ax2.plot(tr_s2, tr_cos, color=color_cos, linewidth=1.8,
                 label="Train cos_μ")
    if vl_s2:
        ax2.plot(vl_s2, vl_cos, color=color_cos, linewidth=1.8,
                 linestyle="--", label="Val cos_μ")
    ax2.tick_params(axis="y", labelcolor=color_cos)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(y=1.0, color=color_cos, linewidth=0.8, linestyle=":", alpha=0.5)

    ax2r = ax2.twinx()
    ax2r.set_ylabel("λ_min / λ_max  (collapse ratio)", color=color_lam, fontsize=10)
    if tr_s2:
        ax2r.plot(tr_s2, tr_lam, color=color_lam, linewidth=1.8,
                  linestyle="-", label="Train λ_min ratio")
    if vl_s2:
        ax2r.plot(vl_s2, vl_lam, color=color_lam, linewidth=1.8,
                  linestyle="--", label="Val λ_min ratio")
    ax2r.tick_params(axis="y", labelcolor=color_lam)
    ax2r.set_ylim(0, None)

    lines2, lbls2   = ax2.get_legend_handles_labels()
    lines2r, lbls2r = ax2r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, lbls2 + lbls2r, loc="upper right", fontsize=8)
    ax2.set_title("Argument II — Collapse Indicators", fontsize=11)

    # ── Panel 3: Participation Ratio (train vs val) ────────────────────────
    ax3 = axes[3]
    c_prat = "#17BECF"

    ax3.set_xlabel("Training steps", fontsize=11)
    ax3.set_ylabel("Participation Ratio", color=c_prat, fontsize=10)
    if tr_s3:
        ax3.plot(tr_s3, tr_prat, color=c_prat, linewidth=1.8,
                 label="Train participation_ratio")
    if vl_s3:
        ax3.plot(vl_s3, vl_prat, color=c_prat, linewidth=1.8,
                 linestyle="--", label="Val participation_ratio")
    ax3.tick_params(axis="y", labelcolor=c_prat)
    ax3.set_ylim(0, 1.05)
    ax3.axhline(y=1.0, color=c_prat, linewidth=0.8, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.set_title("Participation Ratio (Train vs Val)", fontsize=11)

    current_step = tr_steps[-1] if tr_steps else 0
    fig.suptitle(
        f"T-JEPA Training Dynamics  [step {current_step}]  "
        f"— solid=train  dashed=val",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    tmp = PNG_PATH.with_suffix(".tmp.png")
    fig.savefig(str(tmp), dpi=150)
    plt.close(fig)
    tmp.replace(PNG_PATH)


def save_records(
    train_records: list[dict],
    val_records:   list[dict],
) -> None:
    """Train JSON → Val JSON → PNG, strict order, synchronous."""
    _write_json_atomic(train_records, JSON_PATH)
    _write_json_atomic(val_records,   JSON_VAL_PATH)
    try:
        _write_plot_atomic(train_records, val_records)
    except OSError as e:
        log.warning(f"_write_plot_atomic failed: {e} — PNG skipped this step")


def save_checkpoint(
    path, model, optimiser, epoch, global_step,
    train_records, val_records,
    lr_sched, wd_sched, ema_sched,
    queue: "MoCoQueue",
) -> None:
    ckpt = dict(
        epoch                = epoch,
        global_step          = global_step,
        model_state_dict     = model.state_dict(),
        optimiser_state_dict = optimiser.state_dict(),
        train_records        = train_records,
        val_records          = val_records,
        config               = CFG,
        lr_sched             = lr_sched,
        wd_sched             = wd_sched,
        ema_sched            = ema_sched,
        moco_queue_buffer    = queue.buffer.cpu(),
        moco_queue_ptr       = queue.ptr,
        moco_queue_full      = queue.full,
    )
    tmp_path = path.with_suffix(".tmp.pt")
    torch.save(ckpt, tmp_path)
    tmp_path.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device(CFG["device"])
    log.info(f"Device: {device}")

    # ── Training DataLoader ────────────────────────────────────────────────
    train_loader, _ = make_c4_dataloader(
        data_dir           = CFG["data_dir"],
        split              = "train",
        batch_size         = CFG["batch_size"],
        num_workers        = CFG["num_workers"],
        pin_mem            = CFG["pin_mem"],
        max_length         = CFG["max_length"],
        max_span_length    = CFG["max_span_length"],
        max_num_spans      = CFG["max_num_spans"],
        min_num_spans      = CFG["min_num_spans"],
        seed               = 42,
        drop_last          = True,
        persistent_workers = False,
    )
    log.info(
        f"Train dataset: {len(train_loader.dataset):,} sentences, "
        f"{len(train_loader):,} batches/epoch"
    )

    # ── Validation DataLoader (held-out eval) ─────────────────────────────
    # Uses the val split — data the model NEVER trains on.
    # drop_last=False to use every val sample.
    val_loader, _ = make_c4_dataloader(
        data_dir           = CFG["data_dir"],
        split              = "val",
        batch_size         = CFG["batch_size"],
        num_workers        = CFG["num_workers"],
        pin_mem            = CFG["pin_mem"],
        max_length         = CFG["max_length"],
        max_span_length    = CFG["max_span_length"],
        max_num_spans      = CFG["max_num_spans"],
        min_num_spans      = CFG["min_num_spans"],
        seed               = 0,
        drop_last          = False,
        persistent_workers = False,
    )
    log.info(
        f"Val dataset:   {len(val_loader.dataset):,} sentences, "
        f"{len(val_loader):,} batches"
    )

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

    # ── Initialise MoCo queue ──────────────────────────────────────────────
    moco_queue = MoCoQueue(
        queue_size = CFG["moco_queue_size"],
        embed_dim  = CFG["hidden_dim"],
        device     = device,
    )
    log.info(
        f"MoCo queue initialised — size={CFG['moco_queue_size']}, "
        f"embed_dim={CFG['hidden_dim']}"
    )

    trainable_params = (list(model.context_encoder.parameters()) +
                        list(model.predictor.parameters()))
    optimiser = torch.optim.AdamW(
        trainable_params, lr=CFG["start_lr"], weight_decay=CFG["weight_decay"]
    )

    steps_per_epoch = len(train_loader)
    total_steps     = CFG["epochs"] * steps_per_epoch
    lr_sched, wd_sched, ema_sched = get_lr_wd_ema_schedulers(total_steps, steps_per_epoch)

    train_records = []
    val_records   = []
    global_step   = 0

    epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Epochs", unit="ep", position=0)

    for epoch in epoch_bar:
        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()
        epoch_losses = []

        iter_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader), desc=f"Ep {epoch:03d}", unit="it",
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

            with torch.amp.autocast(
                device_type="cuda",
                enabled=CFG["use_bfloat16"] and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                out  = model(batch)
                loss = out["span_loss"]

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            model.update_target_encoder(decay=current_ema)

            # ── Update MoCo queue (AFTER optimiser + EMA update) ──────────
            with torch.no_grad():
                if "z_target" in out:
                    z_tgt_raw = out["z_target"].detach()
                    z_enq = z_tgt_raw.mean(dim=1) if z_tgt_raw.dim() == 3 else z_tgt_raw
                else:
                    z_full = model.encode_full_sequence(batch, use_target=True)
                    z_enq  = z_full.detach().mean(dim=1)
            moco_queue.enqueue(z_enq)

            epoch_losses.append(loss.item())
            iter_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.5f}",
                q=f"{len(moco_queue)}",
                step=global_step,
            )

            # ── Training-time metric logging ───────────────────────────────
            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    z_full = model.encode_full_sequence(batch, use_target=False)
                    z_flat = z_full.detach().reshape(-1, CFG["hidden_dim"])
                    rank_info = compute_effective_rank(z_flat, embed_dim=CFG["hidden_dim"])

                record = dict(
                    global_step         = global_step,
                    epoch               = epoch,
                    iter                = it,
                    split               = "train",
                    loss                = round(loss.item(), 6),
                    effective_rank      = round(rank_info["effective_rank"],      4),
                    normalized_rank     = round(rank_info["normalized_rank"],     6),
                    participation_ratio = round(rank_info["participation_ratio"], 4),
                    embed_dim           = CFG["hidden_dim"],
                    model               = "T-JEPA",
                    moco_queue_len      = len(moco_queue),
                )

                # Arg I: Residual Variance (span positions only)
                if global_step % CFG["arg1_every"] == 0:
                    with torch.no_grad():
                        span_mask_bool = batch["span_mask"].bool()
                        z_pred_span = out["predicted_hidden"].detach()[span_mask_bool]
                        z_tgt_span  = out["target_hidden"].detach()[span_mask_bool]
                        res_var = compute_residual_variance(z_pred_span, z_tgt_span)
                    record["residual_var"] = res_var

                # Arg II: collapse metrics
                if global_step % CFG["arg2_every"] == 0:
                    arg2 = compute_arg2_metrics(z_flat, embed_dim=CFG["hidden_dim"])
                    record.update(arg2)
                    log.info(
                        f"[TRAIN ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"res_var={record.get('residual_var', float('nan')):.6f}  "
                        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
                        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
                        f"cos_p95={arg2['cosine_sim_p95']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"lr={current_lr:.6f}"
                    )
                else:
                    log.info(
                        f"[TRAIN ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"res_var={record.get('residual_var', float('nan')):.6f}  "
                        f"q={len(moco_queue)}  "
                        f"lr={current_lr:.6f}"
                    )

                train_records.append(record)

            # ── Held-out evaluation every eval_every steps ─────────────────
            # Why here: we want to track how representation quality on UNSEEN
            # data evolves alongside training-set metrics. The gap between
            # train and val curves is direct evidence of overfitting (or lack
            # thereof) in the representation space.
            if global_step % CFG["eval_every"] == 0:
                log.info(
                    f"  → Running held-out eval at step {global_step} ..."
                )
                val_record = run_held_out_eval(
                    model, val_loader, device, global_step, epoch
                )
                val_records.append(val_record)

            # Save after both train record and potential val record are appended
            if global_step % CFG["log_every"] == 0 or global_step % CFG["eval_every"] == 0:
                save_records(train_records, val_records)

            global_step += 1

        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {epoch:03d} complete  mean_loss={mean_ep_loss:.4f}")
        epoch_bar.set_postfix(mean_loss=f"{mean_ep_loss:.4f}")

        log.info("Saving records (Train JSON → Val JSON → PNG) …")
        save_records(train_records, val_records)

        log.info("Saving checkpoint …")
        save_checkpoint(
            CKPT_PATH, model, optimiser, epoch, global_step,
            train_records, val_records,
            lr_sched, wd_sched, ema_sched,
            moco_queue,
        )
        log.info(f"Checkpoint saved → {CKPT_PATH}")

    log.info(
        f"Records saved → {JSON_PATH} ({len(train_records)} train entries)  "
        f"{JSON_VAL_PATH} ({len(val_records)} val entries)"
    )
    log.info("Training complete.")
    return train_records, val_records


if __name__ == "__main__":
    train()