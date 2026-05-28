# tjepa_training.py  (fair-comparison edition + Argument II metrics + MoCo Queue)
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
# ARGUMENT II METRICS (added):
#   • lambda_min       — smallest eigenvalue of Σ_z (→0 signals collapse direction)
#   • lambda_min_ratio — lambda_min / lambda_max (relative scale; →0 = collapse)
#   • cosine_sim_mean  — mean pairwise cosine similarity of representations
#   • cosine_sim_std   — std of pairwise cosine similarity
#   • cosine_sim_p95   — 95th percentile (shift toward 1.0 = collapse)
#   • cosine_sim_hist  — 10-bin histogram counts (edges: -1.0 to 1.0, uniform)
#   All Arg-II metrics computed every arg2_every steps (default: same as log_every).
#
# MOCO QUEUE (new):
#   • moco_queue_size = 2048      — circular FIFO queue of target span representations
#   • Queue stores pooled z_T keys from the target encoder (detached, L2-normalised)
#   • InfoNCE (if used for diagnostics) now has access to queue negatives
#   • Queue enqueued AFTER each optimiser + EMA step
#   • moco_queue_len logged every log_every steps; queue state saved in checkpoint
#
# I/O ORDERING GUARANTEE:
#   All disk writes go through save_records() → save_checkpoint() in strict order:
#     1. JSON  (atomic: .tmp.json → replace)
#     2. PNG   (atomic: .tmp.png  → replace)
#     3. Checkpoint (atomic: .tmp.pt → replace, epoch-end only)
#   No two of these steps ever run concurrently; each awaits the previous one.
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

JSON_PATH = ARG_I_DIR / "T-JEPA.json"
PNG_PATH  = ARG_I_DIR / "T-JEPA.png"
CKPT_PATH = ARG_I_DIR / "T-JEPA_latest.pt"

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
    # Argument II metrics — set to same as log_every or higher to save compute
    # cosine sim over a subsample of arg2_sample_size vectors (cap to avoid OOM)
    arg2_every       = 10,
    arg2_sample_size = 2048,
    # ── MOCO QUEUE ────────────────────────────────────────────────────────
    moco_queue_size  = 2048,       # number of keys stored in the MoCo queue
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
        • Keys are stored as float32 regardless of training dtype to keep
          cosine-similarity computations numerically stable.
        • No gradients flow through the queue (all ops are @torch.no_grad).

    Typical usage inside the training loop:
        # After optimiser.step() and EMA update:
        queue.enqueue(z_tgt_pool)          # z_tgt_pool: [B, D], already detached
        # Inspect queue:
        negatives = queue.get_keys()       # [K, D]  — can be used as extra negatives
    """

    def __init__(self, queue_size: int, embed_dim: int, device: torch.device):
        self.queue_size = queue_size
        self.embed_dim  = embed_dim
        self.device     = device

        # Initialise with random unit vectors so the queue is usable from step 0.
        buf = torch.randn(queue_size, embed_dim, device=device)
        self.buffer = F.normalize(buf, p=2, dim=1)   # [K, D]
        self.ptr    = 0
        self.full   = False

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """
        Enqueue a batch of keys into the circular buffer.

        Args:
            keys: [B, D] float tensor (need not be pre-normalised; L2-normed here).

        Notes:
            - If B > queue_size the enqueue silently wraps; in practice B << K.
            - keys are cast to float32 before storage.
        """
        keys = F.normalize(keys.detach().float(), p=2, dim=1)  # [B, D]
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
        """
        Return all valid keys currently stored in the queue.

        Returns:
            [K, D] float32 tensor where K = queue_size (full) or ptr (warming up).
        """
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
# Argument II metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_arg2_metrics(z_flat: torch.Tensor, embed_dim: int) -> dict:
    """
    Compute Argument II collapse metrics from a flat representation matrix.

    Args:
        z_flat:    [N, D] float tensor of representations (already detached)
        embed_dim: D, used for reference only

    Returns dict with:
        lambda_min        — smallest eigenvalue of Σ_z (absolute)
        lambda_min_ratio  — lambda_min / lambda_max (relative; →0 = collapse direction)
        cosine_sim_mean   — mean pairwise cosine similarity
        cosine_sim_std    — std  pairwise cosine similarity
        cosine_sim_p95    — 95th percentile (shift toward 1.0 signals collapse)
        cosine_sim_hist   — list of 10 bin counts (edges: -1.0 to 1.0, uniform)

    Implementation notes:
        • lambda_min: computed from covariance eigenspectrum via torch.linalg.eigvalsh
          (symmetric → real eigenvalues; numerically stable).
        • Cosine sim: computed on a random subsample ≤ arg2_sample_size to avoid
          O(N²) memory blowup. Upper-triangle only (exclude self-similarity).
        • All tensors cast to float32 before linear algebra to avoid bfloat16 issues.
    """
    z = z_flat.float()   # ensure float32 for numerical stability

    # ── λ_min of Σ_z ──────────────────────────────────────────────────────
    mean_z     = z.mean(dim=0, keepdim=True)       # [1, D]
    z_centered = z - mean_z                         # [N, D]
    cov        = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)  # [D, D]
    try:
        eigvals       = torch.linalg.eigvalsh(cov)  # ascending, real
        lambda_min    = eigvals[0].item()
        lambda_max    = eigvals[-1].item()
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

    z_norm  = F.normalize(z_sample, p=2, dim=1)    # [S, D]
    sim_mat = z_norm @ z_norm.T                     # [S, S]
    S       = z_norm.shape[0]
    triu_idx = torch.triu_indices(S, S, offset=1, device=z.device)
    sim_vals = sim_mat[triu_idx[0], triu_idx[1]]   # [S*(S-1)/2]

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
# Atomic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json_atomic(records: list[dict]) -> None:
    tmp = JSON_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(JSON_PATH)


def _write_plot_atomic(records: list[dict]) -> None:
    steps  = [r["global_step"]     for r in records]
    losses = [r["loss"]            for r in records]
    nranks = [r["normalized_rank"] for r in records]

    # Arg II traces (only steps where metrics were computed)
    arg2_steps  = [r["global_step"]      for r in records if "cosine_sim_mean" in r]
    cos_means   = [r["cosine_sim_mean"]  for r in records if "cosine_sim_mean" in r]
    lam_ratios  = [r["lambda_min_ratio"] for r in records if "lambda_min_ratio" in r]

    # Queue fill trace
    q_steps = [r["global_step"]    for r in records if "moco_queue_len" in r]
    q_lens  = [r["moco_queue_len"] for r in records if "moco_queue_len" in r]

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    # ── Left panel: Loss + Normalized Rank ────────────────────────────────
    ax1 = axes[0]
    color_loss = "#D85A30"
    color_rank = "#8B2500"

    ax1.set_xlabel("Training steps", fontsize=12)
    ax1.set_ylabel("MSE Loss (log scale)", color=color_loss, fontsize=12)
    ax1.plot(steps, losses, color=color_loss, linewidth=1.8, label="T-JEPA loss")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax1r = ax1.twinx()
    ax1r.set_ylabel("Normalized Effective Rank (rank / embed_dim)", color=color_rank, fontsize=11)
    ax1r.plot(steps, nranks, color=color_rank, linewidth=1.8, linestyle="--",
              label="T-JEPA norm. eff. rank")
    ax1r.tick_params(axis="y", labelcolor=color_rank)
    ax1r.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.set_title("Loss & Effective Rank", fontsize=12)

    # ── Middle panel: Arg II — Cosine Sim Mean + λ_min ratio ──────────────
    ax2 = axes[1]
    color_cos = "#2CA02C"
    color_lam = "#9467BD"

    ax2.set_xlabel("Training steps", fontsize=12)
    ax2.set_ylabel("Mean Pairwise Cosine Similarity", color=color_cos, fontsize=11)
    if arg2_steps:
        ax2.plot(arg2_steps, cos_means, color=color_cos, linewidth=1.8,
                 label="cosine_sim_mean")
    ax2.tick_params(axis="y", labelcolor=color_cos)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(y=1.0, color=color_cos, linewidth=0.8, linestyle=":", alpha=0.5)

    ax2r = ax2.twinx()
    ax2r.set_ylabel("λ_min / λ_max  (collapse ratio)", color=color_lam, fontsize=11)
    if arg2_steps:
        ax2r.plot(arg2_steps, lam_ratios, color=color_lam, linewidth=1.8,
                  linestyle="--", label="λ_min ratio")
    ax2r.tick_params(axis="y", labelcolor=color_lam)
    ax2r.set_ylim(0, None)

    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2r.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=9)
    ax2.set_title("Argument II — Collapse Indicators", fontsize=12)

    # ── Right panel: MoCo Queue fill progress ─────────────────────────────
    ax3 = axes[2]
    c_q = "#E377C2"
    ax3.set_xlabel("Training steps", fontsize=12)
    ax3.set_ylabel("MoCo Queue Length", color=c_q, fontsize=11)
    if q_steps:
        ax3.plot(q_steps, q_lens, color=c_q, linewidth=1.8, label="queue len")
    ax3.axhline(y=CFG["moco_queue_size"], color=c_q, linewidth=0.8,
                linestyle=":", alpha=0.5, label=f"max={CFG['moco_queue_size']}")
    ax3.tick_params(axis="y", labelcolor=c_q)
    ax3.set_ylim(0, CFG["moco_queue_size"] * 1.05)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_title("MoCo Queue Fill", fontsize=12)

    current_step = steps[-1] if steps else 0
    fig.suptitle(
        f"T-JEPA Training Dynamics  [step {current_step}]",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    tmp = PNG_PATH.with_suffix(".tmp.png")
    fig.savefig(str(tmp), dpi=100)
    plt.close(fig)
    tmp.replace(PNG_PATH)


def save_records(records: list[dict]) -> None:
    """JSON → PNG, strict order, synchronous."""
    _write_json_atomic(records)
    try:
        _write_plot_atomic(records)
    except OSError as e:
        log.warning(f"_write_plot_atomic failed: {e} — PNG skipped this step")


def save_checkpoint(
    path, model, optimiser, epoch, global_step,
    records, lr_sched, wd_sched, ema_sched,
    queue: "MoCoQueue",
) -> None:
    ckpt = dict(
        epoch                = epoch,
        global_step          = global_step,
        model_state_dict     = model.state_dict(),
        optimiser_state_dict = optimiser.state_dict(),
        records              = records,
        config               = CFG,
        lr_sched             = lr_sched,
        wd_sched             = wd_sched,
        ema_sched            = ema_sched,
        # ── MoCo queue state ──────────────────────────────────────────────
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

    loader, _ = make_c4_dataloader(
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
    log.info(f"Dataset: {len(loader.dataset):,} sentences, {len(loader):,} batches/epoch")

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

    steps_per_epoch = len(loader)
    total_steps     = CFG["epochs"] * steps_per_epoch
    lr_sched, wd_sched, ema_sched = get_lr_wd_ema_schedulers(total_steps, steps_per_epoch)

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
            # Enqueue the pooled target span representations from this batch.
            # out["z_target"] is expected to be [B, num_spans, D] or [B*S, L, D].
            # We pool to [B, D] before enqueueing.
            with torch.no_grad():
                if "z_target" in out:
                    z_tgt_raw = out["z_target"].detach()
                    # Handle variable shapes: flatten all but last dim → [N, D]
                    if z_tgt_raw.dim() == 3:
                        # [B, S, D] or [B*S, L, D] — mean-pool middle dim
                        z_enq = z_tgt_raw.mean(dim=1)   # [B or B*S, D]
                    else:
                        z_enq = z_tgt_raw               # already [N, D]
                else:
                    # Fallback: encode full sequence with target encoder, pool → [B, D]
                    z_full = model.encode_full_sequence(batch, use_target=True)  # [B, L, D]
                    z_enq  = z_full.detach().mean(dim=1)                         # [B, D]
            moco_queue.enqueue(z_enq)

            epoch_losses.append(loss.item())
            iter_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.5f}",
                q=f"{len(moco_queue)}",
                step=global_step,
            )

            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    # FAIR: encode FULL clean sequence (all tokens, no span filter)
                    z_full = model.encode_full_sequence(batch, use_target=False)
                    z_flat = z_full.detach().reshape(-1, CFG["hidden_dim"])   # [B*L, D]
                    rank_info = compute_effective_rank(z_flat, embed_dim=CFG["hidden_dim"])

                record = dict(
                    global_step         = global_step,
                    epoch               = epoch,
                    iter                = it,
                    loss                = round(loss.item(), 6),
                    effective_rank      = round(rank_info["effective_rank"],      4),
                    normalized_rank     = round(rank_info["normalized_rank"],     6),
                    participation_ratio = round(rank_info["participation_ratio"], 4),
                    embed_dim           = CFG["hidden_dim"],
                    model               = "T-JEPA",
                    moco_queue_len      = len(moco_queue),
                )

                # ── Argument II metrics ────────────────────────────────────
                if global_step % CFG["arg2_every"] == 0:
                    arg2 = compute_arg2_metrics(z_flat, embed_dim=CFG["hidden_dim"])
                    record.update(arg2)
                    log.info(
                        f"[ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
                        f"cos_sim_mean={arg2['cosine_sim_mean']:.4f}  "
                        f"cos_sim_p95={arg2['cosine_sim_p95']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"lr={current_lr:.6f}"
                    )
                else:
                    log.info(
                        f"[ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"lr={current_lr:.6f}"
                    )

                records.append(record)
                save_records(records)

            global_step += 1

        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {epoch:03d} complete  mean_loss={mean_ep_loss:.4f}")
        epoch_bar.set_postfix(mean_loss=f"{mean_ep_loss:.4f}")

        log.info("Saving records (JSON → PNG) …")
        save_records(records)

        log.info("Saving checkpoint …")
        save_checkpoint(
            CKPT_PATH, model, optimiser, epoch, global_step,
            records, lr_sched, wd_sched, ema_sched,
            moco_queue,
        )
        log.info(f"Checkpoint saved → {CKPT_PATH}")

    log.info(f"Records saved → {JSON_PATH}  ({len(records)} entries)")
    log.info("Training complete.")
    return records


if __name__ == "__main__":
    train()