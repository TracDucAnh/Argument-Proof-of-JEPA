# ijepa_training.py  (Arg-I-II edition — stable EMA + full metric suite + MoCo Queue + Held-Out Eval)
# ─────────────────────────────────────────────────────────────────────────────
# Trains I-JEPA for N epochs on local ImageNet-1K subset.
#
# CHANGES vs. previous edition:
#   STABILITY FIXES (mirrored in T-JEPA for fair comparison):
#     • ema_range = (0.996, 0.996)  — constant tau, eliminates late-training spike
#     • warmup = 10 epochs          — longer warmup, smoother LR ramp
#     • grad_clip = 0.3             — tighter clipping, prevents loss explosion
#     • epochs = 15
#
#   MOCO QUEUE (carried over):
#     • moco_queue_size = 2048      — circular FIFO queue of target representations
#     • Queue stores z_T keys from the target encoder (detached, L2-normalised)
#     • InfoNCE MI proxy now uses the queue as extra negatives:
#         logits = [pos_pair] + [queue_negatives]  → richer contrastive signal
#     • Queue is updated AFTER each optimiser step (enqueue current batch, dequeue oldest)
#     • No gradient flows through queue entries (momentum-encoder keys only)
#
#   ARGUMENT I METRICS (carried over):
#     • mi_proxy       — InfoNCE lower bound on I(z_C; z_T), primary Arg I metric
#                        (now computed with queue negatives when queue is warm)
#     • residual_var   — Var(ẑ_T − z_T), irreducible variance proxy (Prop 4.12)
#
#   ARGUMENT II METRICS (carried over):
#     • lambda_min, lambda_min_ratio, cosine_sim_mean/std/p95/hist
#
#   HELD-OUT EVALUATION (new — mirrors T-JEPA implementation):
#     • Every eval_every steps a full pass over the val DataLoader is run.
#     • The context_encoder, predictor and target_encoder are set to eval() during
#       this pass; NO gradients, NO parameter updates, NO batch-norm stat changes.
#     • ALL metrics computed on held-out data: loss, effective_rank, normalized_rank,
#       participation_ratio, mi_proxy, residual_var, lambda_min, lambda_min_ratio,
#       cosine_sim_mean, cosine_sim_std, cosine_sim_p95, cosine_sim_hist.
#     • Records saved to a SEPARATE JSON: ../Arg-I-II/I-JEPA_val.json
#     • Plots show BOTH train (solid) and val (dashed) curves on the same axes.
#     • Why held-out is the right choice:
#         – Training metrics are computed on batches that just updated the gradient,
#           creating a confirmation bias — the model has already "seen" those patches.
#         – Held-out metrics reflect genuine generalisation of the representation,
#           which is the quantity of interest for Argument I/II claims.
#         – Reviewer standard: any empirical claim about representation quality
#           (rank, collapse, irreducible variance) must be supported by out-of-
#           distribution evidence, not in-sample statistics.
#
#   OUTPUT: ../Arg-I-II/I-JEPA.{json,png,pt}  and  ../Arg-I-II/I-JEPA_val.json
#
#   FAIR COMPARISON:
#     • Rank computed from forward_all_patches (all patches, no masking)
#     • Identical hyperparams to T-JEPA (ema, warmup, clip, epochs, lr)
#
# I/O ORDERING GUARANTEE:
#   All disk writes go through save_records() → save_checkpoint() in strict order:
#     1. Train JSON  (atomic)
#     2. Val   JSON  (atomic)
#     3. PNG         (atomic)
#     4. Checkpoint  (atomic, epoch-end only)
#
# Usage (run from I-JEPA/ directory):
#   python ijepa_training.py
# ─────────────────────────────────────────────────────────────────────────────

import copy
import json
import math
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

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()   # .../I-JEPA/
PROJECT_DIR  = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_II_DIR = PROJECT_DIR / "Arg-I-II"
ARG_I_II_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH     = ARG_I_II_DIR / "I-JEPA.json"
JSON_VAL_PATH = ARG_I_II_DIR / "I-JEPA_val.json"
PNG_PATH      = ARG_I_II_DIR / "I-JEPA.png"
CKPT_PATH     = ARG_I_II_DIR / "I-JEPA_latest.pt"

sys.path.insert(0, str(SCRIPT_DIR))
from ijepa_architecture import (
    vit_huge, vit_predictor, compute_effective_rank
)
from ijepa_dataloader import make_imagenet1k_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # data
    data_dir        = SCRIPT_DIR / "data",
    batch_size      = 64,
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
    # model
    model_name      = "vit_huge",
    embed_dim       = 1280,        # ViT-H
    pred_embed_dim  = 384,
    pred_depth      = 12,
    pred_heads      = 16,
    use_bfloat16    = True,
    # ── STABILITY PARAMS (mirrored with T-JEPA) ───────────────────────────
    epochs          = 15,
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 10,          # 10 epochs warmup (was 40 steps — now epoch-based)
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 0.996),  # FIXED tau — eliminates late-spike artefact
    grad_clip       = 0.3,         # tighter than default 1.0
    # ── MOCO QUEUE ────────────────────────────────────────────────────────
    moco_queue_size = 2048,        # number of keys stored in the MoCo queue
    # ── LOGGING ───────────────────────────────────────────────────────────
    log_every       = 10,
    # ── HELD-OUT EVAL ─────────────────────────────────────────────────────
    # Every eval_every steps: full val-split pass, all metrics, separate JSON
    eval_every      = 400,
    eval_max_batches = None,   # None = full val set; set int to cap (faster debug)
    # Arg I metrics cadence (same as log_every by default)
    arg1_every      = 10,
    mi_temperature  = 0.1,         # InfoNCE temperature τ
    # Arg II metrics cadence
    arg2_every      = 10,
    arg2_sample_size = 2048,
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
    Circular FIFO queue that stores L2-normalised target-encoder keys.

    Design:
        • Fixed-size buffer of shape [queue_size, embed_dim] on `device`.
        • ptr tracks the next write position (wraps around modulo queue_size).
        • Keys are stored as float32 regardless of training dtype to keep
          the InfoNCE computation numerically stable.
        • No gradients flow through the queue (all ops are @torch.no_grad).

    Typical usage inside the training loop:
        # After optimiser.step() and ema_update():
        queue.enqueue(z_tgt_pool)          # z_tgt_pool: [B, D], already detached
        # Inside compute_infonce_mi():
        negatives = queue.get_keys()       # [K, D] — used as extra negatives
    """

    def __init__(self, queue_size: int, embed_dim: int, device: torch.device):
        self.queue_size = queue_size
        self.embed_dim  = embed_dim
        self.device     = device

        # Buffer initialised with random unit vectors so the queue is usable
        # from step 0 (avoids NaN in early InfoNCE before queue is fully warm).
        buf = torch.randn(queue_size, embed_dim, device=device)
        self.buffer = F.normalize(buf, p=2, dim=1)   # [K, D]
        self.ptr    = 0                               # next write position
        self.full   = False                           # True once ptr has wrapped

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """
        Enqueue a batch of keys into the circular buffer.

        Args:
            keys: [B, D] float tensor (need not be normalised; will be L2-normed here)

        Notes:
            - If B > queue_size the enqueue silently wraps; in practice B << queue_size.
            - keys are cast to float32 before storage.
        """
        keys = F.normalize(keys.detach().float(), p=2, dim=1)  # [B, D]
        B    = keys.shape[0]

        end_ptr = self.ptr + B
        if end_ptr <= self.queue_size:
            self.buffer[self.ptr:end_ptr] = keys
        else:
            # Split across wrap boundary
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
            [K, D] float32 tensor where K = queue_size (if full) or ptr (if not yet full).
        """
        if self.full:
            return self.buffer.clone()          # [queue_size, D]
        else:
            return self.buffer[:self.ptr].clone()  # [ptr, D]  (may be empty at step 0)

    def __len__(self) -> int:
        return self.queue_size if self.full else self.ptr


# ─────────────────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_wd_ema_schedulers(total_steps, steps_per_epoch):
    # warmup is now epoch-based for stability
    warmup_steps = CFG["warmup"] * steps_per_epoch
    lr_schedule  = np.zeros(total_steps)
    wd_schedule  = np.zeros(total_steps)
    ema_schedule = np.zeros(total_steps)

    for step in range(total_steps):
        # LR: linear warmup → cosine decay
        if step < warmup_steps:
            lr_schedule[step] = (
                CFG["start_lr"]
                + step * (CFG["lr"] - CFG["start_lr"]) / max(1, warmup_steps)
            )
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = (
                CFG["final_lr"]
                + 0.5 * (CFG["lr"] - CFG["final_lr"])
                * (1 + math.cos(math.pi * progress))
            )

        # Weight decay: cosine ramp up
        progress = step / total_steps
        wd_schedule[step] = (
            CFG["weight_decay"]
            + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"])
            * (1 - math.cos(math.pi * progress))
        )

        # EMA: FIXED tau (ema_range[0] == ema_range[1])
        # Keeping cosine formula for compatibility, but with equal bounds
        # → tau is constant = ema_range[0] throughout training
        ema_schedule[step] = (
            CFG["ema_range"][1]
            - 0.5 * (CFG["ema_range"][1] - CFG["ema_range"][0])
            * (1 + math.cos(math.pi * progress))
        )

    return lr_schedule, wd_schedule, ema_schedule


# ─────────────────────────────────────────────────────────────────────────────
# EMA update
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


# ─────────────────────────────────────────────────────────────────────────────
# Argument I metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_infonce_mi(
    z_ctx: torch.Tensor,
    z_tgt: torch.Tensor,
    temperature: float = 0.1,
    queue: "MoCoQueue | None" = None,
) -> float:
    """
    InfoNCE lower bound on I(z_C; z_T), optionally augmented with MoCo queue negatives.

    Without queue (queue=None or empty):
        Standard in-batch InfoNCE:
            I >= log(N) - CrossEntropy(sim_matrix / τ, diag_labels)
        where N = batch size.

    With queue (queue provided and non-empty):
        Logits are extended with queue keys as extra negatives:
            logits_i = [ z_Ci·z_Ti/τ  (positive),
                         z_Ci·q_k/τ   for k in queue_keys (negatives) ]
        This gives:
            N_eff = 1 + |queue|
            I >= log(N_eff) - CrossEntropy(extended_logits, zeros_label)
        The log(N_eff) term reflects the larger effective dictionary,
        providing a tighter bound once the queue is warm.

    Args:
        z_ctx:       [B, D] context representations (pooled over patches)
        z_tgt:       [B, D] paired target representations
        temperature: τ (default 0.1)
        queue:       MoCoQueue instance or None.
                     When provided its get_keys() are appended as extra negatives.

    Returns:
        float ≥ 0: InfoNCE lower bound (higher = more MI)

    Notes:
        - L2-normalize before dot product → cosine similarity logits
        - Cast to float32 to avoid bfloat16 precision issues
        - Result clamped to 0 to handle early-training numerical noise
    """
    z_c = F.normalize(z_ctx.float(), p=2, dim=1)   # [B, D]
    z_t = F.normalize(z_tgt.float(), p=2, dim=1)   # [B, D]

    queue_keys = queue.get_keys() if (queue is not None and len(queue) > 0) else None

    if queue_keys is None or queue_keys.shape[0] == 0:
        # ── Standard in-batch InfoNCE ──────────────────────────────────────
        logits = z_c @ z_t.T / temperature              # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        ce     = F.cross_entropy(logits, labels)
        bound  = math.log(logits.shape[0]) - ce.item()
    else:
        # ── Queue-augmented InfoNCE ────────────────────────────────────────
        # Positive scores: dot product of each query with its own key → [B]
        pos_scores = (z_c * z_t).sum(dim=1, keepdim=True) / temperature  # [B, 1]

        # Negative scores: each query against all queue keys → [B, K]
        queue_keys = queue_keys.to(z_c.device)                            # [K, D]
        neg_scores = z_c @ queue_keys.T / temperature                     # [B, K]

        # Concatenate: column 0 = positive, columns 1..K = negatives → [B, 1+K]
        logits = torch.cat([pos_scores, neg_scores], dim=1)               # [B, 1+K]
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)                        # positive at index 0

        ce     = F.cross_entropy(logits, labels)
        n_eff  = 1 + queue_keys.shape[0]
        bound  = math.log(n_eff) - ce.item()

    return max(0.0, round(bound, 6))


@torch.no_grad()
def compute_residual_variance(
    z_pred: torch.Tensor,
    z_tgt:  torch.Tensor,
) -> float:
    """
    Residual variance = Var(ẑ_T − z_T).

    This is the empirical proxy for the irreducible variance term in
    Proposition 4.12 / the bias-variance decomposition (Section 4.4).

    For images: predictor improves over time → residual_var decreases.
    For text:   lexical ambiguity creates a floor → residual_var plateaus.

    Formula:
        residual   = ẑ_T - z_T                       [N, D]
        residual_var = E[||residual - E[residual]||²]
                     = mean over N of sum over D of centered² entries

    Args:
        z_pred: [N, D] predictor output ẑ_T (detached)
        z_tgt:  [N, D] target encoder output z_T (detached)

    Returns:
        float: scalar total variance of the residual (float32)
    """
    residual  = (z_pred - z_tgt).float()             # [N, D]
    mean_res  = residual.mean(dim=0, keepdim=True)   # [1, D]
    centered  = residual - mean_res                  # [N, D]
    var       = (centered ** 2).mean().item()
    return round(var, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Argument II metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_arg2_metrics(z_flat: torch.Tensor, embed_dim: int) -> dict:
    z = z_flat.double()  # float64 ngay từ đầu, trước khi tính cov

    mean_z     = z.mean(dim=0, keepdim=True)
    z_centered = z - mean_z
    cov        = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)

    try:
        cov_sym          = (cov + cov.T) * 0.5
        eigvals          = torch.linalg.eigvalsh(cov_sym)   # đã double, không cần cast
        lambda_min       = eigvals[0].item()
        lambda_max       = eigvals[-1].item()
        lambda_min_ratio = (lambda_min / lambda_max) if lambda_max > 1e-12 else 0.0
    except Exception:
        lambda_min       = float("nan")
        lambda_min_ratio = float("nan")

    N = z.shape[0]
    sample_size = min(N, CFG["arg2_sample_size"])
    if sample_size < N:
        idx      = torch.randperm(N, device=z.device)[:sample_size]
        z_sample = z[idx]
    else:
        z_sample = z

    z_norm  = F.normalize(z_sample.float(), p=2, dim=1)  # float32 đủ cho cosine sim
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
    context_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    val_loader,
    device: torch.device,
    global_step: int,
    epoch: int,
    moco_queue: "MoCoQueue",
) -> dict:
    """
    Run a full pass over the validation DataLoader and compute ALL metrics.

    Why this is the correct approach:
        • No gradient update has touched these samples → zero confirmation bias.
        • Aggregating over the entire val split (or eval_max_batches batches)
          gives low-variance estimates suitable for Argument I/II claims.
        • The representation quality we care about — whether I-JEPA collapses,
          whether residual_var decreases — should hold on unseen data, not just
          on the batch that just trained the model.

    Metrics computed (identical formulae to the training-time versions):
        loss, effective_rank, normalized_rank, participation_ratio,
        mi_proxy, residual_var, lambda_min, lambda_min_ratio,
        cosine_sim_mean, cosine_sim_std, cosine_sim_p95, cosine_sim_hist

    All intermediate tensors are accumulated across batches before the final
    eigenspectrum / cosine-sim computation, so the result reflects the full
    distribution of val representations — not just one mini-batch.

    The MoCo queue is passed through (read-only during eval) so that the
    queue-augmented InfoNCE bound is computed identically to training-time.
    NO keys are enqueued during the eval pass.

    Args:
        context_encoder: online context encoder (ViT-H)
        predictor:       predictor transformer
        target_encoder:  EMA target encoder (no grad)
        val_loader:      DataLoader over the val split
        device:          torch.device
        global_step:     current training step (for record labelling)
        epoch:           current epoch
        moco_queue:      MoCoQueue (read-only during eval)

    Returns:
        dict with all metrics plus global_step, epoch, split="val"
    """
    context_encoder.eval()
    predictor.eval()
    target_encoder.eval()

    total_loss = 0.0
    n_batches  = 0

    # Accumulate representations for rank / Arg-II across all val batches
    z_flat_list    = []   # [B*N, D] per batch → cat for rank & arg2

    # Accumulate for Arg I: pooled context + target reps for MI
    z_ctx_pool_list = []
    z_tgt_pool_list = []

    # Accumulate for residual var
    z_pred_pool_list = []

    max_batches = CFG["eval_max_batches"]
    total_val_batches = len(val_loader) if max_batches is None else min(len(val_loader), max_batches)
    
    val_bar = tqdm(
        enumerate(val_loader), 
        total=total_val_batches, 
        desc=f"Val [Step {global_step}]", 
        unit="it", 
        leave=False, 
        dynamic_ncols=True,
        position=2
    )

    for batch_idx, (imgs, masks_enc, masks_pred) in val_bar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs       = imgs.to(device, non_blocking=True)
        masks_enc  = masks_enc.to(device,  non_blocking=True)
        masks_pred = masks_pred.to(device, non_blocking=True)

        masks_enc_flat  = masks_enc[:, 0, :]
        masks_pred_list = [masks_pred[:, k, :] for k in range(masks_pred.shape[1])]

        with torch.amp.autocast(
            device_type="cuda",
            enabled=CFG["use_bfloat16"] and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            z_target_full = target_encoder(imgs)           # [B, N, D]
            z_targets     = []
            for m in masks_pred_list:
                idx = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                z_targets.append(torch.gather(z_target_full, 1, idx))
            z_target_cat = torch.cat(z_targets, dim=0)    # [B*K, T, D]

            z_ctx  = context_encoder(imgs, masks=masks_enc_flat)
            z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)
            loss   = F.mse_loss(z_pred, z_target_cat)

        total_loss += loss.item()
        n_batches  += 1

        # Full-sequence representations for rank / Arg-II (all patches, no masking)
        z_all  = context_encoder.forward_all_patches(imgs)          # [B, N, D]
        z_flat_list.append(
            z_all.detach().reshape(-1, CFG["embed_dim"]).float().cpu()
        )

        # Pooled reps for MI proxy and residual var
        B = imgs.shape[0]
        z_ctx_pool = z_ctx.detach().float().mean(dim=1)             # [B, D]
        z_ctx_pool_list.append(z_ctx_pool.cpu())

        m0      = masks_pred_list[0]                                # [B, K_pred]
        idx     = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
        z_tgt_blk  = torch.gather(z_target_full.detach(), 1, idx)  # [B, K_pred, D]
        z_tgt_pool = z_tgt_blk.float().mean(dim=1)                 # [B, D]
        z_tgt_pool_list.append(z_tgt_pool.cpu())

        # Predictor output for residual var (first block only)
        z_pred_blk0 = z_pred.detach()[:B].float()                  # [B, K_pred, D]
        z_pred_pool_list.append(z_pred_blk0.mean(dim=1).cpu())     # [B, D]

    # ── Aggregate ──────────────────────────────────────────────────────────
    mean_loss = total_loss / max(n_batches, 1)

    # Rank metrics — computed on the full accumulated z_flat
    z_flat_all = torch.cat(z_flat_list, dim=0).to(device)          # [N_total, D]
    # Subsample if too large to avoid OOM on eigendecomposition
    if z_flat_all.shape[0] > 32768:
        idx        = torch.randperm(z_flat_all.shape[0], device=device)[:32768]
        z_flat_all = z_flat_all[idx]

    rank_info = compute_effective_rank(z_flat_all, embed_dim=CFG["embed_dim"])

    # MI proxy — aggregated over full val set
    z_ctx_all  = torch.cat(z_ctx_pool_list,  dim=0).to(device)     # [N_total, D]
    z_tgt_all  = torch.cat(z_tgt_pool_list,  dim=0).to(device)
    z_pred_all = torch.cat(z_pred_pool_list, dim=0).to(device)

    # Subsample for MI and residual var to avoid OOM
    if z_ctx_all.shape[0] > 4096:
        idx        = torch.randperm(z_ctx_all.shape[0], device=device)[:4096]
        z_ctx_sub  = z_ctx_all[idx]
        z_tgt_sub  = z_tgt_all[idx]
        z_pred_sub = z_pred_all[idx]
    else:
        z_ctx_sub  = z_ctx_all
        z_tgt_sub  = z_tgt_all
        z_pred_sub = z_pred_all

    mi_proxy = compute_infonce_mi(
        z_ctx_sub, z_tgt_sub,
        temperature=CFG["mi_temperature"],
        queue=moco_queue,
    )

    res_var = compute_residual_variance(z_pred_sub, z_tgt_sub)

    # Arg II collapse metrics
    arg2 = compute_arg2_metrics(z_flat_all, embed_dim=CFG["embed_dim"])

    val_record = dict(
        global_step         = global_step,
        epoch               = epoch,
        split               = "val",
        loss                = round(mean_loss,                            6),
        effective_rank      = round(rank_info["effective_rank"],          4),
        normalized_rank     = round(rank_info["normalized_rank"],         6),
        participation_ratio = round(rank_info["participation_ratio"],     4),
        embed_dim           = CFG["embed_dim"],
        model               = "I-JEPA",
        mi_proxy            = mi_proxy,
        residual_var        = res_var,
        **arg2,
    )

    log.info(
        f"  [VAL step {global_step:06d}]  "
        f"loss={mean_loss:.4f}  "
        f"eff_rank={rank_info['effective_rank']:.2f}  "
        f"norm_rank={rank_info['normalized_rank']:.4f}  "
        f"mi={mi_proxy:.4f}  "
        f"res_var={res_var:.6f}  "
        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
        f"cos_p95={arg2['cosine_sim_p95']:.4f}  "
        f"n_batches={n_batches}"
    )

    # Restore training modes
    context_encoder.train()
    predictor.train()
    # target_encoder stays eval (no grad, EMA only) — same as during training

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
    Panel 1 — Arg I: MI proxy + Residual Variance
    Panel 2 — Arg II: Cosine Sim Mean + λ_min ratio
    Panel 3 — MoCo Queue fill (train only) + Participation Ratio (train vs val)
    """

    def _extract(records, key):
        return (
            [r["global_step"] for r in records if key in r],
            [r[key]           for r in records if key in r],
        )

    # ── train traces ──────────────────────────────────────────────────────
    tr_steps,  tr_loss   = _extract(train_records, "loss")
    _,         tr_nrank  = _extract(train_records, "normalized_rank")
    tr_s1,     tr_mi     = _extract(train_records, "mi_proxy")
    _,         tr_rv     = _extract(train_records, "residual_var")
    tr_s2,     tr_cos    = _extract(train_records, "cosine_sim_mean")
    _,         tr_lam    = _extract(train_records, "lambda_min_ratio")
    tr_sq,     tr_qlen   = _extract(train_records, "moco_queue_len")
    tr_s3,     tr_prat   = _extract(train_records, "participation_ratio")

    # ── val traces ────────────────────────────────────────────────────────
    vl_steps,  vl_loss   = _extract(val_records, "loss")
    _,         vl_nrank  = _extract(val_records, "normalized_rank")
    vl_s1,     vl_mi     = _extract(val_records, "mi_proxy")
    _,         vl_rv     = _extract(val_records, "residual_var")
    vl_s2,     vl_cos    = _extract(val_records, "cosine_sim_mean")
    _,         vl_lam    = _extract(val_records, "lambda_min_ratio")
    vl_s3,     vl_prat   = _extract(val_records, "participation_ratio")

    fig, axes = plt.subplots(1, 4, figsize=(28, 5))

    # ── Panel 0: Loss + Normalized Rank ───────────────────────────────────
    ax0 = axes[0]
    c_loss = "#378ADD"
    c_rank = "#D85A30"

    ax0.set_xlabel("Training steps", fontsize=11)
    ax0.set_ylabel("MSE Loss", color=c_loss, fontsize=11)
    if tr_steps:
        ax0.plot(tr_steps, tr_loss, color=c_loss, linewidth=1.6,
                 label="Train loss")
    if vl_steps:
        ax0.plot(vl_steps, vl_loss, color=c_loss, linewidth=1.6,
                 linestyle="--", label="Val loss")
    ax0.tick_params(axis="y", labelcolor=c_loss)

    ax0r = ax0.twinx()
    ax0r.set_ylabel("Normalized Eff. Rank (rank / embed_dim)", color=c_rank, fontsize=10)
    if tr_steps:
        ax0r.plot(tr_steps, tr_nrank, color=c_rank, linewidth=1.6,
                  linestyle="-", label="Train norm.rank")
    if vl_steps:
        ax0r.plot(vl_steps, vl_nrank, color=c_rank, linewidth=1.6,
                  linestyle=":", label="Val norm.rank")
    ax0r.tick_params(axis="y", labelcolor=c_rank)
    ax0r.set_ylim(0, 1)

    lines0, lbls0   = ax0.get_legend_handles_labels()
    lines0r, lbls0r = ax0r.get_legend_handles_labels()
    ax0.legend(lines0 + lines0r, lbls0 + lbls0r, loc="upper right", fontsize=8)
    ax0.set_title("Loss & Effective Rank", fontsize=11)

    # ── Panel 1: Arg I — MI proxy + Residual Variance ─────────────────────
    ax1 = axes[1]
    c_mi = "#1F77B4"
    c_rv = "#FF7F0E"

    ax1.set_xlabel("Training steps", fontsize=11)
    ax1.set_ylabel("InfoNCE MI proxy  I(z_C; z_T)", color=c_mi, fontsize=10)
    if tr_s1:
        ax1.plot(tr_s1, tr_mi, color=c_mi, linewidth=1.6,
                 label="Train MI proxy")
    if vl_s1:
        ax1.plot(vl_s1, vl_mi, color=c_mi, linewidth=1.6,
                 linestyle="--", label="Val MI proxy")
    ax1.tick_params(axis="y", labelcolor=c_mi)
    ax1.axhline(y=0.0, color=c_mi, linewidth=0.6, linestyle=":", alpha=0.35)

    ax1r = ax1.twinx()
    ax1r.set_ylabel("Residual Var  Var(ẑ_T − z_T)", color=c_rv, fontsize=10)
    if tr_s1:
        ax1r.plot(tr_s1, tr_rv, color=c_rv, linewidth=1.6,
                  linestyle="-", label="Train residual var")
    if vl_s1:
        ax1r.plot(vl_s1, vl_rv, color=c_rv, linewidth=1.6,
                  linestyle="--", label="Val residual var")
    ax1r.tick_params(axis="y", labelcolor=c_rv)

    lines1, lbls1   = ax1.get_legend_handles_labels()
    lines1r, lbls1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, lbls1 + lbls1r, loc="upper right", fontsize=8)
    ax1.set_title("Arg I — Entropy Ceiling  (solid=train  dashed=val)", fontsize=11)

    # ── Panel 2: Arg II — Cosine Sim + λ_min ratio ────────────────────────
    ax2 = axes[2]
    c_cos = "#2CA02C"
    c_lam = "#9467BD"

    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylabel("Mean Pairwise Cosine Similarity", color=c_cos, fontsize=10)
    if tr_s2:
        ax2.plot(tr_s2, tr_cos, color=c_cos, linewidth=1.6,
                 label="Train cos_μ")
    if vl_s2:
        ax2.plot(vl_s2, vl_cos, color=c_cos, linewidth=1.6,
                 linestyle="--", label="Val cos_μ")
    ax2.tick_params(axis="y", labelcolor=c_cos)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(y=1.0, color=c_cos, linewidth=0.7, linestyle=":", alpha=0.4)

    ax2r = ax2.twinx()
    ax2r.set_ylabel("λ_min / λ_max  (collapse ratio)", color=c_lam, fontsize=10)
    if tr_s2:
        ax2r.plot(tr_s2, tr_lam, color=c_lam, linewidth=1.6,
                  linestyle="-", label="Train λ_min ratio")
    if vl_s2:
        ax2r.plot(vl_s2, vl_lam, color=c_lam, linewidth=1.6,
                  linestyle="--", label="Val λ_min ratio")
    ax2r.tick_params(axis="y", labelcolor=c_lam)
    ax2r.set_ylim(0, None)

    lines2, lbls2   = ax2.get_legend_handles_labels()
    lines2r, lbls2r = ax2r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, lbls2 + lbls2r, loc="upper right", fontsize=8)
    ax2.set_title("Arg II — Collapse Indicators  (solid=train  dashed=val)", fontsize=11)

    # ── Panel 3: MoCo Queue fill + Participation Ratio (train vs val) ──────
    ax3 = axes[3]
    c_q    = "#E377C2"
    c_prat = "#17BECF"

    ax3.set_xlabel("Training steps", fontsize=11)
    ax3.set_ylabel("MoCo Queue Length", color=c_q, fontsize=10)
    if tr_sq:
        ax3.plot(tr_sq, tr_qlen, color=c_q, linewidth=1.6,
                 label="Queue len (train)")
    ax3.axhline(y=CFG["moco_queue_size"], color=c_q, linewidth=0.7,
                linestyle=":", alpha=0.5, label=f"max={CFG['moco_queue_size']}")
    ax3.tick_params(axis="y", labelcolor=c_q)
    ax3.set_ylim(0, CFG["moco_queue_size"] * 1.05)

    ax3r = ax3.twinx()
    ax3r.set_ylabel("Participation Ratio", color=c_prat, fontsize=10)
    if tr_s3:
        ax3r.plot(tr_s3, tr_prat, color=c_prat, linewidth=1.6,
                  linestyle="-", label="Train participation_ratio")
    if vl_s3:
        ax3r.plot(vl_s3, vl_prat, color=c_prat, linewidth=1.6,
                  linestyle="--", label="Val participation_ratio")
    ax3r.tick_params(axis="y", labelcolor=c_prat)
    ax3r.set_ylim(0, 1.05)
    ax3r.axhline(y=1.0, color=c_prat, linewidth=0.7, linestyle=":", alpha=0.4)

    lines3,  lbls3  = ax3.get_legend_handles_labels()
    lines3r, lbls3r = ax3r.get_legend_handles_labels()
    ax3.legend(lines3 + lines3r, lbls3 + lbls3r, loc="lower right", fontsize=8)
    ax3.set_title("MoCo Queue Fill + Participation Ratio", fontsize=11)

    current_step = tr_steps[-1] if tr_steps else 0
    fig.suptitle(
        f"I-JEPA Training Dynamics  [step {current_step}]  "
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
    path, context_encoder, predictor, target_encoder,
    optimiser, epoch, global_step,
    train_records, val_records,
    lr_sched, wd_sched, ema_sched,
    queue: "MoCoQueue",
) -> None:
    ckpt = dict(
        epoch                 = epoch,
        global_step           = global_step,
        context_encoder_state = context_encoder.state_dict(),
        predictor_state       = predictor.state_dict(),
        target_encoder_state  = target_encoder.state_dict(),
        optimiser_state_dict  = optimiser.state_dict(),
        train_records         = train_records,
        val_records           = val_records,
        config                = CFG,
        lr_sched              = lr_sched,
        wd_sched              = wd_sched,
        ema_sched             = ema_sched,
        # ── MoCo queue state ──────────────────────────────────────────────
        moco_queue_buffer     = queue.buffer.cpu(),
        moco_queue_ptr        = queue.ptr,
        moco_queue_full       = queue.full,
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
    log.info(f"Output dir: {ARG_I_II_DIR}")

    # ── Training DataLoader ────────────────────────────────────────────────
    loader, _ = make_imagenet1k_dataloader(
        data_dir           = CFG["data_dir"],
        split              = "train",
        batch_size         = CFG["batch_size"],
        num_workers        = CFG["num_workers"],
        pin_mem            = CFG["pin_mem"],
        crop_size          = CFG["crop_size"],
        patch_size         = CFG["patch_size"],
        enc_mask_scale     = CFG["enc_mask_scale"],
        pred_mask_scale    = CFG["pred_mask_scale"],
        aspect_ratio       = CFG["aspect_ratio"],
        num_enc_masks      = CFG["num_enc_masks"],
        num_pred_masks     = CFG["num_pred_masks"],
        min_keep           = CFG["min_keep"],
        allow_overlap      = CFG["allow_overlap"],
        use_masking        = True,
        persistent_workers = CFG["num_workers"] > 0,
    )
    log.info(f"Train dataset: {len(loader.dataset):,} images | {len(loader):,} batches/epoch")

    # ── Validation DataLoader (held-out eval) ─────────────────────────────
    # Uses the val split — data the model NEVER trains on.
    # drop_last=False to use every val sample.
    val_loader, _ = make_imagenet1k_dataloader(
        data_dir           = CFG["data_dir"],
        split              = "val",
        batch_size         = CFG["batch_size"],
        num_workers        = CFG["num_workers"],
        pin_mem            = CFG["pin_mem"],
        crop_size          = CFG["crop_size"],
        patch_size         = CFG["patch_size"],
        enc_mask_scale     = CFG["enc_mask_scale"],
        pred_mask_scale    = CFG["pred_mask_scale"],
        aspect_ratio       = CFG["aspect_ratio"],
        num_enc_masks      = CFG["num_enc_masks"],
        num_pred_masks     = CFG["num_pred_masks"],
        min_keep           = CFG["min_keep"],
        allow_overlap      = CFG["allow_overlap"],
        use_masking        = True,
        drop_last          = False,
        persistent_workers = CFG["num_workers"] > 0,
    )
    log.info(f"Val dataset:   {len(val_loader.dataset):,} images | {len(val_loader):,} batches")

    N_PATCHES = (CFG["crop_size"] // CFG["patch_size"]) ** 2

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

    ctx_params  = sum(p.numel() for p in context_encoder.parameters()) / 1e6
    pred_params = sum(p.numel() for p in predictor.parameters())       / 1e6
    log.info(
        f"Params — context_encoder: {ctx_params:.1f}M  "
        f"predictor: {pred_params:.1f}M  "
        f"(target_encoder: EMA copy, τ={CFG['ema_range'][0]}, no grad)"
    )

    # ── Initialise MoCo queue ──────────────────────────────────────────────
    moco_queue = MoCoQueue(
        queue_size = CFG["moco_queue_size"],
        embed_dim  = CFG["embed_dim"],
        device     = device,
    )
    log.info(
        f"MoCo queue initialised — size={CFG['moco_queue_size']}, "
        f"embed_dim={CFG['embed_dim']}"
    )

    params    = list(context_encoder.parameters()) + list(predictor.parameters())
    optimiser = torch.optim.AdamW(
        params, lr=CFG["start_lr"], weight_decay=CFG["weight_decay"]
    )

    steps_per_epoch = len(loader)
    total_steps     = CFG["epochs"] * steps_per_epoch
    lr_sched, wd_sched, ema_sched = get_lr_wd_ema_schedulers(total_steps, steps_per_epoch)

    train_records = []
    val_records   = []
    global_step   = 0

    epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Epochs", unit="ep", position=0)

    for epoch in epoch_bar:
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        epoch_losses = []

        iter_bar = tqdm(
            enumerate(loader, start=1),
            total=len(loader), desc=f"Ep {epoch:03d}", unit="it",
            position=1, leave=False, dynamic_ncols=True,
        )

        for it, (imgs, masks_enc, masks_pred) in iter_bar:
            current_lr  = lr_sched[global_step]
            current_wd  = wd_sched[global_step]
            current_ema = ema_sched[global_step]

            for pg in optimiser.param_groups:
                pg["lr"]           = current_lr
                pg["weight_decay"] = current_wd

            imgs       = imgs.to(device, non_blocking=True)
            masks_enc  = masks_enc.to(device,  non_blocking=True)
            masks_pred = masks_pred.to(device, non_blocking=True)

            masks_enc_flat  = masks_enc[:, 0, :]
            masks_pred_list = [masks_pred[:, k, :] for k in range(masks_pred.shape[1])]

            with torch.amp.autocast(
                device_type="cuda",
                enabled=CFG["use_bfloat16"] and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                with torch.no_grad():
                    z_target_full = target_encoder(imgs)          # [B, N, D]
                    z_targets = []
                    for m in masks_pred_list:
                        idx = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                        z_targets.append(torch.gather(z_target_full, 1, idx))
                    z_target_cat = torch.cat(z_targets, dim=0)    # [B*K, T, D]

                z_ctx  = context_encoder(imgs, masks=masks_enc_flat)
                z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)
                loss   = F.mse_loss(z_pred, z_target_cat)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            # ── Tighter gradient clipping for stability ────────────────────
            torch.nn.utils.clip_grad_norm_(params, max_norm=CFG["grad_clip"])
            optimiser.step()
            ema_update(context_encoder, target_encoder, current_ema)

            # ── Update MoCo queue (AFTER optimiser + EMA update) ──────────
            # Pool first pred-mask target block → [B, D] to enqueue
            with torch.no_grad():
                m0      = masks_pred_list[0]                          # [B, K_pred]
                idx_q   = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                z_enq   = torch.gather(z_target_full.detach(), 1, idx_q)  # [B, K_pred, D]
                z_enq   = z_enq.mean(dim=1)                              # [B, D]
            moco_queue.enqueue(z_enq)

            epoch_losses.append(loss.item())
            iter_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.5f}",
                tau=f"{current_ema:.4f}",
                q=f"{len(moco_queue)}",
                step=global_step,
            )

            # ── Training-time metric logging ───────────────────────────────
            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    # ── Rank: ALL patches, no masking (fair vs T-JEPA) ────
                    z_all  = context_encoder.forward_all_patches(imgs)    # [B, N, D]
                    z_flat = z_all.detach().reshape(-1, CFG["embed_dim"]) # [B*N, D]
                    rank_info = compute_effective_rank(z_flat, embed_dim=CFG["embed_dim"])

                record = dict(
                    global_step         = global_step,
                    epoch               = epoch,
                    iter                = it,
                    split               = "train",
                    loss                = round(loss.item(),                      6),
                    effective_rank      = round(rank_info["effective_rank"],      4),
                    normalized_rank     = round(rank_info["normalized_rank"],     6),
                    participation_ratio = round(rank_info["participation_ratio"], 4),
                    embed_dim           = CFG["embed_dim"],
                    model               = "I-JEPA",
                    ema_tau             = round(float(current_ema),               6),
                    moco_queue_len      = len(moco_queue),
                )

                # ── Argument I metrics ─────────────────────────────────────
                if global_step % CFG["arg1_every"] == 0:
                    with torch.no_grad():
                        # Pool context reps to [B, D] for paired InfoNCE
                        z_ctx_pool = z_ctx.detach().float().mean(dim=1)   # [B, D]

                        # Target: gather first pred-mask block, pool → [B, D]
                        m0  = masks_pred_list[0]                          # [B, K_pred]
                        idx = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                        z_tgt_blk  = torch.gather(
                            z_target_full.detach(), 1, idx
                        )                                                  # [B, K_pred, D]
                        z_tgt_pool = z_tgt_blk.float().mean(dim=1)        # [B, D]

                        # InfoNCE with MoCo queue negatives
                        mi_proxy = compute_infonce_mi(
                            z_ctx_pool, z_tgt_pool,
                            temperature=CFG["mi_temperature"],
                            queue=moco_queue,
                        )

                        # Residual var: compare predictor output vs target
                        # Use first pred-mask block only (consistent pairing)
                        # z_pred shape: [B*K_masks, K_pred_tokens, D]
                        # Slice first K_masks=1 block → [B, K_pred, D]
                        B = imgs.shape[0]
                        z_pred_blk0 = z_pred.detach()[:B].float()         # [B, K_pred, D]
                        res_var = compute_residual_variance(
                            z_pred_blk0.mean(dim=1),                       # [B, D]
                            z_tgt_pool,                                    # [B, D]
                        )

                    record["mi_proxy"]     = mi_proxy
                    record["residual_var"] = res_var

                # ── Argument II metrics ────────────────────────────────────
                if global_step % CFG["arg2_every"] == 0:
                    arg2 = compute_arg2_metrics(z_flat, embed_dim=CFG["embed_dim"])
                    record.update(arg2)
                    log.info(
                        f"[TRAIN ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"mi={record.get('mi_proxy', float('nan')):.4f}  "
                        f"res_var={record.get('residual_var', float('nan')):.4f}  "
                        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
                        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"τ={current_ema:.4f}  lr={current_lr:.6f}"
                    )
                else:
                    log.info(
                        f"[TRAIN ep {epoch:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"τ={current_ema:.4f}  lr={current_lr:.6f}"
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
                    context_encoder, predictor, target_encoder,
                    val_loader, device, global_step, epoch,
                    moco_queue,
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
            CKPT_PATH,
            context_encoder, predictor, target_encoder,
            optimiser, epoch, global_step,
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