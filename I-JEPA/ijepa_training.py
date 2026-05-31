# ijepa_training.py  (Arg-I-II-III edition — stable EMA + full metric suite + MoCo Queue + Held-Out Eval)
# ─────────────────────────────────────────────────────────────────────────────
# Trains I-JEPA for N epochs on local ImageNet-1K subset.
#
# CHANGES vs. previous edition:
#   STABILITY FIXES (mirrored in T-JEPA for fair comparison):
#     • ema_range = (0.996, 0.996)  — constant tau
#     • warmup = 10 epochs
#     • grad_clip = 0.3
#     • epochs = 15
#
#   MOCO QUEUE (carried over):
#     • moco_queue_size = 2048
#
#   ARGUMENT I METRICS (carried over):
#     • mi_proxy  — InfoNCE lower bound on I(z_C; z_T)
#
#   ARGUMENT II METRICS (carried over):
#     • lambda_min, lambda_min_ratio, cosine_sim_mean/std/p95/hist
#
#   ARGUMENT III — IRREDUCIBLE VARIANCE:
#     • compute_residual_variance() REMOVED — it measured Var(ẑ_T − z_T),
#       i.e. predictor error, which is confounded by predictor capacity.
#     • compute_arg3_irreducible_variance() added: measures Var(z* | x_C, p_j)
#       by generating K augmented crops of the same target patch region,
#       encoding each through the frozen target encoder, and computing
#       within-context variance across K outputs.
#     • tqdm progress bar during Arg III computation (split label: train/val).
#     • Panel 4 in the plot shows irred_var (train vs val) over training steps.
#     • Logged every arg3_every=500 steps on BOTH train and val splits.
#     • Saved to: I-JEPA_arg3_train.json / I-JEPA_arg3_val.json
#
#   HELD-OUT EVALUATION (carried over):
#     • Every eval_every steps: full val-split pass, all metrics, separate JSON.
#
#   EPOCH/ITER DISPLAY:
#     • epoch and iter (it) are always derived from global_step + steps_per_epoch.
#     • This guarantees correct display even when resuming from checkpoints, and
#       prevents `it` from resetting to 1 every epoch in logs and JSON records.
#     • Formula:
#         epoch_display = (global_step // steps_per_epoch) + 1
#         it_display    = (global_step %  steps_per_epoch) + 1
#
# I/O ORDERING GUARANTEE:
#   Train JSON → Val JSON → PNG → Checkpoint (epoch-end only)
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

JSON_PATH          = ARG_I_II_DIR / "I-JEPA.json"
JSON_VAL_PATH      = ARG_I_II_DIR / "I-JEPA_val.json"
JSON_ARG3_TRAIN    = ARG_I_II_DIR / "I-JEPA_arg3_train.json"
JSON_ARG3_VAL      = ARG_I_II_DIR / "I-JEPA_arg3_val.json"
PNG_PATH           = ARG_I_II_DIR / "I-JEPA.png"
CKPT_PATH          = ARG_I_II_DIR / "I-JEPA_latest.pt"

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
    warmup          = 10,
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 0.996),  # FIXED tau
    grad_clip       = 0.3,
    # ── MOCO QUEUE ────────────────────────────────────────────────────────
    moco_queue_size = 2048,
    # ── LOGGING ───────────────────────────────────────────────────────────
    log_every       = 10,
    # ── HELD-OUT EVAL ─────────────────────────────────────────────────────
    eval_every      = 400,
    eval_max_batches = None,
    # ── ARGUMENT I metrics ────────────────────────────────────────────────
    arg1_every      = 10,
    mi_temperature  = 0.1,
    # ── ARGUMENT II metrics ───────────────────────────────────────────────
    arg2_every      = 10,
    arg2_sample_size = 2048,
    # ── ARGUMENT III metrics ──────────────────────────────────────────────
    arg3_every      = 500,    # every 500 steps, on both train and val
    arg3_K          = 16,     # K augmented completions per context
    arg3_N_ctx      = 200,    # number of contexts to average over
    arg3_aug_sigma  = 0.2,    # Gaussian noise std for augmentation
    device          = "cuda" if torch.cuda.is_available() else "cpu",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch/iter helpers — always derived from global_step, never from loop vars
# ─────────────────────────────────────────────────────────────────────────────

def epoch_of(global_step: int, steps_per_epoch: int) -> int:
    """1-based epoch number derived from global_step."""
    return (global_step // steps_per_epoch) + 1


def iter_of(global_step: int, steps_per_epoch: int) -> int:
    """1-based iteration-within-epoch derived from global_step."""
    return (global_step % steps_per_epoch) + 1


# ─────────────────────────────────────────────────────────────────────────────
# MoCo Queue
# ─────────────────────────────────────────────────────────────────────────────

class MoCoQueue:
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

        progress = step / total_steps
        wd_schedule[step] = (
            CFG["weight_decay"]
            + 0.5 * (CFG["final_weight_decay"] - CFG["weight_decay"])
            * (1 - math.cos(math.pi * progress))
        )

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
    z_c = F.normalize(z_ctx.float(), p=2, dim=1)
    z_t = F.normalize(z_tgt.float(), p=2, dim=1)

    queue_keys = queue.get_keys() if (queue is not None and len(queue) > 0) else None

    if queue_keys is None or queue_keys.shape[0] == 0:
        logits = z_c @ z_t.T / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        ce     = F.cross_entropy(logits, labels)
        bound  = math.log(logits.shape[0]) - ce.item()
    else:
        pos_scores = (z_c * z_t).sum(dim=1, keepdim=True) / temperature
        queue_keys = queue_keys.to(z_c.device)
        neg_scores = z_c @ queue_keys.T / temperature

        logits = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)

        ce     = F.cross_entropy(logits, labels)
        n_eff  = 1 + queue_keys.shape[0]
        bound  = math.log(n_eff) - ce.item()

    return max(0.0, round(bound, 6))


# ─────────────────────────────────────────────────────────────────────────────
# Argument II metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_arg2_metrics(z_flat: torch.Tensor, embed_dim: int) -> dict:
    z = z_flat.double()

    mean_z     = z.mean(dim=0, keepdim=True)
    z_centered = z - mean_z
    cov        = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)

    try:
        cov_sym          = (cov + cov.T) * 0.5
        eigvals          = torch.linalg.eigvalsh(cov_sym)
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

    z_norm   = F.normalize(z_sample.float(), p=2, dim=1)
    sim_mat  = z_norm @ z_norm.T
    S        = z_norm.shape[0]
    triu_idx = torch.triu_indices(S, S, offset=1, device=z.device)
    sim_vals = sim_mat[triu_idx[0], triu_idx[1]]

    sim_cpu = sim_vals.cpu().float().numpy()
    cosine_sim_mean = float(np.mean(sim_cpu))
    cosine_sim_std  = float(np.std(sim_cpu))
    cosine_sim_p95  = float(np.percentile(sim_cpu, 95))

    hist_counts, _ = np.histogram(sim_cpu, bins=10, range=(-1.0, 1.0))
    cosine_sim_hist = hist_counts.tolist()

    return dict(
        lambda_min        = float(f"{lambda_min:.6e}"),
        lambda_min_ratio  = float(f"{lambda_min_ratio:.6e}"),
        cosine_sim_mean   = round(cosine_sim_mean, 6),
        cosine_sim_std    = round(cosine_sim_std,  6),
        cosine_sim_p95    = round(cosine_sim_p95,  6),
        cosine_sim_hist   = cosine_sim_hist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument III — Irreducible Variance  Var(z* | x_C, p_j)
# ─────────────────────────────────────────────────────────────────────────────
#
# PROTOCOL (image):
#   For each context image:
#     1. Identify the first pred-mask block positions p_j.
#     2. Generate K augmented versions of the image:
#          - Additive Gaussian noise: N(0, aug_sigma) in pixel space
#          - Random sparse sign-flip (0.5% of pixels)
#          - Clamp to original range
#     3. Forward each augmented image through frozen target encoder.
#     4. Gather hidden states at positions p_j; pool over patch tokens → [K, D].
#     5. Compute variance across K vectors → scalar per context.
#   Average over N_ctx contexts.
#
#   tqdm progress bar shows contexts processed vs N_ctx, labelled by split.

@torch.no_grad()
def compute_arg3_irreducible_variance(
    target_encoder: torch.nn.Module,
    loader_iter,
    device: torch.device,
    embed_dim: int,
    K: int           = 16,
    N_ctx: int       = 200,
    aug_sigma: float = 0.2,
    split: str       = "train",
) -> dict:
    """
    Estimate Var(z* | x_C, p_j) for images.

    Args:
        target_encoder: frozen EMA ViT-H encoder, kept in .eval()
        loader_iter:    iterator yielding (imgs, masks_enc, masks_pred) tuples
        device:         torch.device
        embed_dim:      D (1280 for ViT-H)
        K:              number of augmented copies per image
        N_ctx:          number of contexts to average over
        aug_sigma:      std of additive Gaussian noise
        split:          "train" or "val" — used in tqdm description

    Returns:
        dict:
            irred_var  — mean Var(z*|x_C,p_j) over N_ctx contexts
            n_contexts — actual contexts processed
    """
    target_encoder.eval()

    all_vars      = []
    contexts_done = 0

    pbar = tqdm(
        total=N_ctx,
        desc=f"  Arg III [{split}]",
        unit="ctx",
        leave=False,
        dynamic_ncols=True,
        position=3,
    )

    while contexts_done < N_ctx:
        try:
            imgs, masks_enc, masks_pred = next(loader_iter)
        except StopIteration:
            break

        imgs       = imgs.to(device, non_blocking=True)        # [B, C, H, W]
        masks_pred = masks_pred.to(device, non_blocking=True)  # [B, num_pred, K_pat]

        B      = imgs.shape[0]
        budget = min(B, N_ctx - contexts_done)
        imgs       = imgs[:budget]
        masks_pred = masks_pred[:budget]

        # Use first pred-mask block to define position p_j
        m0 = masks_pred[:, 0, :]                               # [budget, K_patches]

        # ── Generate K augmented completions per image ─────────────────────
        z_completions = torch.zeros(budget, K, embed_dim, device=device)

        for k in range(K):
            imgs_aug = imgs.clone().float()

            # 1) Additive Gaussian noise
            imgs_aug = imgs_aug + torch.randn_like(imgs_aug) * aug_sigma

            # 2) Random sparse sign-flip (0.5% of pixels)
            flip_mask = torch.rand_like(imgs_aug) < 0.005
            imgs_aug[flip_mask] = -imgs_aug[flip_mask]

            # Clamp to valid range (works for both [0,1] and [-1,1] normalised)
            imgs_aug = imgs_aug.clamp(-1.0, 1.0).to(imgs.dtype)

            # Forward through frozen target encoder → [budget, N_patches, D]
            z_full = target_encoder(imgs_aug)

            # Gather patch positions from pred-mask → [budget, K_patches, D]
            idx     = m0.unsqueeze(-1).expand(-1, -1, embed_dim)
            z_patch = torch.gather(z_full, 1, idx)             # [budget, K_patches, D]

            # Pool over K_patches → [budget, D]
            z_completions[:, k, :] = z_patch.mean(dim=1).float()

        # ── Within-context variance across K augmented copies ──────────────
        z     = z_completions.float()                          # [budget, K, D]
        z_bar = z.mean(dim=1, keepdim=True)                    # [budget, 1, D]
        # sum squared deviations over D, mean over K → [budget]
        var_per_ctx = ((z - z_bar) ** 2).sum(dim=-1).mean(dim=1)
        all_vars.append(var_per_ctx.mean().item())

        contexts_done += budget
        pbar.update(budget)

    pbar.close()

    mean_var = float(np.mean(all_vars)) if all_vars else float("nan")

    return dict(
        irred_var  = round(mean_var, 8),
        n_contexts = contexts_done,
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
    epoch: int,          # pre-derived via epoch_of(); passed in for the record
    moco_queue: "MoCoQueue",
) -> dict:
    context_encoder.eval()
    predictor.eval()
    target_encoder.eval()

    total_loss = 0.0
    n_batches  = 0

    z_flat_list     = []
    z_ctx_pool_list = []
    z_tgt_pool_list = []

    max_batches = CFG["eval_max_batches"]
    total_val_batches = len(val_loader) if max_batches is None else min(len(val_loader), max_batches)

    val_bar = tqdm(
        enumerate(val_loader),
        total=total_val_batches,
        desc=f"Val [Step {global_step}]",
        unit="it",
        leave=False,
        dynamic_ncols=True,
        position=2,
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
            z_target_full = target_encoder(imgs)
            z_targets     = []
            for m in masks_pred_list:
                idx = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                z_targets.append(torch.gather(z_target_full, 1, idx))
            z_target_cat = torch.cat(z_targets, dim=0)

            z_ctx  = context_encoder(imgs, masks=masks_enc_flat)
            z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)
            loss   = F.mse_loss(z_pred, z_target_cat)

        total_loss += loss.item()
        n_batches  += 1

        z_all = context_encoder.forward_all_patches(imgs)
        z_flat_list.append(
            z_all.detach().reshape(-1, CFG["embed_dim"]).float().cpu()
        )

        z_ctx_pool = z_ctx.detach().float().mean(dim=1)
        z_ctx_pool_list.append(z_ctx_pool.cpu())

        m0         = masks_pred_list[0]
        idx        = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
        z_tgt_blk  = torch.gather(z_target_full.detach(), 1, idx)
        z_tgt_pool = z_tgt_blk.float().mean(dim=1)
        z_tgt_pool_list.append(z_tgt_pool.cpu())

    mean_loss = total_loss / max(n_batches, 1)

    z_flat_all = torch.cat(z_flat_list, dim=0).to(device)
    if z_flat_all.shape[0] > 32768:
        idx        = torch.randperm(z_flat_all.shape[0], device=device)[:32768]
        z_flat_all = z_flat_all[idx]

    rank_info = compute_effective_rank(z_flat_all, embed_dim=CFG["embed_dim"])

    z_ctx_all = torch.cat(z_ctx_pool_list, dim=0).to(device)
    z_tgt_all = torch.cat(z_tgt_pool_list, dim=0).to(device)

    if z_ctx_all.shape[0] > 4096:
        idx       = torch.randperm(z_ctx_all.shape[0], device=device)[:4096]
        z_ctx_sub = z_ctx_all[idx]
        z_tgt_sub = z_tgt_all[idx]
    else:
        z_ctx_sub = z_ctx_all
        z_tgt_sub = z_tgt_all

    mi_proxy = compute_infonce_mi(
        z_ctx_sub, z_tgt_sub,
        temperature=CFG["mi_temperature"],
        queue=moco_queue,
    )

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
        **arg2,
    )

    log.info(
        f"  [VAL step {global_step:06d}]  "
        f"loss={mean_loss:.4f}  "
        f"eff_rank={rank_info['effective_rank']:.2f}  "
        f"norm_rank={rank_info['normalized_rank']:.4f}  "
        f"mi={mi_proxy:.4f}  "
        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
        f"cos_p95={arg2['cosine_sim_p95']:.4f}  "
        f"n_batches={n_batches}"
    )

    context_encoder.train()
    predictor.train()

    return val_record


# ─────────────────────────────────────────────────────────────────────────────
# Atomic I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_json_atomic(records: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_plot_atomic(
    train_records:      list[dict],
    val_records:        list[dict],
    arg3_train_records: list[dict],
    arg3_val_records:   list[dict],
) -> None:
    """
    5-panel plot with train (solid) and val (dashed) curves overlaid.

    Panel 0 — Loss + Normalized Rank
    Panel 1 — Arg I: MI proxy
    Panel 2 — Arg II: Cosine Sim Mean + λ_min ratio
    Panel 3 — MoCo Queue fill (train only) + Participation Ratio (train vs val)
    Panel 4 — Arg III: Irreducible Variance (train vs val)
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
    tr_s2,     tr_cos    = _extract(train_records, "cosine_sim_mean")
    _,         tr_lam    = _extract(train_records, "lambda_min_ratio")
    tr_sq,     tr_qlen   = _extract(train_records, "moco_queue_len")
    tr_s3,     tr_prat   = _extract(train_records, "participation_ratio")

    # ── val traces ────────────────────────────────────────────────────────
    vl_steps,  vl_loss   = _extract(val_records, "loss")
    _,         vl_nrank  = _extract(val_records, "normalized_rank")
    vl_s1,     vl_mi     = _extract(val_records, "mi_proxy")
    vl_s2,     vl_cos    = _extract(val_records, "cosine_sim_mean")
    _,         vl_lam    = _extract(val_records, "lambda_min_ratio")
    vl_s3,     vl_prat   = _extract(val_records, "participation_ratio")

    # ── Arg III traces ────────────────────────────────────────────────────
    a3tr_steps, a3tr_var = _extract(arg3_train_records, "irred_var")
    a3vl_steps, a3vl_var = _extract(arg3_val_records,   "irred_var")

    fig, axes = plt.subplots(1, 5, figsize=(35, 5))

    # ── Panel 0: Loss + Normalized Rank ───────────────────────────────────
    ax0    = axes[0]
    c_loss = "#378ADD"
    c_rank = "#D85A30"

    ax0.set_xlabel("Training steps", fontsize=11)
    ax0.set_ylabel("MSE Loss", color=c_loss, fontsize=11)
    if tr_steps:
        ax0.plot(tr_steps, tr_loss, color=c_loss, linewidth=1.6, label="Train loss")
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

    lines0,  lbls0  = ax0.get_legend_handles_labels()
    lines0r, lbls0r = ax0r.get_legend_handles_labels()
    ax0.legend(lines0 + lines0r, lbls0 + lbls0r, loc="upper right", fontsize=8)
    ax0.set_title("Loss & Effective Rank", fontsize=11)

    # ── Panel 1: Arg I — MI proxy ──────────────────────────────────────────
    ax1  = axes[1]
    c_mi = "#1F77B4"

    ax1.set_xlabel("Training steps", fontsize=11)
    ax1.set_ylabel("InfoNCE MI proxy  I(z_C; z_T)", color=c_mi, fontsize=10)
    if tr_s1:
        ax1.plot(tr_s1, tr_mi, color=c_mi, linewidth=1.6, label="Train MI proxy")
    if vl_s1:
        ax1.plot(vl_s1, vl_mi, color=c_mi, linewidth=1.6,
                 linestyle="--", label="Val MI proxy")
    ax1.tick_params(axis="y", labelcolor=c_mi)
    ax1.axhline(y=0.0, color=c_mi, linewidth=0.6, linestyle=":", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Arg I — MI proxy  (solid=train  dashed=val)", fontsize=11)

    # ── Panel 2: Arg II — Cosine Sim + λ_min ratio ────────────────────────
    ax2   = axes[2]
    c_cos = "#2CA02C"
    c_lam = "#9467BD"

    ax2.set_xlabel("Training steps", fontsize=11)
    ax2.set_ylabel("Mean Pairwise Cosine Similarity", color=c_cos, fontsize=10)
    if tr_s2:
        ax2.plot(tr_s2, tr_cos, color=c_cos, linewidth=1.6, label="Train cos_μ")
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

    lines2,  lbls2  = ax2.get_legend_handles_labels()
    lines2r, lbls2r = ax2r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, lbls2 + lbls2r, loc="upper right", fontsize=8)
    ax2.set_title("Arg II — Collapse Indicators  (solid=train  dashed=val)", fontsize=11)

    # ── Panel 3: MoCo Queue fill + Participation Ratio ────────────────────
    ax3    = axes[3]
    c_q    = "#E377C2"
    c_prat = "#17BECF"

    ax3.set_xlabel("Training steps", fontsize=11)
    ax3.set_ylabel("MoCo Queue Length", color=c_q, fontsize=10)
    if tr_sq:
        ax3.plot(tr_sq, tr_qlen, color=c_q, linewidth=1.6, label="Queue len (train)")
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

    # ── Panel 4: Arg III — Irreducible Variance ───────────────────────────
    ax4    = axes[4]
    c_ivar = "#FF7F0E"

    ax4.set_xlabel("Training steps", fontsize=11)
    ax4.set_ylabel("Irred. Variance  Var(z* | x_C, p_j)", color=c_ivar, fontsize=10)

    if a3tr_steps:
        ax4.plot(a3tr_steps, a3tr_var, color=c_ivar, linewidth=1.8,
                 linestyle="-", marker="o", markersize=4, label="Train irred_var")
    if a3vl_steps:
        ax4.plot(a3vl_steps, a3vl_var, color=c_ivar, linewidth=1.8,
                 linestyle="--", marker="s", markersize=4, label="Val irred_var")

    ax4.tick_params(axis="y", labelcolor=c_ivar)
    ax4.set_ylim(bottom=0)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_title(
        "Arg III — Irreducible Variance\n"
        "(K aug crops → frozen ViT-H EMA encoder)",
        fontsize=10,
    )

    # ── Suptitle ──────────────────────────────────────────────────────────
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
    train_records:      list[dict],
    val_records:        list[dict],
    arg3_train_records: list[dict],
    arg3_val_records:   list[dict],
) -> None:
    _write_json_atomic(train_records, JSON_PATH)
    _write_json_atomic(val_records,   JSON_VAL_PATH)
    try:
        _write_plot_atomic(train_records, val_records, arg3_train_records, arg3_val_records)
    except OSError as e:
        log.warning(f"_write_plot_atomic failed: {e} — PNG skipped this step")


def save_checkpoint(
    path, context_encoder, predictor, target_encoder,
    optimiser, epoch, global_step,
    train_records, val_records,
    arg3_train_records, arg3_val_records,
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
        arg3_train_records    = arg3_train_records,
        arg3_val_records      = arg3_val_records,
        config                = CFG,
        lr_sched              = lr_sched,
        wd_sched              = wd_sched,
        ema_sched             = ema_sched,
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

    # ── Log the resolved epoch/step sizes once so mismatches are obvious ──
    log.info(
        f"steps_per_epoch={steps_per_epoch}  "
        f"total_steps={total_steps}  "
        f"epochs={CFG['epochs']}"
    )

    train_records      = []
    val_records        = []
    arg3_train_records = []
    arg3_val_records   = []
    global_step        = 0

    epoch_bar = tqdm(range(1, CFG["epochs"] + 1), desc="Epochs", unit="ep", position=0)

    for _epoch_loop_var in epoch_bar:
        # ── epoch and iter are always derived from global_step ─────────────
        # _epoch_loop_var is only used to drive the outer loop count.
        # All logging, records, and display use epoch_of() / iter_of() so
        # they remain correct after checkpoint resumes and across epoch boundaries.

        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        epoch_losses = []

        ep_display_start = epoch_of(global_step, steps_per_epoch)

        iter_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"Ep {ep_display_start:03d}",
            unit="it",
            position=1,
            leave=True,
            dynamic_ncols=True,
        )

        for imgs, masks_enc, masks_pred in iter_bar:
            # ── Derive epoch and iter from global_step ────────────────────
            ep = epoch_of(global_step, steps_per_epoch)   # 1-based, never resets
            it = iter_of(global_step, steps_per_epoch)    # 1-based within epoch

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
                    z_target_full = target_encoder(imgs)
                    z_targets = []
                    for m in masks_pred_list:
                        idx = m.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                        z_targets.append(torch.gather(z_target_full, 1, idx))
                    z_target_cat = torch.cat(z_targets, dim=0)

                z_ctx  = context_encoder(imgs, masks=masks_enc_flat)
                z_pred = predictor(z_ctx, masks_x=masks_enc_flat, masks=masks_pred_list)
                loss   = F.mse_loss(z_pred, z_target_cat)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=CFG["grad_clip"])
            optimiser.step()
            ema_update(context_encoder, target_encoder, current_ema)

            with torch.no_grad():
                m0    = masks_pred_list[0]
                idx_q = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                z_enq = torch.gather(z_target_full.detach(), 1, idx_q).mean(dim=1)
            moco_queue.enqueue(z_enq)

            epoch_losses.append(loss.item())
            iter_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.5f}",
                tau=f"{current_ema:.4f}",
                q=f"{len(moco_queue)}",
                ep=ep,
                it=it,
                step=global_step,
            )

            if global_step % CFG["log_every"] == 0:
                with torch.no_grad():
                    z_all  = context_encoder.forward_all_patches(imgs)
                    z_flat = z_all.detach().reshape(-1, CFG["embed_dim"])
                    rank_info = compute_effective_rank(z_flat, embed_dim=CFG["embed_dim"])

                record = dict(
                    global_step         = global_step,
                    epoch               = ep,   # derived from global_step
                    iter                = it,   # derived from global_step
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

                if global_step % CFG["arg1_every"] == 0:
                    with torch.no_grad():
                        z_ctx_pool = z_ctx.detach().float().mean(dim=1)
                        m0         = masks_pred_list[0]
                        idx        = m0.unsqueeze(-1).expand(-1, -1, CFG["embed_dim"])
                        z_tgt_blk  = torch.gather(z_target_full.detach(), 1, idx)
                        z_tgt_pool = z_tgt_blk.float().mean(dim=1)

                        mi_proxy = compute_infonce_mi(
                            z_ctx_pool, z_tgt_pool,
                            temperature=CFG["mi_temperature"],
                            queue=moco_queue,
                        )
                    record["mi_proxy"] = mi_proxy

                if global_step % CFG["arg2_every"] == 0:
                    arg2 = compute_arg2_metrics(z_flat, embed_dim=CFG["embed_dim"])
                    record.update(arg2)
                    log.info(
                        f"[TRAIN ep {ep:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"mi={record.get('mi_proxy', float('nan')):.4f}  "
                        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
                        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"τ={current_ema:.4f}  lr={current_lr:.6f}"
                    )
                else:
                    log.info(
                        f"[TRAIN ep {ep:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"q={len(moco_queue)}  "
                        f"τ={current_ema:.4f}  lr={current_lr:.6f}"
                    )

                train_records.append(record)

            if global_step % CFG["eval_every"] == 0:
                log.info(f"  → Running held-out eval at step {global_step} ...")
                val_record = run_held_out_eval(
                    context_encoder, predictor, target_encoder,
                    val_loader, device, global_step,
                    epoch=ep,   # derived from global_step
                    moco_queue=moco_queue,
                )
                val_records.append(val_record)

            # ── Argument III: irreducible variance every arg3_every steps ──
            if global_step % CFG["arg3_every"] == 0:
                log.info(f"  → Computing Arg III irreducible variance at step {global_step} ...")

                # --- TRAIN split ---
                arg3_train = compute_arg3_irreducible_variance(
                    target_encoder = target_encoder,
                    loader_iter    = iter(loader),
                    device         = device,
                    embed_dim      = CFG["embed_dim"],
                    K              = CFG["arg3_K"],
                    N_ctx          = CFG["arg3_N_ctx"],
                    aug_sigma      = CFG["arg3_aug_sigma"],
                    split          = "train",
                )
                arg3_train_records.append(dict(
                    global_step = global_step,
                    epoch       = ep,   # derived from global_step
                    split       = "train",
                    model       = "I-JEPA",
                    irred_var   = arg3_train["irred_var"],
                    n_contexts  = arg3_train["n_contexts"],
                ))

                # --- VAL split ---
                arg3_val = compute_arg3_irreducible_variance(
                    target_encoder = target_encoder,
                    loader_iter    = iter(val_loader),
                    device         = device,
                    embed_dim      = CFG["embed_dim"],
                    K              = CFG["arg3_K"],
                    N_ctx          = CFG["arg3_N_ctx"],
                    aug_sigma      = CFG["arg3_aug_sigma"],
                    split          = "val",
                )
                arg3_val_records.append(dict(
                    global_step = global_step,
                    epoch       = ep,   # derived from global_step
                    split       = "val",
                    model       = "I-JEPA",
                    irred_var   = arg3_val["irred_var"],
                    n_contexts  = arg3_val["n_contexts"],
                ))

                log.info(
                    f"  [ARG III step {global_step:06d}]  "
                    f"irred_var train={arg3_train['irred_var']:.6f}  "
                    f"val={arg3_val['irred_var']:.6f}  "
                    f"(K={CFG['arg3_K']}, N_ctx={CFG['arg3_N_ctx']})"
                )

                _write_json_atomic(arg3_train_records, JSON_ARG3_TRAIN)
                _write_json_atomic(arg3_val_records,   JSON_ARG3_VAL)

            if global_step % CFG["log_every"] == 0 or global_step % CFG["eval_every"] == 0:
                save_records(train_records, val_records, arg3_train_records, arg3_val_records)

            global_step += 1

        # ── End of epoch summary ──────────────────────────────────────────
        finished_ep  = epoch_of(global_step - 1, steps_per_epoch)
        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {finished_ep:03d} complete  mean_loss={mean_ep_loss:.4f}")
        epoch_bar.set_postfix(mean_loss=f"{mean_ep_loss:.4f}", ep=finished_ep)

        log.info("Saving records (Train JSON → Val JSON → PNG) …")
        save_records(train_records, val_records, arg3_train_records, arg3_val_records)

        log.info("Saving checkpoint …")
        save_checkpoint(
            CKPT_PATH,
            context_encoder, predictor, target_encoder,
            optimiser,
            epoch       = finished_ep,   # derived, not loop var
            global_step = global_step,
            train_records      = train_records,
            val_records        = val_records,
            arg3_train_records = arg3_train_records,
            arg3_val_records   = arg3_val_records,
            lr_sched    = lr_sched,
            wd_sched    = wd_sched,
            ema_sched   = ema_sched,
            queue       = moco_queue,
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