# tjepa_training.py  (Arg-I-II-III edition — stable EMA + full metric suite + MoCo Queue + Held-Out Eval)
# ─────────────────────────────────────────────────────────────────────────────
# Trains T-JEPA for N epochs on local C4-subset (BERT-Large settings).
# Logs JEPA loss + FAIR effective rank metrics every log_every iters.
#
# FAIR COMPARISON CHANGES vs. I-JEPA:
#   • Identical hyperparams (ema, warmup, clip, epochs, lr)
#   • Effective rank computed from ALL tokens via encode_full_sequence()
#   • MoCo Queue used for InfoNCE MI proxy
#   • Held-Out Evaluation with full metrics matching I-JEPA
#   • Plotting logic identical (5 panels: Loss/Rank, MI, ArgII, Queue/PR, ArgIII)
#
# ARGUMENT III — IRREDUCIBLE VARIANCE:
#   • Uses a SEPARATE frozen bert-base-uncased (pretrained) as the MLM oracle
#     to obtain p(token | context). This is fully independent of the T-JEPA
#     training loop — weights never change, no gradient flows through it.
#   • K token completions sampled from bert-base MLM distribution at pos j.
#   • Each completion is then encoded through the frozen TARGET ENCODER
#     (BERT-Large EMA) to get z* ∈ R^1024.
#   • Variance computed across K z* vectors → Var(z* | x_C, p_j).
#   • tqdm progress bar during Arg III computation.
#   • Panel 4 in the plot shows irred_var (train vs val) over training steps.
#   • Logged every arg3_every=500 steps on BOTH train and val splits.
#   • Saved to: T-JEPA_arg3_train.json / T-JEPA_arg3_val.json
#
# EPOCH/ITER DISPLAY:
#   • epoch and iter (it) are always derived from global_step + steps_per_epoch.
#   • This guarantees correct display even when resuming from checkpoints, and
#     prevents `it` from resetting to 1 every epoch in logs and JSON records.
#   • Formula:
#       epoch_display = (global_step // steps_per_epoch) + 1
#       it_display    = (global_step %  steps_per_epoch) + 1
#
# I/O ORDERING GUARANTEE:
#   All disk writes go through save_records() → save_checkpoint() in strict order.
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
from transformers import BertForMaskedLM as _BertForMaskedLM

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()   # .../T-JEPA/
PROJECT_DIR = SCRIPT_DIR.parent                  # .../ICLR EMPIRICAL EVIDENCES/
ARG_I_DIR   = PROJECT_DIR / "Arg-I"
ARG_I_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH          = ARG_I_DIR / "T-JEPA.json"
JSON_VAL_PATH      = ARG_I_DIR / "T-JEPA_val.json"
JSON_ARG3_TRAIN    = ARG_I_DIR / "T-JEPA_arg3_train.json"
JSON_ARG3_VAL      = ARG_I_DIR / "T-JEPA_arg3_val.json"
PNG_PATH           = ARG_I_DIR / "T-JEPA.png"
CKPT_PATH          = ARG_I_DIR / "T-JEPA_latest.pt"

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
    # ── STABILITY PARAMS (mirrored with I-JEPA) ───────────────────────────
    epochs          = 15,
    start_lr        = 0.0002,
    lr              = 0.001,
    final_lr        = 1.0e-06,
    warmup          = 10,
    weight_decay    = 0.04,
    final_weight_decay = 0.4,
    ema_range       = (0.996, 0.996),  # FIXED tau
    grad_clip       = 0.3,
    # training
    log_every       = 10,
    # ── HELD-OUT EVAL ─────────────────────────────────────────────────────
    eval_every      = 400,
    eval_max_batches = None,
    # ── ARGUMENT I metrics ────────────────────────────────────────────────
    arg1_every       = 10,
    mi_temperature   = 0.1,
    # ── ARGUMENT II metrics ───────────────────────────────────────────────
    arg2_every       = 10,
    arg2_sample_size = 2048,
    # ── ARGUMENT III metrics ──────────────────────────────────────────────
    arg3_every       = 500,   # every 500 steps, on both train and val
    arg3_K           = 16,    # K token completions per masked position
    arg3_N_ctx       = 200,   # number of contexts to average over
    arg3_temperature = 1.0,   # softmax temperature for token sampling
    arg3_min_prob    = 0.01,  # minimum token probability to be a candidate
    # bert-base-uncased used as frozen MLM oracle for Arg III token sampling
    arg3_ref_model   = "bert-base-uncased",
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

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

    z_norm  = F.normalize(z_sample.float(), p=2, dim=1)
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
        lambda_min        = float(f"{lambda_min:.6e}"),
        lambda_min_ratio  = float(f"{lambda_min_ratio:.6e}"),
        cosine_sim_mean   = round(cosine_sim_mean,  6),
        cosine_sim_std    = round(cosine_sim_std,   6),
        cosine_sim_p95    = round(cosine_sim_p95,   6),
        cosine_sim_hist   = cosine_sim_hist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument III — Irreducible Variance  Var(z* | x_C, p_j)
# ─────────────────────────────────────────────────────────────────────────────
#
# DESIGN:
#   • ref_mlm_model  — frozen bert-base-uncased (pretrained, NEVER updated).
#                      Used ONLY to get p(token | context) at masked position j.
#                      Completely independent of T-JEPA training loop.
#   • target_encoder — frozen BERT-Large EMA encoder.
#                      Used to encode K completed sequences → z* ∈ R^1024.
#
#   Two models serve orthogonal roles:
#     ref_mlm_model  → "what tokens are plausible here?"
#     target_encoder → "how different are representations of those tokens?"
#
#   tqdm progress bar shows contexts processed vs N_ctx.

@torch.no_grad()
def compute_arg3_irreducible_variance(
    target_encoder: torch.nn.Module,
    ref_mlm_model: torch.nn.Module,
    loader_iter,
    device: torch.device,
    tokenizer,
    K: int             = 16,
    N_ctx: int         = 200,
    temperature: float = 1.0,
    min_prob: float    = 0.01,
    split: str         = "train",
) -> dict:
    """
    Estimate Var(z* | x_C, p_j) for text.

    Args:
        target_encoder: frozen EMA BERT-Large encoder (model.target_encoder)
        ref_mlm_model:  frozen bert-base-uncased BertForMaskedLM — used solely
                        to sample K plausible token completions at masked pos j.
                        Has NO connection to the T-JEPA training loop.
        loader_iter:    iterator over make_c4_dataloader batches
        device:         torch.device
        tokenizer:      HuggingFace tokenizer (for mask_token_id)
        K:              number of sampled completions per masked position
        N_ctx:          number of contexts to average over
        temperature:    softmax temperature for token sampling (ref_mlm_model)
        min_prob:       minimum token probability threshold (ref_mlm_model)
        split:          "train" or "val" — used in tqdm description

    Returns:
        dict:
            irred_var          — mean Var(z*|x_C,p_j) over N_ctx contexts
            n_contexts         — actual contexts processed
            mean_n_candidates  — mean number of above-threshold candidates
    """
    target_encoder.eval()
    ref_mlm_model.eval()

    MASK_ID = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 103

    all_vars          = []
    n_candidates_list = []
    contexts_done     = 0

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
            batch = next(loader_iter)
        except StopIteration:
            break

        input_ids      = batch["clean_input_ids"].to(device)
        attention_mask = batch["clean_attention_mask"].to(device)
        span_mask      = batch["span_mask"].to(device)

        B, L   = input_ids.shape
        budget = min(B, N_ctx - contexts_done)
        input_ids      = input_ids[:budget]
        attention_mask = attention_mask[:budget]
        span_mask      = span_mask[:budget]

        # ── Find first masked position per row ────────────────────────────
        valid_rows = []
        masked_pos = []
        for i in range(budget):
            positions = span_mask[i].bool().nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                continue
            valid_rows.append(i)
            masked_pos.append(positions[0].item())

        if not valid_rows:
            continue

        ids_ctx  = input_ids[valid_rows]        # [V, L]
        attn_ctx = attention_mask[valid_rows]   # [V, L]
        V = ids_ctx.shape[0]
        masked_pos_t = torch.tensor(masked_pos, device=device)  # [V]

        # ── Replace masked position with [MASK] ───────────────────────────
        ids_masked = ids_ctx.clone()
        ids_masked[torch.arange(V, device=device), masked_pos_t] = MASK_ID

        # ── Get MLM logits from frozen bert-base-uncased ──────────────────
        # ref_mlm_model is bert-base (D=768), independent of target_encoder (D=1024)
        ref_out    = ref_mlm_model(
            input_ids      = ids_masked,
            attention_mask = attn_ctx,
        )
        # logits: [V, L, vocab_size]  (vocab_size same for bert-base and bert-large)
        ref_logits = ref_out.logits

        # Extract logits at masked position j per row → [V, vocab_size]
        pos_logits = ref_logits[torch.arange(V, device=device), masked_pos_t]

        # ── Sample K completions per context ──────────────────────────────
        embed_dim     = CFG["hidden_dim"]   # 1024, matches target_encoder output
        z_completions = torch.zeros(V, K, embed_dim, device=device)

        for i in range(V):
            # Temperature-scaled probabilities from bert-base
            p_i = torch.softmax(pos_logits[i] / temperature, dim=-1)  # [vocab_size]

            # Filter by min_prob
            cand_ids   = (p_i >= min_prob).nonzero(as_tuple=False).squeeze(-1)
            cand_probs = p_i[cand_ids]
            n_cand     = cand_ids.shape[0]
            n_candidates_list.append(n_cand)

            if n_cand == 0:
                cand_ids   = p_i.argmax(keepdim=True)
                cand_probs = torch.ones(1, device=device)
                n_cand     = 1

            # Normalise and sample K tokens
            cand_probs        = cand_probs / cand_probs.sum()
            sampled_idx       = torch.multinomial(cand_probs, num_samples=K, replacement=(n_cand < K))
            sampled_token_ids = cand_ids[sampled_idx]  # [K]

            # Build K variants: replace position j with each sampled token
            ids_k  = ids_masked[i:i+1].expand(K, -1).clone()  # [K, L]
            ids_k[torch.arange(K, device=device), masked_pos_t[i]] = sampled_token_ids
            attn_k = attn_ctx[i:i+1].expand(K, -1)            # [K, L]

            # Encode through frozen target_encoder (BERT-Large EMA, D=1024)
            z_full_k = target_encoder(
                input_ids      = ids_k,
                attention_mask = attn_k,
            ).last_hidden_state                                 # [K, L, 1024]

            z_at_j = z_full_k[:, masked_pos_t[i], :]           # [K, 1024]
            z_completions[i] = z_at_j.float()

        # ── Within-context variance across K completions ──────────────────
        z     = z_completions.float()                           # [V, K, D]
        z_bar = z.mean(dim=1, keepdim=True)                     # [V, 1, D]
        var_per_ctx = ((z - z_bar) ** 2).sum(dim=-1).mean(dim=1)  # [V]
        all_vars.append(var_per_ctx.mean().item())

        contexts_done += V
        pbar.update(V)

    pbar.close()

    mean_var    = float(np.mean(all_vars))          if all_vars          else float("nan")
    mean_n_cand = float(np.mean(n_candidates_list)) if n_candidates_list else float("nan")

    return dict(
        irred_var         = round(mean_var,    8),
        n_contexts        = contexts_done,
        mean_n_candidates = round(mean_n_cand, 2),
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
    epoch: int,          # pre-derived via epoch_of(); passed in for the record
    moco_queue: "MoCoQueue",
) -> dict:
    model.context_encoder.eval()
    model.predictor.eval()
    model.target_encoder.eval()

    total_loss    = 0.0
    n_batches     = 0

    z_flat_list      = []
    z_ctx_pool_list  = []
    z_tgt_pool_list  = []

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

    for batch_idx, batch in val_bar:
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

        z_full_ctx = model.encode_full_sequence(batch, use_target=False)
        z_flat_list.append(
            z_full_ctx.detach().reshape(-1, CFG["hidden_dim"]).float().cpu()
        )
        z_ctx_pool_list.append(z_full_ctx.detach().float().mean(dim=1).cpu())

        if "z_target" in out:
            z_tgt_raw  = out["z_target"].detach()
            z_tgt_pool = z_tgt_raw.mean(dim=1) if z_tgt_raw.dim() == 3 else z_tgt_raw
        else:
            z_full_tgt = model.encode_full_sequence(batch, use_target=True)
            z_tgt_pool = z_full_tgt.detach().float().mean(dim=1)
        z_tgt_pool_list.append(z_tgt_pool.cpu())

    mean_loss = total_loss / max(n_batches, 1)

    z_flat_all = torch.cat(z_flat_list, dim=0).to(device)
    if z_flat_all.shape[0] > 32768:
        idx        = torch.randperm(z_flat_all.shape[0], device=device)[:32768]
        z_flat_all = z_flat_all[idx]

    rank_info = compute_effective_rank(z_flat_all, embed_dim=CFG["hidden_dim"])

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

    model.context_encoder.train()
    model.predictor.train()

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
        "(bert-base oracle → BERT-Large EMA encoder)",
        fontsize=10,
    )

    # ── Suptitle ──────────────────────────────────────────────────────────
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
    path, model, optimiser, epoch, global_step,
    train_records, val_records,
    arg3_train_records, arg3_val_records,
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
        arg3_train_records   = arg3_train_records,
        arg3_val_records     = arg3_val_records,
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
    log.info(f"Train dataset: {len(train_loader.dataset):,} sentences, {len(train_loader):,} batches/epoch")

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
    log.info(f"Val dataset:   {len(val_loader.dataset):,} sentences, {len(val_loader):,} batches")

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
        f"(target_encoder: EMA copy, τ={CFG['ema_range'][0]}, no grad)"
    )

    # ── Arg III: load frozen bert-base-uncased as MLM oracle ──────────────
    # Completely separate from T-JEPA. Used ONLY to sample plausible token
    # completions at masked positions. Never updated during training.
    log.info(f"Loading frozen MLM oracle: {CFG['arg3_ref_model']} ...")
    ref_mlm_model = _BertForMaskedLM.from_pretrained(CFG["arg3_ref_model"]).to(device)
    ref_mlm_model.eval()
    for p in ref_mlm_model.parameters():
        p.requires_grad_(False)
    ref_params = sum(p.numel() for p in ref_mlm_model.parameters()) / 1e6
    log.info(f"  MLM oracle loaded — {ref_params:.1f}M params, fully frozen.")

    tokenizer = train_loader.dataset.tokenizer

    moco_queue = MoCoQueue(
        queue_size = CFG["moco_queue_size"],
        embed_dim  = CFG["hidden_dim"],
        device     = device,
    )
    log.info(f"MoCo queue initialised — size={CFG['moco_queue_size']}, embed_dim={CFG['hidden_dim']}")

    trainable_params = (list(model.context_encoder.parameters()) +
                        list(model.predictor.parameters()))
    optimiser = torch.optim.AdamW(
        trainable_params, lr=CFG["start_lr"], weight_decay=CFG["weight_decay"]
    )

    steps_per_epoch = len(train_loader)
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

        model.context_encoder.train()
        model.predictor.train()
        model.target_encoder.eval()
        epoch_losses = []

        # Compute the display epoch once at the start of this loop iteration.
        # It equals _epoch_loop_var when global_step==0 or when running fresh,
        # but stays correct after a resume where global_step is non-zero.
        ep_display_start = epoch_of(global_step, steps_per_epoch)

        iter_bar = tqdm(
            train_loader,
            total=len(train_loader),
            unit="it",
            position=1,
            leave=True,
            dynamic_ncols=True,
        )

        for batch in iter_bar:
            # ── Derive epoch and iter from global_step ────────────────────
            ep  = epoch_of(global_step, steps_per_epoch)   # 1-based, never resets
            it  = iter_of(global_step, steps_per_epoch)    # 1-based within epoch

            iter_bar.set_description(f"Ep {ep:03d}")

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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=CFG["grad_clip"])
            optimiser.step()
            model.update_target_encoder(decay=current_ema)

            with torch.no_grad():
                if "z_target" in out:
                    z_tgt_raw  = out["z_target"].detach()
                    z_tgt_pool = z_tgt_raw.mean(dim=1) if z_tgt_raw.dim() == 3 else z_tgt_raw
                else:
                    z_full_tgt = model.encode_full_sequence(batch, use_target=True)
                    z_tgt_pool = z_full_tgt.detach().mean(dim=1)
            moco_queue.enqueue(z_tgt_pool)

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
                    z_full_ctx = model.encode_full_sequence(batch, use_target=False)
                    z_flat     = z_full_ctx.detach().reshape(-1, CFG["hidden_dim"])
                    rank_info  = compute_effective_rank(z_flat, embed_dim=CFG["hidden_dim"])

                record = dict(
                    global_step         = global_step,
                    epoch               = ep,   # derived from global_step
                    iter                = it,   # derived from global_step
                    split               = "train",
                    loss                = round(loss.item(), 6),
                    effective_rank      = round(rank_info["effective_rank"],      4),
                    normalized_rank     = round(rank_info["normalized_rank"],     6),
                    participation_ratio = round(rank_info["participation_ratio"], 4),
                    embed_dim           = CFG["hidden_dim"],
                    model               = "T-JEPA",
                    ema_tau             = round(float(current_ema),               6),
                    moco_queue_len      = len(moco_queue),
                )

                if global_step % CFG["arg1_every"] == 0:
                    with torch.no_grad():
                        z_ctx_pool = z_full_ctx.detach().float().mean(dim=1)
                        mi_proxy   = compute_infonce_mi(
                            z_ctx_pool, z_tgt_pool,
                            temperature=CFG["mi_temperature"],
                            queue=moco_queue,
                        )
                    record["mi_proxy"] = mi_proxy

                if global_step % CFG["arg2_every"] == 0:
                    arg2 = compute_arg2_metrics(z_flat, embed_dim=CFG["hidden_dim"])
                    record.update(arg2)
                    log.info(
                        f"[TRAIN ep {ep:03d}|it {it:04d}|step {global_step:06d}]  "
                        f"loss={loss.item():.4f}  "
                        f"eff_rank={rank_info['effective_rank']:.2f}  "
                        f"norm_rank={rank_info['normalized_rank']:.4f}  "
                        f"mi={record.get('mi_proxy', float('nan')):.4f}  "
                        f"λ_min_ratio={arg2['lambda_min_ratio']:.4f}  "
                        f"cos_μ={arg2['cosine_sim_mean']:.4f}  "
                        f"cos_p95={arg2['cosine_sim_p95']:.4f}  "
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
                    model, val_loader, device, global_step,
                    epoch=ep,   # derived from global_step
                    moco_queue=moco_queue,
                )
                val_records.append(val_record)

            # ── Argument III: irreducible variance every arg3_every steps ──
            if global_step % CFG["arg3_every"] == 0:
                log.info(f"  → Computing Arg III irreducible variance at step {global_step} ...")

                # --- TRAIN split ---
                arg3_train = compute_arg3_irreducible_variance(
                    target_encoder = model.target_encoder,
                    ref_mlm_model  = ref_mlm_model,
                    loader_iter    = iter(train_loader),
                    device         = device,
                    tokenizer      = tokenizer,
                    K              = CFG["arg3_K"],
                    N_ctx          = CFG["arg3_N_ctx"],
                    temperature    = CFG["arg3_temperature"],
                    min_prob       = CFG["arg3_min_prob"],
                    split          = "train",
                )
                arg3_train_records.append(dict(
                    global_step       = global_step,
                    epoch             = ep,   # derived from global_step
                    split             = "train",
                    model             = "T-JEPA",
                    irred_var         = arg3_train["irred_var"],
                    n_contexts        = arg3_train["n_contexts"],
                    mean_n_candidates = arg3_train["mean_n_candidates"],
                ))

                # --- VAL split ---
                arg3_val = compute_arg3_irreducible_variance(
                    target_encoder = model.target_encoder,
                    ref_mlm_model  = ref_mlm_model,
                    loader_iter    = iter(val_loader),
                    device         = device,
                    tokenizer      = tokenizer,
                    K              = CFG["arg3_K"],
                    N_ctx          = CFG["arg3_N_ctx"],
                    temperature    = CFG["arg3_temperature"],
                    min_prob       = CFG["arg3_min_prob"],
                    split          = "val",
                )
                arg3_val_records.append(dict(
                    global_step       = global_step,
                    epoch             = ep,   # derived from global_step
                    split             = "val",
                    model             = "T-JEPA",
                    irred_var         = arg3_val["irred_var"],
                    n_contexts        = arg3_val["n_contexts"],
                    mean_n_candidates = arg3_val["mean_n_candidates"],
                ))

                log.info(
                    f"  [ARG III step {global_step:06d}]  "
                    f"irred_var train={arg3_train['irred_var']:.6f}  "
                    f"val={arg3_val['irred_var']:.6f}  "
                    f"candidates μ={arg3_train['mean_n_candidates']:.1f}  "
                    f"(K={CFG['arg3_K']}, N_ctx={CFG['arg3_N_ctx']})"
                )

                _write_json_atomic(arg3_train_records, JSON_ARG3_TRAIN)
                _write_json_atomic(arg3_val_records,   JSON_ARG3_VAL)

            if global_step % CFG["log_every"] == 0 or global_step % CFG["eval_every"] == 0:
                save_records(train_records, val_records, arg3_train_records, arg3_val_records)

            global_step += 1

        # ── End of epoch summary ──────────────────────────────────────────
        # Use ep_display_start (epoch at entry) for the "complete" message;
        # epoch_of(global_step-1) is also correct and identical here.
        finished_ep = epoch_of(global_step - 1, steps_per_epoch)
        mean_ep_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"── Epoch {finished_ep:03d} complete  mean_loss={mean_ep_loss:.4f}")
        epoch_bar.set_postfix(mean_loss=f"{mean_ep_loss:.4f}", ep=finished_ep)

        log.info("Saving records (Train JSON → Val JSON → PNG) …")
        save_records(train_records, val_records, arg3_train_records, arg3_val_records)

        log.info("Saving checkpoint …")
        save_checkpoint(
            CKPT_PATH, model, optimiser,
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