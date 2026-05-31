# draw_loss_compare.py
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 4 plot so sánh I-JEPA vs T-JEPA:
#   plot1_loss_train.png  — Train loss (I-JEPA.json vs T-JEPA.json)
#   plot2_loss_val.png    — Val   loss (I-JEPA_val.json vs T-JEPA_val.json)
#   plot3_loss_all.png    — Left:  Loss (train+val) dual-axis log
#                           Right: Effective Rank (train+val) log scale dual-axis
#                                  (= plot4 style, merged into dual-axis)
#                                  with point annotations at step ~2000 & final step
#   plot4_rank_log.png    — Effective Rank (log scale) 1×2 panel, T-JEPA | I-JEPA
#
# Chỉ vẽ đến step 15 000.
# Đánh dấu vị trí T-JEPA bắt đầu mất ổn định tại step 9,340.
# Output: ./loss/
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN  = HERE / "I-JEPA.json"
TJEPA_TRAIN  = HERE / "T-JEPA.json"
IJEPA_VAL    = HERE / "I-JEPA_val.json"
TJEPA_VAL    = HERE / "T-JEPA_val.json"

OUT_DIR  = HERE / "loss"
OUT_DIR.mkdir(exist_ok=True)

STEP_CAP         = 15_000
INSTABILITY_STEP = 9_340

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN      = "#60A5FA"
C_IJEPA_VAL        = "#2563EB"
C_TJEPA_TRAIN      = "#F4A07A"
C_TJEPA_VAL        = "#D85A30"
C_INSTABILITY      = "#DC2626"
C_INSTABILITY_FILL = "#FCA5A5"

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE   = 18
FONT_LABEL   = 16
FONT_TICK    = 14
FONT_LEGEND  = 10
FONT_ANNOT   = 8
FONT_POINT   = 13          # size cho point annotations
LW           = 1.7
LW_VAL       = 2
ALPHA_GRID   = 0.25
ALPHA_INSTAB = 0.15
FIG_DPI      = 300

LW_TRAIN_FADED = 1.0
ALPHA_TRAIN    = 0.8
LW_VAL_BOLD    = 2.5
ALPHA_VAL      = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_loss(path: Path, step_cap: int = STEP_CAP) -> tuple[list, list]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    return (
        [r["global_step"] for r in filtered],
        [r["loss"]        for r in filtered],
    )


def load_full(path: Path, step_cap: int = STEP_CAP) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    return {
        "steps":          [r["global_step"]   for r in filtered],
        "losses":         [r["loss"]           for r in filtered],
        "effective_rank": [r["effective_rank"] for r in filtered],
    }


def add_instability_marker(ax, label: bool = True):
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3,
               label=f"T-JEPA instability (step {INSTABILITY_STEP:,})" if label else None)
    ymin, ymax = ax.get_ylim()
    ax.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ymax),
        xytext=(INSTABILITY_STEP - 300, ymax),
        fontsize=FONT_ANNOT - 8,
        color=C_INSTABILITY,
        va="top",
        ha="right",
    )


def style_ax(ax, title: str, ylabel: str = "MSE Loss (log scale)",
             add_marker: bool = True, legend_loc: str = "upper right"):
    ax.set_yscale("log")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if add_marker:
        add_instability_marker(ax, label=True)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.85, loc=legend_loc,
              borderpad=0.5, labelspacing=0.3, handlelength=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Train loss only
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    s_i, l_i = load_loss(IJEPA_TRAIN)
    s_t, l_t = load_loss(TJEPA_TRAIN)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    ax.plot(s_i, l_i, color=C_IJEPA_TRAIN, linewidth=LW, label="I-JEPA (train)")
    ax.plot(s_t, l_t, color=C_TJEPA_TRAIN, linewidth=LW, label="T-JEPA (train)")
    style_ax(ax, title="Train Loss Comparison (log scale)")
    fig.tight_layout()
    out = OUT_DIR / "plot1_loss_train.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val loss only
# ─────────────────────────────────────────────────────────────────────────────

def plot2_val():
    s_i, l_i = load_loss(IJEPA_VAL)
    s_t, l_t = load_loss(TJEPA_VAL)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    ax.plot(s_i, l_i, color=C_IJEPA_VAL, linewidth=LW_VAL, linestyle="--",
            label="I-JEPA (val)")
    ax.plot(s_t, l_t, color=C_TJEPA_VAL, linewidth=LW_VAL, linestyle="--",
            label="T-JEPA (val)")
    style_ax(ax, title="Validation Loss Comparison (log scale)")
    fig.tight_layout()
    out = OUT_DIR / "plot2_loss_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: panel dual-axis LOSS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_loss_dual(ax, t_entries, i_entries, color_t, color_i,
                    title="", legend_loc="upper right"):
    lines_t = []
    for e in t_entries:
        ln, = ax.plot(e["steps"], e["losses"],
                      color=e["color"], linewidth=e["lw"],
                      linestyle=e["ls"], label=e["label"],
                      alpha=e.get("alpha", 1.0))
        lines_t.append(ln)

    ax.set_yscale("log")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_ylabel("T-JEPA Loss (log scale)", fontsize=FONT_LABEL, color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(color_t)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    ax2 = ax.twinx()
    lines_i = []
    for e in i_entries:
        ln, = ax2.plot(e["steps"], e["losses"],
                       color=e["color"], linewidth=e["lw"],
                       linestyle=e["ls"], label=e["label"],
                       alpha=e.get("alpha", 1.0))
        lines_i.append(ln)

    ax2.set_yscale("log")
    ax2.set_ylabel("I-JEPA Loss (log scale)", fontsize=FONT_LABEL, color=color_i)
    ax2.tick_params(axis="y", labelcolor=color_i, labelsize=FONT_TICK)
    ax2.spines["right"].set_edgecolor(color_i)
    ax2.spines["top"].set_visible(False)

    add_instability_marker(ax, label=True)

    instab_line = plt.Line2D([0], [0], color=C_INSTABILITY, linewidth=1.5,
                              linestyle="--",
                              label=f"T-JEPA instability ({INSTABILITY_STEP:,})")
    all_lines = lines_t + lines_i
    ax.legend(handles=all_lines + [instab_line],
              labels=[ln.get_label() for ln in all_lines] +
                     [f"T-JEPA instability ({INSTABILITY_STEP:,})"],
              fontsize=FONT_LEGEND, framealpha=0.85, loc=legend_loc,
              borderpad=0.5, labelspacing=0.3, handlelength=1.5)
    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# Helper: lấy giá trị effective_rank tại một step cụ thể (gần nhất)
# ─────────────────────────────────────────────────────────────────────────────

def _get_value_near_step(steps, values, target_step):
    """Trả về (step_actual, value) tại điểm gần target_step nhất."""
    valid_pairs = [(s, v) for s, v in zip(steps, values) if v is not None and v > 0]
    if not valid_pairs:
        return None, None
    closest = min(valid_pairs, key=lambda x: abs(x[0] - target_step))
    return closest


def _get_final_value(steps, values):
    """Trả về (step, value) của điểm cuối hợp lệ."""
    valid_pairs = [(s, v) for s, v in zip(steps, values) if v is not None and v > 0]
    if not valid_pairs:
        return None, None
    return valid_pairs[-1]


def safe_rank(vals):
    return [max(v, 1e-2) if v is not None else 1e-2 for v in vals]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: vẽ point annotation lên canvas
#   - Dot tại vị trí thực tế
#   - Text label với giá trị, in đậm, size lớn
#   - Arrow nếu cần offset
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_point(ax, step, value, color,
                    label_prefix="",
                    text_offset=(60, 20),
                    fontsize=FONT_POINT,
                    ha="left"):
    """Vẽ dot + text annotation tại (step, value) trên ax."""
    if step is None or value is None:
        return
    # Dot — label="_nolegend_" để scatter không xuất hiện trong legend
    ax.scatter([step], [value], color=color, s=80, zorder=10,
               edgecolors="white", linewidths=1.2, label="_nolegend_")
    # Arrow + label
    ax.annotate(
        f"{label_prefix}{value:.2f}",
        xy=(step, value),
        xytext=(text_offset[0], text_offset[1]),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha=ha,
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=color, linewidth=1.2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        connectionstyle="arc3,rad=0.15"),
        zorder=11,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — LEFT: Loss (train+val) dual-axis log
#           RIGHT: Effective Rank log scale, dual-axis T-JEPA vs I-JEPA
#                  với point annotations tại step ~2000 và step cuối
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    it = load_full(IJEPA_TRAIN)
    iv = load_full(IJEPA_VAL)
    tt = load_full(TJEPA_TRAIN)
    tv = load_full(TJEPA_VAL)

    fig, (ax_loss, ax_rank) = plt.subplots(1, 2, figsize=(20, 6), dpi=FIG_DPI)

    # ── LEFT: Loss panel (giữ nguyên) ────────────────────────────────────────
    _draw_loss_dual(
        ax_loss,
        t_entries=[
            dict(steps=tt["steps"], losses=tt["losses"], label="T-JEPA train",
                 color=C_TJEPA_TRAIN, ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
            dict(steps=tv["steps"], losses=tv["losses"], label="T-JEPA val",
                 color=C_TJEPA_VAL,   ls="-", lw=LW_VAL_BOLD,    alpha=ALPHA_VAL),
        ],
        i_entries=[
            dict(steps=it["steps"], losses=it["losses"], label="I-JEPA train",
                 color=C_IJEPA_TRAIN, ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
            dict(steps=iv["steps"], losses=iv["losses"], label="I-JEPA val",
                 color=C_IJEPA_VAL,   ls="-", lw=LW_VAL_BOLD,    alpha=ALPHA_VAL),
        ],
        color_t=C_TJEPA_TRAIN,
        color_i=C_IJEPA_TRAIN,
        title="Loss — Train & Val (log scale, dual axis)",
        legend_loc="upper right",
    )

    # ── RIGHT: Effective Rank, log scale, dual-axis T-JEPA (left) / I-JEPA (right) ──

    # --- Vẽ T-JEPA trên ax_rank (trục trái) ---
    ax_rank.plot(tt["steps"], safe_rank(tt["effective_rank"]),
                 color=C_TJEPA_TRAIN, linewidth=LW_TRAIN_FADED, alpha=ALPHA_TRAIN,
                 linestyle="-", label="T-JEPA train")
    ax_rank.plot(tv["steps"], safe_rank(tv["effective_rank"]),
                 color=C_TJEPA_VAL, linewidth=LW_VAL_BOLD, alpha=ALPHA_VAL,
                 linestyle="-", label="T-JEPA val")

    ax_rank.set_yscale("log")
    ax_rank.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax_rank.set_ylabel("T-JEPA Effective Rank (log scale)", fontsize=FONT_LABEL,
                        color=C_TJEPA_TRAIN)
    ax_rank.tick_params(axis="y", labelcolor=C_TJEPA_TRAIN, labelsize=FONT_TICK)
    ax_rank.tick_params(axis="x", labelsize=FONT_TICK)
    ax_rank.spines["left"].set_edgecolor(C_TJEPA_TRAIN)
    ax_rank.spines["top"].set_visible(False)
    ax_rank.grid(True, alpha=ALPHA_GRID, linestyle="--", which="both")
    ax_rank.set_title("Effective Rank — Train & Val (log scale, dual axis)",
                       fontsize=FONT_TITLE, fontweight="bold")

    # --- Vẽ I-JEPA trên ax_rank2 (trục phải) ---
    ax_rank2 = ax_rank.twinx()
    ax_rank2.plot(it["steps"], safe_rank(it["effective_rank"]),
                  color=C_IJEPA_TRAIN, linewidth=LW_TRAIN_FADED, alpha=ALPHA_TRAIN,
                  linestyle="-", label="I-JEPA train")
    ax_rank2.plot(iv["steps"], safe_rank(iv["effective_rank"]),
                  color=C_IJEPA_VAL, linewidth=LW_VAL_BOLD, alpha=ALPHA_VAL,
                  linestyle="-", label="I-JEPA val")

    ax_rank2.set_yscale("log")
    ax_rank2.set_ylabel("I-JEPA Effective Rank (log scale)", fontsize=FONT_LABEL,
                         color=C_IJEPA_TRAIN)
    ax_rank2.tick_params(axis="y", labelcolor=C_IJEPA_TRAIN, labelsize=FONT_TICK)
    ax_rank2.spines["right"].set_edgecolor(C_IJEPA_TRAIN)
    ax_rank2.spines["top"].set_visible(False)

    # --- Instability marker ---
    ax_rank.axvspan(INSTABILITY_STEP, STEP_CAP,
                    color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax_rank.axvline(x=INSTABILITY_STEP,
                    color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3)
    ylims = ax_rank.get_ylim()
    ax_rank.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ylims[1]),
        xytext=(INSTABILITY_STEP - 400, ylims[1]),
        fontsize=FONT_ANNOT, color=C_INSTABILITY,
        va="top", ha="right",
    )

    # ── POINT ANNOTATIONS: step ~2000 và step cuối ───────────────────────────
    TARGET_STEP = 2000

    # T-JEPA train @ ~2000
    s2k_tt, v2k_tt = _get_value_near_step(tt["steps"], tt["effective_rank"], TARGET_STEP)
    _annotate_point(ax_rank, s2k_tt, v2k_tt, C_TJEPA_TRAIN,
                    label_prefix="train≈",
                    text_offset=(30, -60), ha="right")

    # T-JEPA val @ ~2000
    s2k_tv, v2k_tv = _get_value_near_step(tv["steps"], tv["effective_rank"], TARGET_STEP)
    _annotate_point(ax_rank, s2k_tv, v2k_tv, C_TJEPA_VAL,
                    label_prefix="val≈",
                    text_offset=(30, -40), ha="right")

    # T-JEPA train @ final
    sf_tt, vf_tt = _get_final_value(tt["steps"], tt["effective_rank"])
    _annotate_point(ax_rank, sf_tt, vf_tt, C_TJEPA_TRAIN,
                    label_prefix="train=",
                    text_offset=(-90, -50), ha="right")

    # T-JEPA val @ final
    sf_tv, vf_tv = _get_final_value(tv["steps"], tv["effective_rank"])
    _annotate_point(ax_rank, sf_tv, vf_tv, C_TJEPA_VAL,
                    label_prefix="val=",
                    text_offset=(-90, 10), ha="right")

    # I-JEPA train @ ~2000  (plot lên ax_rank2)
    s2k_it, v2k_it = _get_value_near_step(it["steps"], it["effective_rank"], TARGET_STEP)
    _annotate_point(ax_rank2, s2k_it, v2k_it, C_IJEPA_TRAIN,
                    label_prefix="train≈",
                    text_offset=(60, 30), ha="left")

    # I-JEPA val @ ~2000
    s2k_iv, v2k_iv = _get_value_near_step(iv["steps"], iv["effective_rank"], TARGET_STEP)
    _annotate_point(ax_rank2, s2k_iv, v2k_iv, C_IJEPA_VAL,
                    label_prefix="val≈",
                    text_offset=(60, -15), ha="left")

    # I-JEPA train @ final
    sf_it, vf_it = _get_final_value(it["steps"], it["effective_rank"])
    _annotate_point(ax_rank2, sf_it, vf_it, C_IJEPA_TRAIN,
                    label_prefix="train=",
                    text_offset=(-90, -80), ha="right")

    # I-JEPA val @ final
    sf_iv, vf_iv = _get_final_value(iv["steps"], iv["effective_rank"])
    _annotate_point(ax_rank2, sf_iv, vf_iv, C_IJEPA_VAL,
                    label_prefix="val=",
                    text_offset=(-90, -130), ha="right")

    # ── Legend kết hợp — build thủ công, tránh pick nhầm axvline / scatter ──
    legend_handles = [
        plt.Line2D([0], [0], color=C_TJEPA_TRAIN, linewidth=LW_TRAIN_FADED,
                   linestyle="-", label="T-JEPA train"),
        plt.Line2D([0], [0], color=C_TJEPA_VAL,   linewidth=LW_VAL_BOLD,
                   linestyle="-", label="T-JEPA val"),
        plt.Line2D([0], [0], color=C_IJEPA_TRAIN, linewidth=LW_TRAIN_FADED,
                   linestyle="-", label="I-JEPA train"),
        plt.Line2D([0], [0], color=C_IJEPA_VAL,   linewidth=LW_VAL_BOLD,
                   linestyle="-", label="I-JEPA val"),
        plt.Line2D([0], [0], color=C_INSTABILITY,  linewidth=1.5,
                   linestyle="--", label=f"T-JEPA instability ({INSTABILITY_STEP:,})"),
    ]
    ax_rank.legend(handles=legend_handles,
                   fontsize=FONT_LEGEND, framealpha=0.88, loc="upper left",
                   borderpad=0.5, labelspacing=0.35, handlelength=1.5)

    fig.tight_layout()
    out = OUT_DIR / "plot3_loss_all.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Effective Rank (log scale, dual axis)
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_endpoint(ax, steps, values, label, color, offset_x=-200, offset_y=1.15):
    if not steps or not values:
        return
    valid = [(s, v) for s, v in zip(steps, values) if v is not None and v > 0]
    if not valid:
        return
    sx, sy = valid[-1]
    ax.annotate(
        f"{sy:.1f}",
        xy=(sx, sy),
        xytext=(sx + offset_x, sy * offset_y),
        fontsize=10,
        color=color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def plot4_rank_log():
    it = load_full(IJEPA_TRAIN)
    iv = load_full(IJEPA_VAL)
    tt = load_full(TJEPA_TRAIN)
    tv = load_full(TJEPA_VAL)

    valid_tt = [v for v in tt["effective_rank"] if v is not None]
    valid_tv = [v for v in tv["effective_rank"] if v is not None]

    tt_min, tt_max = (min(valid_tt), max(valid_tt)) if valid_tt else (0, 0)
    tv_min, tv_max = (min(valid_tv), max(valid_tv)) if valid_tv else (0, 0)

    train_rank_str = f"{tt_min:.0f}–{tt_max:.0f}"

    if tv_max - tv_min < 1.0:
        val_rank_str = f"≈ {valid_tv[-1]:.2f}" if valid_tv else "≈ 0"
    else:
        val_rank_str = f"{tv_min:.1f}–{tv_max:.1f}"

    fig, (ax_t, ax_i) = plt.subplots(1, 2, figsize=(18, 5), dpi=FIG_DPI)

    # ── Panel trái: T-JEPA ───────────────────────────────────────────────────
    ax_t.plot(tt["steps"], safe_rank(tt["effective_rank"]),
              color=C_TJEPA_TRAIN, linewidth=LW_TRAIN_FADED, alpha=ALPHA_TRAIN,
              linestyle="-", label="T-JEPA train")
    ax_t.plot(tv["steps"], safe_rank(tv["effective_rank"]),
              color=C_TJEPA_VAL, linewidth=LW_VAL_BOLD, alpha=ALPHA_VAL,
              linestyle="-", label="T-JEPA val")

    ax_t.set_yscale("log")
    ax_t.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax_t.set_ylabel("Effective Rank (log scale)", fontsize=FONT_LABEL)
    ax_t.set_title("T-JEPA — Effective Rank (log scale)", fontsize=FONT_TITLE, fontweight="bold")
    ax_t.tick_params(axis="both", labelsize=FONT_TICK)
    ax_t.grid(True, alpha=ALPHA_GRID, linestyle="--", which="both")
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)

    _annotate_endpoint(ax_t, tt["steps"], tt["effective_rank"],
                       "train", C_TJEPA_TRAIN, offset_x=-1200, offset_y=1.5)
    _annotate_endpoint(ax_t, tv["steps"], tv["effective_rank"],
                       "val",   C_TJEPA_VAL,   offset_x=-1200, offset_y=0.3)

    ax_t.annotate(
        f"Train rank: {train_rank_str}\n(overfitting masking noise)",
        xy=(STEP_CAP * 0.55, 300),
        xytext=(-120, -10),
        textcoords="offset points",
        fontsize=10, color=C_TJEPA_TRAIN, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=C_TJEPA_TRAIN),
    )
    ax_t.annotate(
        f"Val rank {val_rank_str}\n(no generalizable structure)",
        xy=(STEP_CAP * 0.3, 0.05),
        fontsize=10, color=C_TJEPA_VAL, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=C_TJEPA_VAL),
    )

    ax_t.axvspan(INSTABILITY_STEP, STEP_CAP,
                 color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax_t.axvline(x=INSTABILITY_STEP, color=C_INSTABILITY,
                 linewidth=1.5, linestyle="--", zorder=3)
    ymin_t, ymax_t = ax_t.get_ylim()
    ax_t.annotate(
        f"T-JEPA Instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ymax_t),
        xytext=(INSTABILITY_STEP - 400, ymax_t),
        fontsize=FONT_ANNOT, color=C_INSTABILITY,
        va="top", ha="right",
    )

    ax_t.legend(fontsize=FONT_LEGEND, framealpha=0.85, loc="upper left",
                borderpad=0.5, labelspacing=0.3, handlelength=1.5)

    # ── Panel phải: I-JEPA ───────────────────────────────────────────────────
    ax_i.plot(it["steps"], safe_rank(it["effective_rank"]),
              color=C_IJEPA_TRAIN, linewidth=LW_TRAIN_FADED, alpha=ALPHA_TRAIN,
              linestyle="-", label="I-JEPA train")
    ax_i.plot(iv["steps"], safe_rank(iv["effective_rank"]),
              color=C_IJEPA_VAL, linewidth=LW_VAL_BOLD, alpha=ALPHA_VAL,
              linestyle="-", label="I-JEPA val")

    ax_i.set_yscale("log")
    ax_i.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax_i.set_ylabel("Effective Rank (log scale)", fontsize=FONT_LABEL)
    ax_i.set_title("I-JEPA — Effective Rank (log scale)", fontsize=FONT_TITLE, fontweight="bold")
    ax_i.tick_params(axis="both", labelsize=FONT_TICK)
    ax_i.grid(True, alpha=ALPHA_GRID, linestyle="--", which="both")
    ax_i.spines["top"].set_visible(False)
    ax_i.spines["right"].set_visible(False)

    _annotate_endpoint(ax_i, it["steps"], it["effective_rank"],
                       "train", C_IJEPA_TRAIN, offset_x=-1200, offset_y=1.3)
    _annotate_endpoint(ax_i, iv["steps"], iv["effective_rank"],
                       "val",   C_IJEPA_VAL,   offset_x=-1200, offset_y=0.7)

    ax_i.annotate(
        "Train ≈ Val: generalizable\nrich representation space",
        xy=(STEP_CAP * 0.45, 10),
        fontsize=10, color=C_IJEPA_VAL, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec=C_IJEPA_VAL),
    )

    ax_i.axvspan(INSTABILITY_STEP, STEP_CAP,
                 color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB * 0.5, zorder=0)
    ax_i.axvline(x=INSTABILITY_STEP, color=C_INSTABILITY,
                 linewidth=1.0, linestyle=":", zorder=3, alpha=0.5)
    ax_i.annotate(
        f"T-JEPA Instability\n(reference, step {INSTABILITY_STEP:,})",
        xy=(INSTABILITY_STEP, ax_i.get_ylim()[1] if ax_i.get_ylim()[1] > 1 else 10),
        xytext=(INSTABILITY_STEP - 400, ax_i.get_ylim()[1] if ax_i.get_ylim()[1] > 1 else 10),
        fontsize=FONT_ANNOT, color=C_INSTABILITY, alpha=0.6,
        va="top", ha="right",
    )

    ax_i.legend(fontsize=FONT_LEGEND, framealpha=0.85, loc="upper left",
                borderpad=0.5, labelspacing=0.3, handlelength=1.5)

    fig.text(
        0.5, -0.02,
        f"Log scale exposes T-JEPA's train–val effective rank divergence "
        f"(train: {train_rank_str}, val: {val_rank_str}) hidden on linear scale.",
        ha="center", fontsize=16, style="italic", color="#555555",
    )

    fig.tight_layout()
    out = OUT_DIR / "plot4_rank_log.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Output folder: {OUT_DIR}\n")
    plot1_train()
    plot2_val()
    plot3_all()
    plot4_rank_log()
    print("\nDone — 4 plots saved to loss/")


if __name__ == "__main__":
    main()