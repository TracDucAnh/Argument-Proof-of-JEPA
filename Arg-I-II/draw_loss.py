# draw_loss_compare.py
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 3 plot loss (log-scale) so sánh I-JEPA vs T-JEPA:
#   plot1_loss_train.png  — Train loss (I-JEPA.json vs T-JEPA.json)
#   plot2_loss_val.png    — Val   loss (I-JEPA_val.json vs T-JEPA_val.json)
#   plot3_loss_all.png    — Train | Val kế nhau (1×2 panel, dual-axis mỗi panel)
#
# Chỉ vẽ đến step 15 000.
# Đánh dấu vị trí T-JEPA bắt đầu mất ổn định tại step 9,340:
#   – Vertical dashed line màu đỏ
#   – Shaded region màu đỏ nhạt từ step đó sang phải
# Output: ./loss/
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
INSTABILITY_STEP = 9_340   # T-JEPA bắt đầu mất ổn định

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN      = "#2563EB"   # blue-600  — I-JEPA train
C_IJEPA_VAL        = "#60A5FA"   # blue-400  — I-JEPA val  (nhạt hơn)
C_TJEPA_TRAIN      = "#D85A30"   # orange-red — T-JEPA train
C_TJEPA_VAL        = "#F4A07A"   # light orange — T-JEPA val (nhạt hơn)
C_INSTABILITY      = "#DC2626"   # red-600  — dashed line
C_INSTABILITY_FILL = "#FCA5A5"   # red-300  — shaded region

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
FONT_LEGEND = 9
LW          = 1.7
LW_VAL      = 1.4          # val dùng nét mảnh hơn + dashed
ALPHA_GRID  = 0.25
ALPHA_SHADE = 0.15         # độ trong suốt của vùng shaded
FIG_DPI     = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> tuple[list, list]:
    """Trả về (steps, losses) đã lọc đến step_cap."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    steps  = [r["global_step"] for r in filtered]
    losses = [r["loss"]        for r in filtered]
    return steps, losses


def add_instability_marker(ax, label: bool = True):
    """Thêm dashed line đỏ + shaded region từ INSTABILITY_STEP đến STEP_CAP."""
    ymin, ymax = ax.get_ylim()
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=ALPHA_SHADE, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5,
               linestyle="--", zorder=3,
               label="T-JEPA instability" if label else None)
    ax.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ymax),
        xytext=(INSTABILITY_STEP + 300, ymax),
        fontsize=8,
        color=C_INSTABILITY,
        va="top",
        ha="left",
    )


def style_ax(ax, title: str, ylabel: str = "MSE Loss (log scale)",
             add_marker: bool = True):
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
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.88)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Train loss only
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    s_i, l_i = load_json(IJEPA_TRAIN)
    s_t, l_t = load_json(TJEPA_TRAIN)

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
    s_i, l_i = load_json(IJEPA_VAL)
    s_t, l_t = load_json(TJEPA_VAL)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    ax.plot(s_i, l_i, color=C_IJEPA_VAL, linewidth=LW, label="I-JEPA (val)")
    ax.plot(s_t, l_t, color=C_TJEPA_VAL, linewidth=LW, label="T-JEPA (val)")
    style_ax(ax, title="Validation Loss Comparison (log scale)")
    fig.tight_layout()
    out = OUT_DIR / "plot2_loss_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: vẽ 1 dual-axis loss panel (T-JEPA trái, I-JEPA phải)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_loss_dual(ax,
                    s_t, l_t, label_t, color_t,
                    s_i, l_i, label_i, color_i,
                    ls_t="-", ls_i="-",
                    title: str = ""):
    """Dual-axis log-scale loss panel + instability marker."""
    # ── Trục trái: T-JEPA ──
    ln_t, = ax.plot(s_t, l_t, color=color_t, linewidth=LW,
                    linestyle=ls_t, label=label_t)
    ax.set_yscale("log")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_ylabel("T-JEPA Loss (log scale)", fontsize=FONT_LABEL, color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(color_t)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    # ── Trục phải: I-JEPA ──
    ax2 = ax.twinx()
    ln_i, = ax2.plot(s_i, l_i, color=color_i, linewidth=LW_VAL,
                     linestyle=ls_i, label=label_i)
    ax2.set_yscale("log")
    ax2.set_ylabel("I-JEPA Loss (log scale)", fontsize=FONT_LABEL, color=color_i)
    ax2.tick_params(axis="y", labelcolor=color_i, labelsize=FONT_TICK)
    ax2.spines["right"].set_edgecolor(color_i)
    ax2.spines["top"].set_visible(False)

    # ── Instability marker ──
    add_instability_marker(ax, label=True)

    # ── Gộp legend ──
    instab_line = plt.Line2D([0], [0], color=C_INSTABILITY, linewidth=1.5,
                              linestyle="--", label="T-JEPA instability")
    ax.legend(handles=[ln_t, ln_i, instab_line],
              labels=[label_t, label_i, "T-JEPA instability"],
              fontsize=FONT_LEGEND, framealpha=0.88, loc="upper right")

    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Train | Val kế nhau (1×2 panel, dual-axis mỗi panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    s_it, l_it = load_json(IJEPA_TRAIN)
    s_iv, l_iv = load_json(IJEPA_VAL)
    s_tt, l_tt = load_json(TJEPA_TRAIN)
    s_tv, l_tv = load_json(TJEPA_VAL)

    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(18, 5), dpi=FIG_DPI)

    _draw_loss_dual(
        ax_tr,
        s_tt, l_tt, "T-JEPA (train)", C_TJEPA_TRAIN,
        s_it, l_it, "I-JEPA (train)", C_IJEPA_TRAIN,
        ls_t="-", ls_i="-",
        title="Train Loss (log scale, dual axis)",
    )
    _draw_loss_dual(
        ax_val,
        s_tv, l_tv, "T-JEPA (val)", C_TJEPA_VAL,
        s_iv, l_iv, "I-JEPA (val)", C_IJEPA_VAL,
        ls_t="--", ls_i="--",
        title="Validation Loss (log scale, dual axis)",
    )

    fig.suptitle(
        f"Train & Validation Loss — I-JEPA vs T-JEPA  [steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot3_loss_all.png"
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
    print("\nDone — 3 plots saved to loss/")


if __name__ == "__main__":
    main()