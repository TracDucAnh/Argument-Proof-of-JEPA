# draw_effective_rank.py
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 3 plot effective rank (dual-axis) so sánh I-JEPA vs T-JEPA:
#   plot1_rank_train.png  — Train eff. rank (I-JEPA.json vs T-JEPA.json)
#   plot2_rank_val.png    — Val   eff. rank (I-JEPA_val.json vs T-JEPA_val.json)
#   plot3_rank_all.png    — Train | Val kế nhau (1×2 panel, dual-axis mỗi panel)
#
# Dual-axis: T-JEPA trục trái (cam đỏ), I-JEPA trục phải (xanh dương)
# Chỉ vẽ đến step 15 000.
# Output: ./effective_rank/
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN = HERE / "I-JEPA.json"
TJEPA_TRAIN = HERE / "T-JEPA.json"
IJEPA_VAL   = HERE / "I-JEPA_val.json"
TJEPA_VAL   = HERE / "T-JEPA_val.json"

OUT_DIR  = HERE / "effective_rank"
OUT_DIR.mkdir(exist_ok=True)

STEP_CAP         = 15_000
INSTABILITY_STEP = 9_340   # T-JEPA bắt đầu mất ổn định

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN      = "#2563EB"   # blue-600
C_IJEPA_VAL        = "#60A5FA"   # blue-400  (nhạt hơn, dashed)
C_TJEPA_TRAIN      = "#D85A30"   # orange-red
C_TJEPA_VAL        = "#F4A07A"   # light orange (nhạt hơn, dashed)
C_INSTABILITY      = "#DC2626"   # red-600  — dashed line
C_INSTABILITY_FILL = "#FCA5A5"   # red-300  — shaded region

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE   = 13
FONT_LABEL   = 11
FONT_TICK    = 9
FONT_LEGEND  = 10
FONT_ANNOT   = 8
LW           = 1.7
LW_VAL       = 2
ALPHA_GRID   = 0.25
ALPHA_INSTAB = 0.15
FIG_DPI      = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> tuple[list, list]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    steps = [r["global_step"]    for r in filtered]
    ranks = [r["effective_rank"] for r in filtered]
    return steps, ranks


def add_instability_marker(ax, label: bool = True):
    """Thêm dashed line đỏ + shaded region từ INSTABILITY_STEP đến STEP_CAP."""
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3,
               label="T-JEPA instability" if label else None)
    ymin, ymax = ax.get_ylim()
    ax.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ymax),
        xytext=(INSTABILITY_STEP + 300, ymax),
        fontsize=FONT_ANNOT,
        color=C_INSTABILITY,
        va="top",
        ha="left",
    )


def make_dual_ax(fig, ax,
                 s_t, r_t, label_t, color_t, ls_t,
                 s_i, r_i, label_i, color_i, ls_i,
                 title: str, lw_t=LW, lw_i=LW):
    """
    Vẽ dual-axis lên (fig, ax):
      - Trục trái  : T-JEPA  (cam đỏ)
      - Trục phải  : I-JEPA  (xanh dương)
    Trả về (ax_right, [line handles]).
    """
    # ── Trục trái: T-JEPA ──
    ln1, = ax.plot(s_t, r_t, color=color_t, linewidth=lw_t,
                   linestyle=ls_t, label=label_t)
    ax.set_ylabel("T-JEPA Effective Rank", fontsize=FONT_LABEL, color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(color_t)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)

    # ── Trục phải: I-JEPA ──
    ax2 = ax.twinx()
    ln2, = ax2.plot(s_i, r_i, color=color_i, linewidth=lw_i,
                    linestyle=ls_i, label=label_i)
    ax2.set_ylabel("I-JEPA Effective Rank", fontsize=FONT_LABEL, color=color_i)
    ax2.tick_params(axis="y", labelcolor=color_i, labelsize=FONT_TICK)
    ax2.spines["right"].set_edgecolor(color_i)
    ax2.spines["top"].set_visible(False)

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    # ── Instability marker ──
    add_instability_marker(ax, label=True)

    # ── Gộp legend ──
    instab_line = plt.Line2D([0], [0], color=C_INSTABILITY, linewidth=1.5,
                              linestyle="--", label="T-JEPA instability")
    ax.legend(handles=[ln1, ln2, instab_line],
              labels=[label_t, label_i, "T-JEPA instability"],
              fontsize=FONT_LEGEND, framealpha=0.88, loc="upper right")

    return ax2, [ln1, ln2]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Train effective rank (dual-axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    s_i, r_i = load_json(IJEPA_TRAIN)
    s_t, r_t = load_json(TJEPA_TRAIN)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    make_dual_ax(
        fig, ax,
        s_t, r_t, "T-JEPA train", C_TJEPA_TRAIN, "-",
        s_i, r_i, "I-JEPA train", C_IJEPA_TRAIN, "-",
        title=f"Train Effective Rank — Dual Axis  [steps 0–{STEP_CAP:,}]",
    )
    fig.tight_layout()
    out = OUT_DIR / "plot1_rank_train.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val effective rank (dual-axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot2_val():
    s_i, r_i = load_json(IJEPA_VAL)
    s_t, r_t = load_json(TJEPA_VAL)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    make_dual_ax(
        fig, ax,
        s_t, r_t, "T-JEPA val", C_TJEPA_VAL, "--",
        s_i, r_i, "I-JEPA val", C_IJEPA_VAL, "--",
        title=f"Validation Effective Rank — Dual Axis  [steps 0–{STEP_CAP:,}]",
        lw_t=LW_VAL, lw_i=LW_VAL,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot2_rank_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Train | Val kế nhau (1×2 panel, dual-axis mỗi panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    s_it, r_it = load_json(IJEPA_TRAIN)
    s_iv, r_iv = load_json(IJEPA_VAL)
    s_tt, r_tt = load_json(TJEPA_TRAIN)
    s_tv, r_tv = load_json(TJEPA_VAL)

    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(18, 5), dpi=FIG_DPI)

    make_dual_ax(
        fig, ax_tr,
        s_tt, r_tt, "T-JEPA train", C_TJEPA_TRAIN, "-",
        s_it, r_it, "I-JEPA train", C_IJEPA_TRAIN, "-",
        title="Train Effective Rank (dual axis)",
    )
    make_dual_ax(
        fig, ax_val,
        s_tv, r_tv, "T-JEPA val", C_TJEPA_VAL, "--",
        s_iv, r_iv, "I-JEPA val", C_IJEPA_VAL, "--",
        title="Validation Effective Rank (dual axis)",
        lw_t=LW_VAL, lw_i=LW_VAL,
    )

    fig.suptitle(
        f"Train & Validation Effective Rank — I-JEPA vs T-JEPA  [steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot3_rank_all.png"
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
    print("\nDone — 3 plots saved to effective_rank/")


if __name__ == "__main__":
    main()