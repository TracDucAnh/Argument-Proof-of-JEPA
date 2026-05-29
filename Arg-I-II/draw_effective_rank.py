# draw_effective_rank.py
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 3 plot effective rank (dual-axis) so sánh I-JEPA vs T-JEPA:
#   plot1_rank_train.png  — Train eff. rank (I-JEPA.json vs T-JEPA.json)
#   plot2_rank_val.png    — Val   eff. rank (I-JEPA_val.json vs T-JEPA_val.json)
#   plot3_rank_all.png    — Train + Val cả hai trên cùng 1 ảnh (dual-axis)
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

STEP_CAP = 15_000

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN = "#2563EB"   # blue-600
C_IJEPA_VAL   = "#60A5FA"   # blue-400  (nhạt hơn, dashed)
C_TJEPA_TRAIN = "#D85A30"   # orange-red
C_TJEPA_VAL   = "#F4A07A"   # light orange (nhạt hơn, dashed)

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
FONT_LEGEND = 10
LW          = 1.7
LW_VAL      = 1.4
ALPHA_GRID  = 0.25
FIG_DPI     = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> tuple[list, list]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    steps = [r["global_step"]    for r in filtered]
    ranks = [r["effective_rank"] for r in filtered]
    return steps, ranks


def make_dual_ax(fig, ax,
                 s_t, r_t, label_t, color_t, ls_t,
                 s_i, r_i, label_i, color_i, ls_i,
                 title: str, lw_t=LW, lw_i=LW):
    """
    Vẽ dual-axis lên (fig, ax):
      - Trục trái  : T-JEPA  (cam đỏ)
      - Trục phải  : I-JEPA  (xanh dương)
    Trả về (ax_left, ax_right, [line handles]).
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
    ax.legend(handles=[ln1, ln2],
              labels=[label_t, label_i],
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
# Plot 3 — Train + Val cả hai (dual-axis, 4 đường)
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    s_it, r_it = load_json(IJEPA_TRAIN)
    s_iv, r_iv = load_json(IJEPA_VAL)
    s_tt, r_tt = load_json(TJEPA_TRAIN)
    s_tv, r_tv = load_json(TJEPA_VAL)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=FIG_DPI)

    # ── Trục trái: T-JEPA (train solid, val dashed) ──
    ln1, = ax.plot(s_tt, r_tt, color=C_TJEPA_TRAIN, linewidth=LW,
                   linestyle="-",  label="T-JEPA train")
    ln2, = ax.plot(s_tv, r_tv, color=C_TJEPA_VAL,   linewidth=LW_VAL,
                   linestyle="--", label="T-JEPA val")
    ax.set_ylabel("T-JEPA Effective Rank", fontsize=FONT_LABEL, color=C_TJEPA_TRAIN)
    ax.tick_params(axis="y", labelcolor=C_TJEPA_TRAIN, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(C_TJEPA_TRAIN)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)

    # ── Trục phải: I-JEPA (train solid, val dashed) ──
    ax2 = ax.twinx()
    ln3, = ax2.plot(s_it, r_it, color=C_IJEPA_TRAIN, linewidth=LW,
                    linestyle="-",  label="I-JEPA train")
    ln4, = ax2.plot(s_iv, r_iv, color=C_IJEPA_VAL,   linewidth=LW_VAL,
                    linestyle="--", label="I-JEPA val")
    ax2.set_ylabel("I-JEPA Effective Rank", fontsize=FONT_LABEL, color=C_IJEPA_TRAIN)
    ax2.tick_params(axis="y", labelcolor=C_IJEPA_TRAIN, labelsize=FONT_TICK)
    ax2.spines["right"].set_edgecolor(C_IJEPA_TRAIN)
    ax2.spines["top"].set_visible(False)

    # ── Gộp legend 4 đường ──
    ax.legend(handles=[ln1, ln2, ln3, ln4],
              labels=["T-JEPA train", "T-JEPA val", "I-JEPA train", "I-JEPA val"],
              fontsize=FONT_LEGEND, framealpha=0.88, loc="upper right")

    ax.set_title(
        f"Train & Validation Effective Rank — I-JEPA vs T-JEPA (dual axis)\n"
        f"[steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE, fontweight="bold",
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