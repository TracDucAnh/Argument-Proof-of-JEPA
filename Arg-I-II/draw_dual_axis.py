# draw_dual_axis.py
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 4 plot so sánh I-JEPA vs T-JEPA từ JSON logs.
#
# Plot 1 : Loss của cả hai trên cùng 1 axis (log-scale)
# Plot 2 : Dual-axis effective rank (raw) — T-JEPA trái, I-JEPA phải
# Plot 3 : Single-axis effective rank (log-scale)
# Plot 4 : Plot 1 + Plot 2 nằm kế nhau (1 row × 2 col)
#
# Chỉ vẽ đến step 15 000.
#
# Usage (chạy từ thư mục Arg-I/ hoặc chỉ định đường dẫn):
#   python draw_dual_axis.py
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent.resolve()
IJEPA_JSON = HERE / "I-JEPA.json"
TJEPA_JSON = HERE / "T-JEPA.json"
OUT_DIR    = HERE
STEP_CAP   = 15_000   # chỉ vẽ đến step này

# ── Màu sắc ──────────────────────────────────────────────────────────────────
# I-JEPA: xanh dương
C_IJEPA_MAIN  = "#2563EB"   # blue-600
C_IJEPA_LIGHT = "#93C5FD"   # blue-300 (cho axis phụ)

# T-JEPA: đỏ cam (giữ nguyên palette gốc)
C_TJEPA_MAIN  = "#D85A30"   # orange-red
C_TJEPA_LIGHT = "#F4A07A"   # light orange (cho axis phụ)

# ── Style chung ───────────────────────────────────────────────────────────────
FONT_SIZE_TITLE  = 13
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 9
FONT_SIZE_LEGEND = 10
LINE_WIDTH       = 1.6
ALPHA_GRID       = 0.25
FIG_DPI          = 120


def load_json(path: Path, step_cap: int) -> dict[str, list]:
    """Load JSON log, lọc đến step_cap, trả về dict of lists."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    filtered = [r for r in records if r["global_step"] <= step_cap]

    keys = ["global_step", "loss", "effective_rank", "normalized_rank", "participation_ratio"]
    out  = {k: [r[k] for r in filtered] for k in keys}
    return out


def apply_common_style(ax, xlabel="Training steps", ylabel=None, ylabel_color="black"):
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, color=ylabel_color)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.spines["top"].set_visible(False)


# ═════════════════════════════════════════════════════════════════════════════
# Hàm vẽ từng plot (dùng lại trong plot 4)
# ═════════════════════════════════════════════════════════════════════════════

def draw_plot1_loss(ax, ijepa, tjepa, title=True):
    """Plot 1: Loss cả hai, cùng axis, log-scale."""
    ax.plot(ijepa["global_step"], ijepa["loss"],
            color=C_IJEPA_MAIN, linewidth=LINE_WIDTH, label="I-JEPA loss")
    ax.plot(tjepa["global_step"], tjepa["loss"],
            color=C_TJEPA_MAIN, linewidth=LINE_WIDTH, label="T-JEPA loss")
    ax.set_yscale("log")
    apply_common_style(ax, ylabel="MSE Loss (log scale)")
    ax.tick_params(axis="y", labelcolor="black")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.85)
    if title:
        ax.set_title("Loss Comparison (log scale)", fontsize=FONT_SIZE_TITLE, fontweight="bold")


def draw_plot2_rank_dual(ax, ijepa, tjepa, title=True):
    """Plot 2: Dual-axis raw effective rank — T-JEPA trái, I-JEPA phải."""
    # Trục trái: T-JEPA
    ax.plot(tjepa["global_step"], tjepa["effective_rank"],
            color=C_TJEPA_MAIN, linewidth=LINE_WIDTH, label="T-JEPA eff. rank")
    apply_common_style(ax, ylabel="T-JEPA Effective Rank", ylabel_color=C_TJEPA_MAIN)
    ax.tick_params(axis="y", labelcolor=C_TJEPA_MAIN)
    ax.spines["left"].set_edgecolor(C_TJEPA_MAIN)

    # Trục phải: I-JEPA
    ax2 = ax.twinx()
    ax2.plot(ijepa["global_step"], ijepa["effective_rank"],
             color=C_IJEPA_MAIN, linewidth=LINE_WIDTH, linestyle="--",
             label="I-JEPA eff. rank")
    ax2.set_ylabel("I-JEPA Effective Rank", fontsize=FONT_SIZE_LABEL, color=C_IJEPA_MAIN)
    ax2.tick_params(axis="y", labelcolor=C_IJEPA_MAIN, labelsize=FONT_SIZE_TICK)
    ax2.spines["right"].set_edgecolor(C_IJEPA_MAIN)
    ax2.spines["top"].set_visible(False)

    # Gộp legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=FONT_SIZE_LEGEND, framealpha=0.85, loc="upper right")

    if title:
        ax.set_title("Effective Rank — Dual Axis (trend view)",
                     fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.set_xlabel("Training steps", fontsize=FONT_SIZE_LABEL)

    return ax2   # trả về trục phải để caller có thể tinh chỉnh nếu cần


def draw_plot3_rank_log(ax, ijepa, tjepa, title=True):
    """Plot 3: Single-axis effective rank, log-scale."""
    ax.plot(ijepa["global_step"], ijepa["effective_rank"],
            color=C_IJEPA_MAIN, linewidth=LINE_WIDTH, label="I-JEPA eff. rank")
    ax.plot(tjepa["global_step"], tjepa["effective_rank"],
            color=C_TJEPA_MAIN, linewidth=LINE_WIDTH, label="T-JEPA eff. rank")
    ax.set_yscale("log")
    apply_common_style(ax, ylabel="Effective Rank (log scale)")
    ax.tick_params(axis="y", labelcolor="black")
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.85)
    if title:
        ax.set_title("Effective Rank Comparison (log scale)",
                     fontsize=FONT_SIZE_TITLE, fontweight="bold")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading I-JEPA  ← {IJEPA_JSON}")
    ijepa = load_json(IJEPA_JSON, STEP_CAP)
    print(f"  {len(ijepa['global_step'])} records (up to step {STEP_CAP:,})")

    print(f"Loading T-JEPA  ← {TJEPA_JSON}")
    tjepa = load_json(TJEPA_JSON, STEP_CAP)
    print(f"  {len(tjepa['global_step'])} records (up to step {STEP_CAP:,})")

    # ── Plot 1: Loss ──────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    draw_plot1_loss(ax1, ijepa, tjepa)
    fig1.tight_layout()
    p1 = OUT_DIR / "plot1_loss.png"
    fig1.savefig(p1, dpi=FIG_DPI)
    plt.close(fig1)
    print(f"Saved → {p1}")

    # ── Plot 2: Dual-axis effective rank ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    draw_plot2_rank_dual(ax2, ijepa, tjepa)
    fig2.tight_layout()
    p2 = OUT_DIR / "plot2_rank_dual.png"
    fig2.savefig(p2, dpi=FIG_DPI)
    plt.close(fig2)
    print(f"Saved → {p2}")

    # ── Plot 3: Single-axis rank, log-scale ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    draw_plot3_rank_log(ax3, ijepa, tjepa)
    fig3.tight_layout()
    p3 = OUT_DIR / "plot3_rank_log.png"
    fig3.savefig(p3, dpi=FIG_DPI)
    plt.close(fig3)
    print(f"Saved → {p3}")

    # ── Plot 4: Plot1 + Plot2 side by side ────────────────────────────────
    fig4, (axA, axB) = plt.subplots(1, 2, figsize=(18, 4.5), dpi=FIG_DPI)
    draw_plot1_loss(axA, ijepa, tjepa, title=True)
    draw_plot2_rank_dual(axB, ijepa, tjepa, title=True)
    fig4.suptitle(
        f"I-JEPA vs T-JEPA — Training Dynamics  [steps 0–{STEP_CAP:,}]",
        fontsize=FONT_SIZE_TITLE + 1, fontweight="bold", y=1.01,
    )
    fig4.tight_layout()
    p4 = OUT_DIR / "plot4_combined.png"
    fig4.savefig(p4, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved → {p4}")

    print("\nDone. 4 plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()