# draw_residual_var.py
# ─────────────────────────────────────────────────────────────────────────────
# Argument I — Irreducible Noise (Proposition 4.12)
#
# Dual-axis: T-JEPA trục trái (cam đỏ), I-JEPA trục phải (xanh dương)
# → mỗi model có scale riêng, thấy rõ plateau của T-JEPA và trend giảm I-JEPA
#
# 3 output:
#   residual_var/plot1_resvar_train.png
#   residual_var/plot2_resvar_val.png
#   residual_var/plot3_resvar_all.png  — train + val, 1×2 panel
#
# Chỉ vẽ đến step 15 000.
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN = HERE / "I-JEPA.json"
TJEPA_TRAIN = HERE / "T-JEPA.json"
IJEPA_VAL   = HERE / "I-JEPA_val.json"
TJEPA_VAL   = HERE / "T-JEPA_val.json"

OUT_DIR = HERE / "residual_var"
OUT_DIR.mkdir(exist_ok=True)

STEP_CAP         = 15_000
INSTABILITY_STEP = 9_340   # T-JEPA bắt đầu mất ổn định

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN = "#2563EB"   # blue-600
C_IJEPA_VAL   = "#60A5FA"   # blue-400
C_TJEPA_TRAIN = "#D85A30"   # orange-red
C_TJEPA_VAL   = "#F4A07A"   # light orange
C_HLINE       = "#9CA3AF"   # gray-400 — plateau line

C_INSTABILITY      = "#DC2626"   # red-600  — dashed line
C_INSTABILITY_FILL = "#FCA5A5"   # red-300  — shaded region

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE  = 12
FONT_LABEL  = 10
FONT_TICK   = 8.5
FONT_LEGEND = 9
FONT_ANNOT  = 8
LW          = 1.7
LW_VAL      = 1.4
LW_HLINE    = 1.1
ALPHA_GRID  = 0.25
ALPHA_INSTAB = 0.15
FIG_DPI     = 150


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    return {
        "steps":        [r["global_step"]  for r in filtered],
        "residual_var": [r["residual_var"] for r in filtered],
    }


def _plateau_level(values: list[float], tail_frac: float = 0.35) -> float:
    """Median của phần đuôi — ước lượng empirical của irreducible noise floor."""
    tail = values[int(len(values) * (1 - tail_frac)):]
    return float(np.median(tail)) if tail else float(np.mean(values))


def add_instability_marker(ax, label: bool = True):
    """Thêm dashed line đỏ + shaded region từ INSTABILITY_STEP đến STEP_CAP."""
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3,
               label=f"T-JEPA instability" if label else None)
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


# ─────────────────────────────────────────────────────────────────────────────
# Core: dual-axis residual var
# T-JEPA trục trái (cam đỏ) — I-JEPA trục phải (xanh dương)
# ─────────────────────────────────────────────────────────────────────────────

def draw_resvar_dual(ax,
                     s_t, rv_t, label_t, color_t, ls_t, lw_t,
                     s_i, rv_i, label_i, color_i, ls_i, lw_i,
                     title: str,
                     plateau_t: float | None = None) -> plt.Axes:
    """
    Trục trái  : T-JEPA residual var (cam đỏ)
    Trục phải  : I-JEPA residual var (xanh dương)
    Trả về ax_right.
    """
    # ── Trục trái: T-JEPA ──
    ln_t, = ax.plot(s_t, rv_t, color=color_t, linewidth=lw_t,
                    linestyle=ls_t, label=label_t)
    ax.set_ylabel("T-JEPA Residual Variance", fontsize=FONT_LABEL, color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(color_t)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    # Horizontal line tại mức plateau của T-JEPA
    if plateau_t is not None:
        ax.axhline(plateau_t, color=C_HLINE, linewidth=LW_HLINE,
                   linestyle=":", zorder=1)
        x_max = s_t[-1] if s_t else STEP_CAP
        ax.annotate(
            f"Irreducible noise floor ≈ {plateau_t:.5f}",
            xy=(x_max * 0.45, plateau_t),
            xytext=(x_max * 0.45, plateau_t * 1.8),
            fontsize=FONT_ANNOT, color=color_t, style="italic",
            arrowprops=dict(arrowstyle="->", color=color_t, lw=0.8),
            va="bottom",
        )

    # ── Trục phải: I-JEPA ──
    ax2 = ax.twinx()
    ln_i, = ax2.plot(s_i, rv_i, color=color_i, linewidth=lw_i,
                     linestyle=ls_i, label=label_i)
    ax2.set_ylabel("I-JEPA Residual Variance", fontsize=FONT_LABEL, color=color_i)
    ax2.tick_params(axis="y", labelcolor=color_i, labelsize=FONT_TICK)
    ax2.spines["right"].set_edgecolor(color_i)
    ax2.spines["top"].set_visible(False)

    # ── Instability marker (vẽ trên ax chính — trục x chung) ──
    add_instability_marker(ax, label=True)

    # Gộp legend
    instab_line = plt.Line2D([0], [0], color=C_INSTABILITY, linewidth=1.5,
                              linestyle="--",
                              label=f"T-JEPA instability")
    ax.legend(handles=[ln_t, ln_i, instab_line],
              labels=[label_t, label_i,
                      f"T-JEPA instability"],
              fontsize=FONT_LEGEND, framealpha=0.88, loc="upper right")

    return ax2


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Train
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    i = load_json(IJEPA_TRAIN)
    t = load_json(TJEPA_TRAIN)
    plateau = _plateau_level(t["residual_var"])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    draw_resvar_dual(
        ax,
        t["steps"], t["residual_var"], "T-JEPA (train)", C_TJEPA_TRAIN, "-", LW,
        i["steps"], i["residual_var"], "I-JEPA (train)", C_IJEPA_TRAIN, "-", LW,
        title=f"Residual Variance — Train  [steps 0–{STEP_CAP:,}]",
        plateau_t=plateau,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot1_resvar_train.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val
# ─────────────────────────────────────────────────────────────────────────────

def plot2_val():
    i = load_json(IJEPA_VAL)
    t = load_json(TJEPA_VAL)
    plateau = _plateau_level(t["residual_var"])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIG_DPI)
    draw_resvar_dual(
        ax,
        t["steps"], t["residual_var"], "T-JEPA (val)", C_TJEPA_VAL, "--", LW_VAL,
        i["steps"], i["residual_var"], "I-JEPA (val)", C_IJEPA_VAL, "--", LW_VAL,
        title=f"Residual Variance — Validation  [steps 0–{STEP_CAP:,}]",
        plateau_t=plateau,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot2_resvar_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Train + Val side by side (1×2 panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    it = load_json(IJEPA_TRAIN)
    iv = load_json(IJEPA_VAL)
    tt = load_json(TJEPA_TRAIN)
    tv = load_json(TJEPA_VAL)

    plateau_train = _plateau_level(tt["residual_var"])
    plateau_val   = _plateau_level(tv["residual_var"])

    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(16, 4.5), dpi=FIG_DPI)

    draw_resvar_dual(
        ax_tr,
        tt["steps"], tt["residual_var"], "T-JEPA (train)", C_TJEPA_TRAIN, "-", LW,
        it["steps"], it["residual_var"], "I-JEPA (train)", C_IJEPA_TRAIN, "-", LW,
        title="Train",
        plateau_t=plateau_train,
    )
    draw_resvar_dual(
        ax_val,
        tv["steps"], tv["residual_var"], "T-JEPA (val)", C_TJEPA_VAL, "--", LW_VAL,
        iv["steps"], iv["residual_var"], "I-JEPA (val)", C_IJEPA_VAL, "--", LW_VAL,
        title="Validation",
        plateau_t=plateau_val,
    )

    fig.suptitle(
        f"Argument I — Residual Variance (Prop. 4.12)  [steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot3_resvar_all.png"
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
    print(f"\nDone — 3 plots saved to {OUT_DIR.name}/")


if __name__ == "__main__":
    main()