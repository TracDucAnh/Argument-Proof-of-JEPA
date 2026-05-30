# draw_lambda_min.py
# ─────────────────────────────────────────────────────────────────────────────
# Argument II — Representation Collapse
#
# Vẽ lambda_min và lambda_min_ratio theo steps:
#   plot1_lambda_train.png  — Train (symlog, cùng trục)
#   plot2_lambda_val.png    — Val   (symlog, cùng trục)
#   plot3_lambda_all.png    — 2×2: (train|val) × (lambda_min|ratio)
#
# Giữ nguyên giá trị âm — oscillation quanh 0 là signal của collapse.
# symlog scale: tuyến tính gần 0, log xa 0 → đọc được cả noise lẫn magnitude.
# Chỉ vẽ đến step 15 000.
# Output: ./lambda_min/
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN = HERE / "I-JEPA.json"
TJEPA_TRAIN = HERE / "T-JEPA.json"
IJEPA_VAL   = HERE / "I-JEPA_val.json"
TJEPA_VAL   = HERE / "T-JEPA_val.json"

OUT_DIR = HERE / "lambda_min"
OUT_DIR.mkdir(exist_ok=True)

STEP_CAP         = 15_000
INSTABILITY_STEP = 9_340

# symlog linear threshold — chọn nhỏ hơn tín hiệu thật để noise hiện rõ
# nếu data thật khác order, chỉnh 2 dòng này
LINTHRESH_LM    = 1e-6
LINTHRESH_RATIO = 1e-6

# ── Màu sắc ──────────────────────────────────────────────────────────────────
C_IJEPA_TRAIN      = "#2563EB"   # blue-600
C_IJEPA_VAL        = "#60A5FA"   # blue-400
C_TJEPA_TRAIN      = "#D85A30"   # orange-red
C_TJEPA_VAL        = "#F97316"   # orange-500
C_INSTABILITY      = "#DC2626"   # red-600
C_INSTABILITY_FILL = "#FCA5A5"   # red-300
C_ZERO             = "#9CA3AF"   # gray-400

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE  = 12
FONT_LABEL  = 10
FONT_TICK   = 8
FONT_LEGEND = 9
FONT_ANNOT  = 8
LW_TRAIN    = 1.2
LW_VAL      = 1.8
LW_ZERO     = 0.8
ALPHA_GRID  = 0.25
FIG_DPI     = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    return {
        "steps":            [r["global_step"]      for r in filtered],
        "lambda_min":       [r["lambda_min"]        for r in filtered],
        "lambda_min_ratio": [r["lambda_min_ratio"]  for r in filtered],
    }


def apply_symlog(ax, linthresh: float):
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    # Format tick labels gọn
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: (f"{v:.0e}" if abs(v) >= linthresh * 10 else f"{v:.2g}")
    ))


def add_instability_marker(ax):
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=0.15, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.4, linestyle="--", zorder=3)
    ylim = ax.get_ylim()
    ax.annotate(
        f"step {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ylim[1]),
        xytext=(INSTABILITY_STEP + 200, ylim[1]),
        fontsize=FONT_ANNOT, color=C_INSTABILITY,
        va="top", ha="left",
    )


def add_zero_line(ax):
    ax.axhline(0, color=C_ZERO, linewidth=LW_ZERO, linestyle=":", zorder=1)


def style_ax(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, STEP_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# Core panel: I-JEPA + T-JEPA trên cùng trục, symlog
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(ax,
               s_t, v_t, label_t, color_t, lw_t,
               s_i, v_i, label_i, color_i, lw_i,
               metric: str, title: str):
    """Vẽ 2 model trên cùng 1 trục symlog."""
    linthresh = LINTHRESH_LM if metric == "lambda_min" else LINTHRESH_RATIO
    ylabel = (r"$\lambda_{\min}(\Sigma_z)$"
              if metric == "lambda_min"
              else r"$\lambda_{\min}\ /\ \lambda_{\max}$")

    ax.plot(s_i, v_i, color=color_i, linewidth=lw_i, label=label_i, zorder=3)
    ax.plot(s_t, v_t, color=color_t, linewidth=lw_t, label=label_t,
            alpha=0.85, zorder=2)
    add_zero_line(ax)
    apply_symlog(ax, linthresh)
    add_instability_marker(ax)
    style_ax(ax, title, ylabel)

    instab_patch = plt.Line2D([0], [0], color=C_INSTABILITY, lw=1.4,
                               linestyle="--", label="T-JEPA instability")
    zero_patch   = plt.Line2D([0], [0], color=C_ZERO, lw=LW_ZERO,
                               linestyle=":", label="zero reference")
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.88,
              handles=[ax.lines[0], ax.lines[1], instab_patch, zero_patch])


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Train
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    i = load_json(IJEPA_TRAIN)
    t = load_json(TJEPA_TRAIN)

    fig, (ax_lm, ax_lr) = plt.subplots(1, 2, figsize=(16, 5), dpi=FIG_DPI)

    draw_panel(ax_lm,
               t["steps"], t["lambda_min"],       "T-JEPA (train)", C_TJEPA_TRAIN, LW_TRAIN,
               i["steps"], i["lambda_min"],       "I-JEPA (train)", C_IJEPA_TRAIN, LW_TRAIN,
               "lambda_min", r"$\lambda_{\min}$ — Train")

    draw_panel(ax_lr,
               t["steps"], t["lambda_min_ratio"], "T-JEPA (train)", C_TJEPA_TRAIN, LW_TRAIN,
               i["steps"], i["lambda_min_ratio"], "I-JEPA (train)", C_IJEPA_TRAIN, LW_TRAIN,
               "lambda_min_ratio", r"$\lambda_{\min}/\lambda_{\max}$ — Train")

    fig.suptitle(
        f"Argument II — Representation Collapse  [train, steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "plot1_lambda_train.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val
# ─────────────────────────────────────────────────────────────────────────────

def plot2_val():
    i = load_json(IJEPA_VAL)
    t = load_json(TJEPA_VAL)

    fig, (ax_lm, ax_lr) = plt.subplots(1, 2, figsize=(16, 5), dpi=FIG_DPI)

    draw_panel(ax_lm,
               t["steps"], t["lambda_min"],       "T-JEPA (val)", C_TJEPA_VAL, LW_VAL,
               i["steps"], i["lambda_min"],       "I-JEPA (val)", C_IJEPA_VAL, LW_VAL,
               "lambda_min", r"$\lambda_{\min}$ — Validation")

    draw_panel(ax_lr,
               t["steps"], t["lambda_min_ratio"], "T-JEPA (val)", C_TJEPA_VAL, LW_VAL,
               i["steps"], i["lambda_min_ratio"], "I-JEPA (val)", C_IJEPA_VAL, LW_VAL,
               "lambda_min_ratio", r"$\lambda_{\min}/\lambda_{\max}$ — Validation")

    fig.suptitle(
        f"Argument II — Representation Collapse  [val, steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "plot2_lambda_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — 2×2: rows=metric, cols=train|val
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    it = load_json(IJEPA_TRAIN)
    iv = load_json(IJEPA_VAL)
    tt = load_json(TJEPA_TRAIN)
    tv = load_json(TJEPA_VAL)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=FIG_DPI)
    (ax_lm_tr, ax_lm_val), (ax_lr_tr, ax_lr_val) = axes

    # Row 1 — lambda_min
    draw_panel(ax_lm_tr,
               tt["steps"], tt["lambda_min"], "T-JEPA (train)", C_TJEPA_TRAIN, LW_TRAIN,
               it["steps"], it["lambda_min"], "I-JEPA (train)", C_IJEPA_TRAIN, LW_TRAIN,
               "lambda_min", r"$\lambda_{\min}$ — Train")

    draw_panel(ax_lm_val,
               tv["steps"], tv["lambda_min"], "T-JEPA (val)", C_TJEPA_VAL, LW_VAL,
               iv["steps"], iv["lambda_min"], "I-JEPA (val)", C_IJEPA_VAL, LW_VAL,
               "lambda_min", r"$\lambda_{\min}$ — Validation")

    # Row 2 — lambda_min_ratio
    draw_panel(ax_lr_tr,
               tt["steps"], tt["lambda_min_ratio"], "T-JEPA (train)", C_TJEPA_TRAIN, LW_TRAIN,
               it["steps"], it["lambda_min_ratio"], "I-JEPA (train)", C_IJEPA_TRAIN, LW_TRAIN,
               "lambda_min_ratio", r"$\lambda_{\min}/\lambda_{\max}$ — Train")

    draw_panel(ax_lr_val,
               tv["steps"], tv["lambda_min_ratio"], "T-JEPA (val)", C_TJEPA_VAL, LW_VAL,
               iv["steps"], iv["lambda_min_ratio"], "I-JEPA (val)", C_IJEPA_VAL, LW_VAL,
               "lambda_min_ratio", r"$\lambda_{\min}/\lambda_{\max}$ — Validation")

    fig.suptitle(
        f"Argument II — Representation Collapse  [train + val, steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE + 1, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        "y-axis: symlog scale  (linear near 0, log elsewhere)  —  "
        "shading = post-collapse region (step > 9,340)",
        ha="center", fontsize=7, color="#888",
    )
    fig.tight_layout()
    out = OUT_DIR / "plot3_lambda_all.png"
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