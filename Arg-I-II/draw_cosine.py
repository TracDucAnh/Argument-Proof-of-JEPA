# draw_cosine.py  — Arg-II: Cosine Similarity Metrics
# ─────────────────────────────────────────────────────────────────────────────
# Vẽ 3 panel cosine similarity cho I-JEPA và T-JEPA (train + val):
#   Panel 0 — Mean pairwise cosine sim theo steps
#   Panel 1 — Std + P95 theo steps (val only)
#   Panel 2 — Histogram overlay: early vs late collapse (val only)
#
# Usage (run từ Arg-I-II/ directory):
#   python draw_cosine.py
#
# Input:  I-JEPA.json, I-JEPA_val.json, T-JEPA.json, T-JEPA_val.json
# Output: cosine.png
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ARG_DIR    = SCRIPT_DIR

I_TRAIN = ARG_DIR / "I-JEPA.json"
I_VAL   = ARG_DIR / "I-JEPA_val.json"
T_TRAIN = ARG_DIR / "T-JEPA.json"
T_VAL   = ARG_DIR / "T-JEPA_val.json"
OUT_PNG = ARG_DIR / "cosine.png"

# ── Colors — identical to draw_loss_compare.py ────────────────────────────────
C_IJEPA_TRAIN      = "#60A5FA"
C_IJEPA_VAL        = "#2563EB"
C_TJEPA_TRAIN      = "#F4A07A"
C_TJEPA_VAL        = "#D85A30"
C_INSTABILITY      = "#DC2626"
C_INSTABILITY_FILL = "#FCA5A5"

# ── Font sizes — identical to draw_loss_compare.py ────────────────────────────
FONT_TITLE  = 18
FONT_LABEL  = 16
FONT_TICK   = 14
FONT_LEGEND = 10
FONT_ANNOT  = 12

# ── Line widths — identical to draw_loss_compare.py ───────────────────────────
LW_TRAIN_FADED = 1.0
ALPHA_TRAIN    = 0.8
LW_VAL_BOLD    = 2.5
ALPHA_VAL      = 1.0
ALPHA_GRID     = 0.25
ALPHA_INSTAB   = 0.15

INSTABILITY_STEP = 9_340
FIG_DPI          = 300


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    if not path.exists():
        print(f"  [WARN] not found: {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract(records: list, key: str):
    steps  = [r["global_step"] for r in records if key in r]
    values = [r[key]           for r in records if key in r]
    return steps, values


def find_record_near(records: list, step: int):
    if not records:
        return None
    return min(records, key=lambda r: abs(r["global_step"] - step))


def hist_to_density(hist: list) -> np.ndarray:
    arr   = np.array(hist, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def add_instability_marker(ax):
    ymin, ymax = ax.get_ylim()
    ax.axvspan(INSTABILITY_STEP, ax.get_xlim()[1],
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3)
    ax.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ymax),
        xytext=(INSTABILITY_STEP - 300, ymax),
        fontsize=FONT_ANNOT-4,
        color=C_INSTABILITY,
        va="top",
        ha="right",
    )


def style_ax(ax, title: str, xlabel: str = "Training steps", ylabel: str = ""):
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Load ──────────────────────────────────────────────────────────────────────
i_train = load_json(I_TRAIN)
i_val   = load_json(I_VAL)
t_train = load_json(T_TRAIN)
t_val   = load_json(T_VAL)

# ── Time-series ───────────────────────────────────────────────────────────────
it_steps, it_cos_mean = extract(i_train, "cosine_sim_mean")
_,        it_cos_std  = extract(i_train, "cosine_sim_std")
_,        it_cos_p95  = extract(i_train, "cosine_sim_p95")

iv_steps, iv_cos_mean = extract(i_val,   "cosine_sim_mean")
_,        iv_cos_std  = extract(i_val,   "cosine_sim_std")
_,        iv_cos_p95  = extract(i_val,   "cosine_sim_p95")

tt_steps, tt_cos_mean = extract(t_train, "cosine_sim_mean")
_,        tt_cos_std  = extract(t_train, "cosine_sim_std")
_,        tt_cos_p95  = extract(t_train, "cosine_sim_p95")

tv_steps, tv_cos_mean = extract(t_val,   "cosine_sim_mean")
_,        tv_cos_std  = extract(t_val,   "cosine_sim_std")
_,        tv_cos_p95  = extract(t_val,   "cosine_sim_p95")

# ── Histogram records ─────────────────────────────────────────────────────────
t_val_with_hist = [r for r in t_val if "cosine_sim_hist" in r]
i_val_with_hist = [r for r in i_val if "cosine_sim_hist" in r]

t_hist_early = t_val_with_hist[0] if t_val_with_hist else None
t_hist_late  = find_record_near(
    [r for r in t_val_with_hist if r["global_step"] >= INSTABILITY_STEP],
    INSTABILITY_STEP
) or (t_val_with_hist[-1] if t_val_with_hist else None)
i_hist_late  = i_val_with_hist[-1] if i_val_with_hist else None

BIN_EDGES   = np.linspace(-1.0, 1.0, 11)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
BIN_WIDTH   = BIN_EDGES[1] - BIN_EDGES[0]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=FIG_DPI)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 0 — Mean cosine similarity
# ─────────────────────────────────────────────────────────────────────────────
ax0 = axes[0]

if it_steps:
    ax0.plot(it_steps, it_cos_mean, color=C_IJEPA_TRAIN, lw=LW_TRAIN_FADED,
             alpha=ALPHA_TRAIN, label="I-JEPA train", zorder=2)
if iv_steps:
    ax0.plot(iv_steps, iv_cos_mean, color=C_IJEPA_VAL, lw=LW_VAL_BOLD,
             alpha=ALPHA_VAL, label="I-JEPA val", zorder=3)
if tt_steps:
    ax0.plot(tt_steps, tt_cos_mean, color=C_TJEPA_TRAIN, lw=LW_TRAIN_FADED,
             alpha=ALPHA_TRAIN, label="T-JEPA train", zorder=2)
if tv_steps:
    ax0.plot(tv_steps, tv_cos_mean, color=C_TJEPA_VAL, lw=LW_VAL_BOLD,
             alpha=ALPHA_VAL, label="T-JEPA val", zorder=3)

ax0.axhline(y=1.0, color="#AAAAAA", lw=0.8, ls=":", zorder=1)
ax0.set_ylim(-0.05, 1.08)

style_ax(ax0,
         title="Mean Cosine Similarity",
         ylabel="Mean Pairwise Cosine Similarity")

add_instability_marker(ax0)

legend_handles_0 = [
    plt.Line2D([0], [0], color=C_TJEPA_TRAIN, lw=LW_TRAIN_FADED, label="T-JEPA train"),
    plt.Line2D([0], [0], color=C_TJEPA_VAL,   lw=LW_VAL_BOLD,    label="T-JEPA val"),
    plt.Line2D([0], [0], color=C_IJEPA_TRAIN, lw=LW_TRAIN_FADED, label="I-JEPA train"),
    plt.Line2D([0], [0], color=C_IJEPA_VAL,   lw=LW_VAL_BOLD,    label="I-JEPA val"),
    plt.Line2D([0], [0], color=C_INSTABILITY,  lw=1.5, ls="--",
               label=f"T-JEPA instability ({INSTABILITY_STEP:,})"),
]
ax0.legend(handles=legend_handles_0, fontsize=FONT_LEGEND, framealpha=0.85,
           loc="lower right", borderpad=0.5, labelspacing=0.3, handlelength=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Std (left axis) + P95 (right axis), val only
# ─────────────────────────────────────────────────────────────────────────────
ax1  = axes[1]
ax1r = ax1.twinx()

if iv_steps:
    ax1.plot(iv_steps, iv_cos_std,  color=C_IJEPA_VAL, lw=LW_VAL_BOLD,
             alpha=ALPHA_VAL, label="I-JEPA std (val)", zorder=3)
    ax1r.plot(iv_steps, iv_cos_p95, color=C_IJEPA_VAL, lw=LW_VAL_BOLD,
              alpha=ALPHA_VAL, ls=":", label="I-JEPA p95 (val)", zorder=3)
if tv_steps:
    ax1.plot(tv_steps, tv_cos_std,  color=C_TJEPA_VAL, lw=LW_VAL_BOLD,
             alpha=ALPHA_VAL, label="T-JEPA std (val)", zorder=3)
    ax1r.plot(tv_steps, tv_cos_p95, color=C_TJEPA_VAL, lw=LW_VAL_BOLD,
              alpha=ALPHA_VAL, ls=":", label="T-JEPA p95 (val)", zorder=3)

ax1r.axhline(y=1.0, color="#AAAAAA", lw=0.8, ls=":", zorder=1)
ax1.set_ylim(bottom=0)
ax1r.set_ylim(0, 1.08)

ax1.grid(True, alpha=ALPHA_GRID, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1.set_xlabel("Training steps", fontsize=FONT_LABEL)
ax1.set_ylabel("Cosine Sim Std  (solid)", fontsize=FONT_LABEL)
ax1r.set_ylabel("Cosine Sim P95  (dotted)", fontsize=FONT_LABEL)
ax1.set_title("Std & P95 of Cosine Similarity  (val only)",
              fontsize=FONT_TITLE, fontweight="bold")
ax1.tick_params(axis="both", labelsize=FONT_TICK)
ax1r.tick_params(axis="y",   labelsize=FONT_TICK)

add_instability_marker(ax1)

legend_handles_1 = [
    plt.Line2D([0], [0], color=C_TJEPA_VAL, lw=LW_VAL_BOLD, ls="-",
               label="T-JEPA std (val)"),
    plt.Line2D([0], [0], color=C_TJEPA_VAL, lw=LW_VAL_BOLD, ls=":",
               label="T-JEPA p95 (val)"),
    plt.Line2D([0], [0], color=C_IJEPA_VAL, lw=LW_VAL_BOLD, ls="-",
               label="I-JEPA std (val)"),
    plt.Line2D([0], [0], color=C_IJEPA_VAL, lw=LW_VAL_BOLD, ls=":",
               label="I-JEPA p95 (val)"),
    plt.Line2D([0], [0], color=C_INSTABILITY, lw=1.5, ls="--",
               label=f"T-JEPA instability ({INSTABILITY_STEP:,})"),
]
ax1.legend(handles=legend_handles_1, fontsize=FONT_LEGEND, framealpha=0.85,
           loc="upper right", borderpad=0.5, labelspacing=0.3, handlelength=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Histogram overlay (val only)
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[2]
ax2.grid(True, axis="y", alpha=ALPHA_GRID, linestyle="--", zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plotted_any = False

if i_hist_late is not None:
    density = hist_to_density(i_hist_late["cosine_sim_hist"])
    ax2.bar(BIN_CENTERS - BIN_WIDTH * 0.3, density, width=BIN_WIDTH * 0.28,
            color=C_IJEPA_VAL, alpha=0.85, zorder=3,
            label=f"I-JEPA late (step {i_hist_late['global_step']})")
    plotted_any = True

if t_hist_early is not None:
    density = hist_to_density(t_hist_early["cosine_sim_hist"])
    ax2.bar(BIN_CENTERS, density, width=BIN_WIDTH * 0.28,
            color=C_TJEPA_TRAIN, alpha=0.90, zorder=3,
            label=f"T-JEPA early (step {t_hist_early['global_step']})")
    plotted_any = True

if t_hist_late is not None:
    density = hist_to_density(t_hist_late["cosine_sim_hist"])
    ax2.bar(BIN_CENTERS + BIN_WIDTH * 0.3, density, width=BIN_WIDTH * 0.28,
            color=C_TJEPA_VAL, alpha=0.95, zorder=3,
            label=f"T-JEPA late (step {t_hist_late['global_step']})")
    plotted_any = True

if not plotted_any:
    ax2.text(0.5, 0.5, "No histogram data found",
             ha="center", va="center", color="#AAAAAA", transform=ax2.transAxes,
             fontsize=FONT_LABEL)

ax2.axvline(x=1.0, color="#AAAAAA", lw=0.8, ls=":", zorder=1)
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Pairwise Cosine Similarity", fontsize=FONT_LABEL)
ax2.set_ylabel("Density",                    fontsize=FONT_LABEL)
ax2.set_title("Cosine Sim Distribution  (val)",
              fontsize=FONT_TITLE, fontweight="bold")
ax2.tick_params(axis="both", labelsize=FONT_TICK)
ax2.legend(fontsize=FONT_LEGEND, framealpha=0.85, loc="upper left",
           borderpad=0.5, labelspacing=0.3, handlelength=1.5)

# ── Suptitle ──────────────────────────────────────────────────────────────────
current_step = max(
    (it_steps[-1] if it_steps else 0),
    (tt_steps[-1] if tt_steps else 0),
)
fig.suptitle(
    f"Arg-II — Cosine Similarity Collapse Indicators  [step {current_step:,}]",
    fontsize=FONT_TITLE + 2, fontweight="bold", y=1.01,
)
fig.tight_layout()

tmp = OUT_PNG.with_suffix(".tmp.png")
fig.savefig(str(tmp), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
tmp.replace(OUT_PNG)

print(f"Saved → {OUT_PNG}")