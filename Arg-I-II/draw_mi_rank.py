# draw_mi_rank.py
# ─────────────────────────────────────────────────────────────────────────────
# Argument I — Entropy Ceiling
#
# Mỗi figure = 1 row × 2 col:
#   Ảnh trái  : MI proxy  — single-axis, I-JEPA vs T-JEPA
#               + shaded entropy ceiling (cận trên lý thuyết của text MI)
#   Ảnh phải  : Effective rank — dual-axis
#               trục trái T-JEPA (cam đỏ), trục phải I-JEPA (xanh dương)
#
# 3 output (train / val / train+val):
#   mi_rank/plot1_mi_rank_train.png
#   mi_rank/plot2_mi_rank_val.png
#   mi_rank/plot3_mi_rank_all.png
#
# Chỉ vẽ đến step 15 000.
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN = HERE / "I-JEPA.json"
TJEPA_TRAIN = HERE / "T-JEPA.json"
IJEPA_VAL   = HERE / "I-JEPA_val.json"
TJEPA_VAL   = HERE / "T-JEPA_val.json"

OUT_DIR  = HERE / "mi_rank"
OUT_DIR.mkdir(exist_ok=True)

STEP_CAP         = 15_000
INSTABILITY_STEP = 9_340   # T-JEPA bắt đầu mất ổn định

# ── Màu sắc (nhất quán với toàn bộ project) ──────────────────────────────────
C_IJEPA_TRAIN = "#60A5FA"    # blue-400
C_IJEPA_VAL   = "#2563EB"  # blue-600
C_TJEPA_TRAIN = "#F4A07A"   # light orange
C_TJEPA_VAL   = "#D85A30"  # orange-red

C_CEILING      = "#FDE68A"   # amber-200 — shaded entropy ceiling
C_CEILING_EDGE = "#F59E0B"   # amber-400

C_INSTABILITY      = "#DC2626"   # red-600  — dashed line
C_INSTABILITY_FILL = "#FCA5A5"   # red-300  — shaded region

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_TITLE  = 18
FONT_LABEL  = 16
FONT_TICK   = 14
FONT_LEGEND = 7.5       # smaller legend font
FONT_ANNOT  = 10
LW          = 1.7
LW_VAL      = 2
ALPHA_GRID  = 0.25
ALPHA_SHADE = 1.0
ALPHA_INSTAB = 0.15
FIG_DPI     = 300

# ── Plot 3 specific line weights ──────────────────────────────────────────────
# val lines are bold, train lines are faded/thin
LW_TRAIN_FADED = 1.0        # thin
ALPHA_TRAIN    = 0.7       # very transparent
LW_VAL_BOLD    = 2.5        # thick
ALPHA_VAL      = 1.0        # fully opaque

# ── Entropy ceiling ───────────────────────────────────────────────────────────
CEILING_HI = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path, step_cap: int = STEP_CAP) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    filtered = [r for r in records if r["global_step"] <= step_cap]
    return {
        "steps":          [r["global_step"]    for r in filtered],
        "mi_proxy":       [r["mi_proxy"]        for r in filtered],
        "effective_rank": [r["effective_rank"]  for r in filtered],
    }


def add_instability_marker(ax, label: bool = True):
    """Thêm dashed line đỏ + shaded region từ INSTABILITY_STEP đến STEP_CAP."""
    ax.axvspan(INSTABILITY_STEP, STEP_CAP,
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3,
               label=f"T-JEPA instability (step {INSTABILITY_STEP:,})" if label else None)
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
# Panel trái: MI proxy — single axis + entropy ceiling shading
# ─────────────────────────────────────────────────────────────────────────────

def draw_mi_panel(ax, entries: list[dict], title: str, legend_loc: str = "upper left"):
    for e in entries:
        ax.plot(e["steps"], e["mi_proxy"],
                color=e["color"], linewidth=e["lw"],
                linestyle=e["ls"], label=e["label"],
                alpha=e.get("alpha", 1.0))

    # Shaded entropy ceiling
    ax.axhspan(0, CEILING_HI,
               alpha=ALPHA_SHADE, color=C_CEILING,
               linewidth=0.8, edgecolor=C_CEILING_EDGE)
    x_lo = entries[0]["steps"][0] if entries else 0
    x_hi = max(e["steps"][-1] for e in entries) if entries else STEP_CAP
    ax.annotate(
        "Entropy ceiling\n(text MI upper bound,\nlexical ambiguity 3–5 bits)",
        xy=(x_lo + (x_hi - x_lo) * 0.02, CEILING_HI * 0.55),
        fontsize=FONT_ANNOT, color="#92400E",
        va="center", style="italic",
    )

    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_ylabel("MI Proxy — InfoNCE lower bound", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")

    add_instability_marker(ax, label=True)

    ceiling_patch = mpatches.Patch(
        facecolor=C_CEILING, edgecolor=C_CEILING_EDGE,
        alpha=0.7, label=f"Entropy ceiling (≤ {CEILING_HI})"
    )
    instab_line = plt.Line2D([0], [0], color=C_INSTABILITY, linewidth=1.5,
                              linestyle="--",
                              label=f"T-JEPA instability ({INSTABILITY_STEP:,})")
    handles = [plt.Line2D([0], [0], color=e["color"], linewidth=e["lw"],
                           linestyle=e["ls"], label=e["label"],
                           alpha=e.get("alpha", 1.0)) for e in entries]
    ax.legend(handles=handles + [ceiling_patch, instab_line],
              fontsize=FONT_LEGEND, framealpha=0.85, loc=legend_loc,
              borderpad=0.5, labelspacing=0.3, handlelength=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Panel phải: Effective rank — dual axis
# ─────────────────────────────────────────────────────────────────────────────

def draw_rank_panel(ax, t_entries: list[dict], i_entries: list[dict],
                    title: str, color_t: str, color_i: str,
                    legend_loc: str = "upper right"):
    lines_t = []
    for e in t_entries:
        ln, = ax.plot(e["steps"], e["effective_rank"],
                      color=e["color"], linewidth=e["lw"],
                      linestyle=e["ls"], label=e["label"],
                      alpha=e.get("alpha", 1.0))
        lines_t.append(ln)

    ax.set_ylabel("T-JEPA Effective Rank", fontsize=FONT_LABEL, color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t, labelsize=FONT_TICK)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines["left"].set_edgecolor(color_t)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--")
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")

    ax2 = ax.twinx()
    lines_i = []
    for e in i_entries:
        ln, = ax2.plot(e["steps"], e["effective_rank"],
                       color=e["color"], linewidth=e["lw"],
                       linestyle=e["ls"], label=e["label"],
                       alpha=e.get("alpha", 1.0))
        lines_i.append(ln)

    ax2.set_ylabel("I-JEPA Effective Rank", fontsize=FONT_LABEL, color=color_i)
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
# Plot 1 — Train
# ─────────────────────────────────────────────────────────────────────────────

def plot1_train():
    i = load_json(IJEPA_TRAIN)
    t = load_json(TJEPA_TRAIN)

    fig, (ax_mi, ax_rk) = plt.subplots(1, 2, figsize=(16, 4.5), dpi=FIG_DPI)

    draw_mi_panel(ax_mi, [
        dict(steps=i["steps"], mi_proxy=i["mi_proxy"],
             label="I-JEPA (train)", color=C_IJEPA_TRAIN, ls="-", lw=LW),
        dict(steps=t["steps"], mi_proxy=t["mi_proxy"],
             label="T-JEPA (train)", color=C_TJEPA_TRAIN, ls="-", lw=LW),
    ], title="MI Proxy — Train")

    draw_rank_panel(ax_rk,
        t_entries=[dict(steps=t["steps"], effective_rank=t["effective_rank"],
                        label="T-JEPA (train)", color=C_TJEPA_TRAIN, ls="-", lw=LW)],
        i_entries=[dict(steps=i["steps"], effective_rank=i["effective_rank"],
                        label="I-JEPA (train)", color=C_IJEPA_TRAIN, ls="-", lw=LW)],
        title="Effective Rank — Train (dual axis)",
        color_t=C_TJEPA_TRAIN, color_i=C_IJEPA_TRAIN,
    )

    fig.suptitle(
        f"Argument I — Entropy Ceiling  [train, steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot1_mi_rank_train.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val
# ─────────────────────────────────────────────────────────────────────────────

def plot2_val():
    i = load_json(IJEPA_VAL)
    t = load_json(TJEPA_VAL)

    fig, (ax_mi, ax_rk) = plt.subplots(1, 2, figsize=(16, 4.5), dpi=FIG_DPI)

    draw_mi_panel(ax_mi, [
        dict(steps=i["steps"], mi_proxy=i["mi_proxy"],
             label="I-JEPA (val)", color=C_IJEPA_VAL, ls="--", lw=LW_VAL),
        dict(steps=t["steps"], mi_proxy=t["mi_proxy"],
             label="T-JEPA (val)", color=C_TJEPA_VAL, ls="--", lw=LW_VAL),
    ], title="MI Proxy — Validation")

    draw_rank_panel(ax_rk,
        t_entries=[dict(steps=t["steps"], effective_rank=t["effective_rank"],
                        label="T-JEPA (val)", color=C_TJEPA_VAL, ls="--", lw=LW_VAL)],
        i_entries=[dict(steps=i["steps"], effective_rank=i["effective_rank"],
                        label="I-JEPA (val)", color=C_IJEPA_VAL, ls="--", lw=LW_VAL)],
        title="Effective Rank — Validation (dual axis)",
        color_t=C_TJEPA_VAL, color_i=C_IJEPA_VAL,
    )

    fig.suptitle(
        f"Argument I — Entropy Ceiling  [val, steps 0–{STEP_CAP:,}]",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "plot2_mi_rank_val.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Train + Val
# Val lines: bold & fully opaque
# Train lines: thin & very transparent (alpha=0.30)
# No suptitle; each subplot has its own compact title (no long dashes)
# Smaller legend, repositioned
# ─────────────────────────────────────────────────────────────────────────────

def plot3_all():
    it = load_json(IJEPA_TRAIN)
    iv = load_json(IJEPA_VAL)
    tt = load_json(TJEPA_TRAIN)
    tv = load_json(TJEPA_VAL)

    fig, (ax_mi, ax_rk) = plt.subplots(1, 2, figsize=(18, 5), dpi=FIG_DPI)

    # ── MI Proxy panel ──
    # Train entries: faded/thin  |  Val entries: bold/opaque
    draw_mi_panel(ax_mi, [
        # train — faded, drawn first so val sits on top
        dict(steps=it["steps"], mi_proxy=it["mi_proxy"],
             label="I-JEPA train", color=C_IJEPA_TRAIN,
             ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
        dict(steps=tt["steps"], mi_proxy=tt["mi_proxy"],
             label="T-JEPA train", color=C_TJEPA_TRAIN,
             ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
        # val — bold
        dict(steps=iv["steps"], mi_proxy=iv["mi_proxy"],
             label="I-JEPA val",   color=C_IJEPA_VAL,
             ls="-", lw=LW_VAL_BOLD, alpha=ALPHA_VAL),
        dict(steps=tv["steps"], mi_proxy=tv["mi_proxy"],
             label="T-JEPA val",   color=C_TJEPA_VAL,
             ls="-", lw=LW_VAL_BOLD, alpha=ALPHA_VAL),
    ], title="MI Proxy (Train & Val)", legend_loc="upper left")

    # ── Effective Rank panel ──
    draw_rank_panel(ax_rk,
        t_entries=[
            # train — faded
            dict(steps=tt["steps"], effective_rank=tt["effective_rank"],
                 label="T-JEPA train", color=C_TJEPA_TRAIN,
                 ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
            # val — bold
            dict(steps=tv["steps"], effective_rank=tv["effective_rank"],
                 label="T-JEPA val",   color=C_TJEPA_VAL,
                 ls="-", lw=LW_VAL_BOLD, alpha=ALPHA_VAL),
        ],
        i_entries=[
            # train — faded
            dict(steps=it["steps"], effective_rank=it["effective_rank"],
                 label="I-JEPA train", color=C_IJEPA_TRAIN,
                 ls="-", lw=LW_TRAIN_FADED, alpha=ALPHA_TRAIN),
            # val — bold
            dict(steps=iv["steps"], effective_rank=iv["effective_rank"],
                 label="I-JEPA val",   color=C_IJEPA_VAL,
                 ls="-", lw=LW_VAL_BOLD, alpha=ALPHA_VAL),
        ],
        title="Effective Rank (Train & Val, dual axis)",
        color_t=C_TJEPA_TRAIN, color_i=C_IJEPA_TRAIN,
        legend_loc="upper right",
    )

    # No suptitle — subplot titles carry all the context
    fig.tight_layout()
    out = OUT_DIR / "plot3_mi_rank_all.png"
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