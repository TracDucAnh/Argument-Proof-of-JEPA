# irr_var.py
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent.resolve()

IJEPA_TRAIN = HERE / "I-JEPA_arg3_train.json"
IJEPA_VAL   = HERE / "I-JEPA_arg3_val.json"
TJEPA_TRAIN = HERE / "T-JEPA_arg3_train.json"
TJEPA_VAL   = HERE / "T-JEPA_arg3_val.json"

OUT_FILE = HERE / "plot_irred_var.png"

STEP_CAP         = None    # None = vẽ tất cả steps
INSTABILITY_STEP = 9_340

C_IJEPA_TRAIN      = "#60A5FA"
C_IJEPA_VAL        = "#2563EB"
C_TJEPA_TRAIN      = "#F4A07A"
C_TJEPA_VAL        = "#D85A30"
C_INSTABILITY      = "#DC2626"
C_INSTABILITY_FILL = "#FCA5A5"

FONT_TITLE  = 17
FONT_LABEL  = 14
FONT_TICK   = 12
FONT_LEGEND = 9
FONT_ANNOT  = 12
FIG_DPI     = 300

LW_TRAIN       = 1.4
LW_VAL         = 2.2
ALPHA_TRAIN    = 0.80
ALPHA_VAL      = 1.0
ALPHA_GRID     = 0.25
ALPHA_INSTAB   = 0.15

FIRSTSTEP = 1700


def load_irred(path: Path, step_cap=STEP_CAP) -> tuple[list, list]:
    """Load global_step + irred_var từ JSON. Nếu step_cap=None thì lấy tất cả."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if step_cap is not None:
        records = [r for r in records if r["global_step"] <= step_cap]
    steps = [r["global_step"] for r in records]
    vals  = [r["irred_var"]   for r in records]
    return steps, vals


def _get_near(steps, vals, target_step):
    pairs = [(s, v) for s, v in zip(steps, vals) if v is not None]
    if not pairs:
        return None, None
    return min(pairs, key=lambda x: abs(x[0] - target_step))


def _get_final(steps, vals):
    pairs = [(s, v) for s, v in zip(steps, vals) if v is not None]
    return pairs[-1] if pairs else (None, None)


def _annotate_point(ax, step, val, color,
                    label_prefix="",
                    text_offset=(50, 18),
                    ha="left",
                    fontsize=FONT_ANNOT):
    if step is None or val is None:
        return
    ax.scatter([step], [val], color=color, s=70, zorder=10,
               edgecolors="white", linewidths=1.2, label="_nolegend_")
    ax.annotate(
        f"{label_prefix}{val:.4f}",
        xy=(step, val),
        xytext=(text_offset[0], text_offset[1]),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha=ha,
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88,
                  ec=color, linewidth=1.1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                        connectionstyle="arc3,rad=0.15"),
        zorder=11,
    )


def _add_instability(ax, x_max):
    """Dải đỏ + đường dashed cho T-JEPA instability."""
    ax.axvspan(INSTABILITY_STEP, x_max,
               color=C_INSTABILITY_FILL, alpha=ALPHA_INSTAB, zorder=0)
    ax.axvline(x=INSTABILITY_STEP,
               color=C_INSTABILITY, linewidth=1.5, linestyle="--", zorder=3,
               label=f"T-JEPA instability (step {INSTABILITY_STEP:,})")


def _style_common(ax, title, ylabel, legend_loc="upper right"):
    ax.set_xlabel("Training steps", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=ALPHA_GRID, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)   # không giới hạn right
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.88, loc=legend_loc,
              borderpad=0.5, labelspacing=0.35, handlelength=1.5)


def main():
    si_t, vi_t = load_irred(IJEPA_TRAIN)
    si_v, vi_v = load_irred(IJEPA_VAL)
    st_t, vt_t = load_irred(TJEPA_TRAIN)
    st_v, vt_v = load_irred(TJEPA_VAL)

    # Xác định step lớn nhất thực tế để vẽ vspan
    all_steps = si_t + si_v + st_t + st_v
    x_max = max(all_steps) if all_steps else INSTABILITY_STEP

    fig, (ax_log, ax_raw) = plt.subplots(
        1, 2, figsize=(20, 6), dpi=FIG_DPI
    )
    fig.subplots_adjust(wspace=0.32)

    # ═══════════════════════════════════════════════════
    # PANEL TRÁI — Log scale
    # ═══════════════════════════════════════════════════
    ax_log.plot(si_t, vi_t, color=C_IJEPA_TRAIN, linewidth=LW_TRAIN,
                alpha=ALPHA_TRAIN, linestyle="-", label="I-JEPA train")
    ax_log.plot(si_v, vi_v, color=C_IJEPA_VAL,   linewidth=LW_VAL,
                alpha=ALPHA_VAL,   linestyle="-", label="I-JEPA val")
    ax_log.plot(st_t, vt_t, color=C_TJEPA_TRAIN, linewidth=LW_TRAIN,
                alpha=ALPHA_TRAIN, linestyle="-", label="T-JEPA train")
    ax_log.plot(st_v, vt_v, color=C_TJEPA_VAL,   linewidth=LW_VAL,
                alpha=ALPHA_VAL,   linestyle="-", label="T-JEPA val")

    ax_log.set_yscale("log")
    _add_instability(ax_log, x_max)

    def _mid_step(steps):
        valid = [s for s in steps if s is not None]
        if not valid:
            return None
        return valid[len(valid) // 2]
        # return 12500

    # Annotations: đầu / giữa / cuối — I-JEPA train
    s0_it, v0_it = _get_near(si_t, vi_t, FIRSTSTEP)
    sm_it, vm_it = _get_near(si_t, vi_t, _mid_step(si_t))
    sf_it, vf_it = _get_final(si_t, vi_t)
    _annotate_point(ax_log, s0_it, v0_it, C_IJEPA_TRAIN,
                    text_offset=(50, -38), ha="left")
    _annotate_point(ax_log, sm_it, vm_it, C_IJEPA_TRAIN,
                    text_offset=(50,  -100), ha="left")
    _annotate_point(ax_log, sf_it, vf_it, C_IJEPA_TRAIN,
                    text_offset=(-70, 100), ha="right")

    # I-JEPA val
    s0_iv, v0_iv = _get_near(si_v, vi_v, FIRSTSTEP)
    sm_iv, vm_iv = _get_near(si_v, vi_v, _mid_step(si_v))
    sf_iv, vf_iv = _get_final(si_v, vi_v)
    _annotate_point(ax_log, s0_iv, v0_iv, C_IJEPA_VAL,
                    text_offset=(50, 30), ha="left")
    _annotate_point(ax_log, sm_iv, vm_iv, C_IJEPA_VAL,
                    text_offset=(50, -60), ha="left")
    _annotate_point(ax_log, sf_iv, vf_iv, C_IJEPA_VAL,
                    text_offset=(-70, 60), ha="right")

    # T-JEPA train
    s0_tt, v0_tt = _get_near(st_t, vt_t, FIRSTSTEP)
    sm_tt, vm_tt = _get_near(st_t, vt_t, _mid_step(st_t))
    sf_tt, vf_tt = _get_final(st_t, vt_t)
    _annotate_point(ax_log, s0_tt, v0_tt, C_TJEPA_TRAIN,
                    text_offset=(50, -32), ha="left")
    _annotate_point(ax_log, sm_tt, vm_tt, C_TJEPA_TRAIN,
                    text_offset=(-50, 60), ha="right")
    _annotate_point(ax_log, sf_tt, vf_tt, C_TJEPA_TRAIN,
                    text_offset=(-70, -50), ha="right")

    # T-JEPA val
    s0_tv, v0_tv = _get_near(st_v, vt_v, FIRSTSTEP)
    sm_tv, vm_tv = _get_near(st_v, vt_v, _mid_step(st_v))
    sf_tv, vf_tv = _get_final(st_v, vt_v)
    _annotate_point(ax_log, s0_tv, v0_tv, C_TJEPA_VAL,
                    text_offset=(50, 32), ha="left")
    _annotate_point(ax_log, sm_tv, vm_tv, C_TJEPA_VAL,
                    text_offset=(-50, 30), ha="right")
    _annotate_point(ax_log, sf_tv, vf_tv, C_TJEPA_VAL,
                    text_offset=(-70, -32), ha="right")

    # Label instability
    ylims_log = ax_log.get_ylim()
    ax_log.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ylims_log[1]),
        xytext=(INSTABILITY_STEP - 350, ylims_log[1]),
        fontsize=FONT_ANNOT - 1,
        color=C_INSTABILITY,
        va="top",
        ha="right",
    )

    _style_common(ax_log,
                  title="Irreducible Variance — Log Scale",
                  ylabel="Irred. Var (log scale)",
                  legend_loc="upper right")

    # ═══════════════════════════════════════════════════
    # PANEL PHẢI — Raw / linear scale
    # ═══════════════════════════════════════════════════
    ax_raw.plot(si_t, vi_t, color=C_IJEPA_TRAIN, linewidth=LW_TRAIN,
                alpha=ALPHA_TRAIN, linestyle="-", label="I-JEPA train")
    ax_raw.plot(si_v, vi_v, color=C_IJEPA_VAL,   linewidth=LW_VAL,
                alpha=ALPHA_VAL,   linestyle="-", label="I-JEPA val")
    ax_raw.plot(st_t, vt_t, color=C_TJEPA_TRAIN, linewidth=LW_TRAIN,
                alpha=ALPHA_TRAIN, linestyle="-", label="T-JEPA train")
    ax_raw.plot(st_v, vt_v, color=C_TJEPA_VAL,   linewidth=LW_VAL,
                alpha=ALPHA_VAL,   linestyle="-", label="T-JEPA val")

    _add_instability(ax_raw, x_max)

    ylims_raw = ax_raw.get_ylim()
    ax_raw.annotate(
        f"T-JEPA instability\nstep {INSTABILITY_STEP:,}",
        xy=(INSTABILITY_STEP, ylims_raw[1]),
        xytext=(INSTABILITY_STEP - 350, ylims_raw[1]),
        fontsize=FONT_ANNOT - 1,
        color=C_INSTABILITY,
        va="top",
        ha="right",
    )

    _style_common(ax_raw,
                  title="Irreducible Variance — Raw Scale",
                  ylabel="Irred. Var",
                  legend_loc="upper right")

    # Footer
    fig.text(
        0.5, -0.02,
        "Irreducible variance Var(z* | z_C, p_j) estimated from K=16 completions per context · "
        "N=200 held-out contexts per step · frozen target encoder",
        ha="center", fontsize=11, style="italic", color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()