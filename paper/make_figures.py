"""Generate the data figures for the snn_reset manuscript from committed results.

Self-contained: reads the validated grid and exhaustive relearning summaries and
writes fig1 (design schematic), fig2 (readout validation), and fig3 (dissociation)
into figures/. Figures 4 and 5 are produced by the experiment's manifold analysis
and copied in separately.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ROOT = HERE.parent
RESULTS = ROOT / "experiments" / "snn_reset" / "results"
# Pinned to specific committed result runs so the figures are reproducible.
VALIDATED = RESULTS / "snn_reset_validated_grid_20260606T043336Z" / "all_task_summary.csv"
EXHAUSTIVE = RESULTS / "snn_reset_relearning_exhaustive_20260606T070543Z" / "all_task_summary.csv"
MANIFOLD = RESULTS / "manifold_analysis_20260613T224052Z"

C = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
     "green": "#009E73", "gray": "#6B7280", "black": "#111827", "light": "#E5E7EB"}
TASK_LABEL = {"conditioned_electrode_association": "Association", "pattern_discrimination": "Pattern"}
TASK_COLOR = {"Association": C["blue"], "Pattern": C["orange"]}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def box(ax, x, y, w, h, text, face, weight="bold"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                         linewidth=0.9, edgecolor="#D1D5DB", facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5,
            weight=weight, color=C["black"], linespacing=1.25)


def arrow(ax, start, end, color="#4B5563"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=2, shrinkB=2))


def fig1_design() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(False)
    xs = np.linspace(0.085, 0.915, 6)
    labels = ["naive\nculture", "train to\ncriterion", "clone full\ntrained state",
              "reset vs\nmatched rest", "post-test\nscore", "relearn\n(savings)"]
    faces = ["#F3F4F6", "#DBEAFE", "#DCFCE7", "#FEE2E2", "#FEF3C7", "#EDE9FE"]
    for i, (x, lab, fc) in enumerate(zip(xs, labels, faces)):
        box(ax, x - 0.072, 0.40, 0.144, 0.30, lab, fc)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.072, 0.55), (xs[i + 1] - 0.072, 0.55))
    ax.text(0.5, 0.90, "Paired-clone reset assay", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.13, "Stimulation enters as 64-channel electrode pulses; behaviour is read as channel-level activity. "
            "The full weight vector is observed for analysis but is never used by the actuator or readout.",
            ha="center", fontsize=7.6, color=C["gray"])
    save(fig, "fig1_design")


def fig2_readout() -> None:
    df = pd.read_csv(VALIDATED)
    df = df[df["task_name"].isin(TASK_LABEL)].copy()
    df["task"] = df["task_name"].map(TASK_LABEL)
    agg = df.groupby("task")[["baseline_score", "trained_score", "naive_weight_control_score"]].mean()
    conds = ["baseline_score", "trained_score", "naive_weight_control_score"]
    names = ["baseline", "trained", "naive-weight\ncontrol"]
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    width = 0.36
    x = np.arange(len(conds))
    for i, task in enumerate(agg.index):
        ax.bar(x + (i - 0.5) * width, agg.loc[task, conds].to_numpy(), width,
               label=task, color=TASK_COLOR[task])
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("task score (positive - negative)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color=C["black"], lw=0.8, alpha=0.5)
    ax.set_title("Readout is weight-dependent")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig2_readout")


def fig3_dissociation() -> None:
    df = pd.read_csv(EXHAUSTIVE)
    df = df[df["task_name"].isin(TASK_LABEL)].copy()
    df["task"] = df["task_name"].map(TASK_LABEL)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), constrained_layout=True)
    for task, sub in df.groupby("task"):
        col = TASK_COLOR[task]
        axes[0].scatter(sub["reset_minus_no_reset_weight_norm"], sub["reset_score"],
                        s=18, alpha=0.7, color=col, label=task)
        axes[1].scatter(sub["erasure_projection_reset_vs_no_reset"], sub["reset_score"],
                        s=18, alpha=0.7, color=col)
        axes[2].scatter(sub["reset_minus_no_reset_weight_norm"], sub["relearn_savings"],
                        s=18, alpha=0.7, color=col)
    axes[0].axhline(0.0, color=C["vermillion"], ls="--", lw=1.0, alpha=0.8)
    axes[0].text(axes[0].get_xlim()[1], 0.0, "naive-weight control ", color=C["vermillion"],
                 fontsize=7, va="bottom", ha="right")
    axes[0].set_xlabel("synaptic displacement\n‖W$_{reset}$ - W$_{no\\,reset}$‖")
    axes[0].set_ylabel("post-reset task score")
    axes[0].set_title("Displacement does not erase")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].axvline(0.0, color=C["black"], ls=":", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("projection onto memory axis")
    axes[1].set_ylabel("post-reset task score")
    axes[1].set_title("Movement is off-axis")
    axes[2].axhline(1.0, color=C["black"], ls="--", lw=0.9, alpha=0.7)
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].set_xlabel("synaptic displacement\n‖W$_{reset}$ - W$_{no\\,reset}$‖")
    axes[2].set_ylabel("relearning savings")
    axes[2].set_title("No functional reset")
    save(fig, "fig3_dissociation")


def copy_manifold_figures() -> None:
    shutil.copyfile(MANIFOLD / "F1_memory_axis_breaking_point.pdf", FIG / "fig4_breaking_point.pdf")
    shutil.copyfile(MANIFOLD / "F2_surgical_ablations.pdf", FIG / "fig5_ablations.pdf")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    fig1_design()
    fig2_readout()
    fig3_dissociation()
    copy_manifold_figures()
    print("wrote:", *(p.name for p in sorted(FIG.glob("*.pdf"))))


if __name__ == "__main__":
    main()
