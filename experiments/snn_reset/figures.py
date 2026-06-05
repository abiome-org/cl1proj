"""Paper-ready figures for the SNN reset grid study.

Reads existing CSV artifacts (no new simulations) and emits one PDF per panel
(each figure is standalone, sized for direct inclusion in the manuscript).

Inputs:
    - experiments/snn_reset/results/full_grid_10k_calibrated_20260602T042050Z/
        summary.csv, raw_trials.csv, pareto.csv
    - experiments/snn_reset/results/control_checks_10k_<DATE>/
        trained_controls.csv, reset_vs_no_reset.csv,
        untrained_controls.csv, noise_diagnostics.csv,
        weight_snapshots.npz  (from instrumented control_checks.py)

Outputs (PDF + PNG) into experiments/snn_reset/results/figures_<DATE>/:
    grid_sweep, grid_beta_effect, grid_burden_scaling, grid_pareto_front,
    weights_distance_scatter, weights_delta_histogram,
    weights_pre_vs_delta, weights_norm_bars,
    control_reset_vs_noreset, control_untrained_drift, control_noise_actuator

Usage:
    .venv-uv/bin/python experiments/snn_reset/figures.py \\
        --controls experiments/snn_reset/results/control_checks_10k_20260604
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "experiments" / "snn_reset" / "results"
GRID_DIR = RESULTS_ROOT / "full_grid_10k_calibrated_20260602T042050Z"


def apply_style() -> None:
    rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.2,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


PROTOCOL_ORDER = ["low_burden_0.75s", "mid_burden_1.5s", "high_burden_3s"]
PROTOCOL_LABEL_SHORT = {
    "low_burden_0.75s": "low (0.75 s)",
    "mid_burden_1.5s": "mid (1.5 s)",
    "high_burden_3s": "high (3.0 s)",
}
PROTOCOL_COLOR = {
    "low_burden_0.75s": "#4C72B0",
    "mid_burden_1.5s": "#DD8452",
    "high_burden_3s": "#C44E52",
}
DURATION_COLOR = {0.75: "#4C72B0", 1.5: "#DD8452", 3.0: "#C44E52"}


@dataclass
class Inputs:
    summary: pd.DataFrame
    raw: pd.DataFrame
    pareto: pd.DataFrame
    trained: pd.DataFrame
    reset_vs: pd.DataFrame
    untrained: pd.DataFrame
    noise: pd.DataFrame
    weights: dict[str, np.ndarray] | None


def latest_controls_dir() -> Path:
    candidates = sorted(RESULTS_ROOT.glob("control_checks_10k_*"))
    if not candidates:
        raise FileNotFoundError("No control_checks_10k_* directory under results/")
    return candidates[-1]


def load_inputs(controls_dir: Path) -> Inputs:
    summary = pd.read_csv(GRID_DIR / "summary.csv")
    raw = pd.read_csv(GRID_DIR / "raw_trials.csv")
    pareto = pd.read_csv(GRID_DIR / "pareto.csv")
    trained = pd.read_csv(controls_dir / "trained_controls.csv")
    reset_vs = pd.read_csv(controls_dir / "reset_vs_no_reset.csv")
    untrained = pd.read_csv(controls_dir / "untrained_controls.csv")
    noise = pd.read_csv(controls_dir / "noise_diagnostics.csv")
    npz_path = controls_dir / "weight_snapshots.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            weights = {k: data[k] for k in data.files}
    else:
        weights = None
    return Inputs(summary, raw, pareto, trained, reset_vs, untrained, noise, weights)


# ---------------------------------------------------------------------------
# Grid figures (from raw_trials.csv / summary.csv / pareto.csv)
# ---------------------------------------------------------------------------

def figure_grid_sweep(inp: Inputs, out_dir: Path) -> None:
    """Headline: weight erasure & savings across all 540 protocols."""
    raw = inp.raw.copy()
    summary = inp.summary.copy()

    grouped = (
        raw.groupby("protocol_id")[["weight_erasure", "savings"]]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = ["__".join(c) for c in grouped.columns]
    grouped["weight_erasure__sem"] = (
        grouped["weight_erasure__std"] / np.sqrt(grouped["weight_erasure__count"])
    )
    grouped["savings__sem"] = (
        grouped["savings__std"] / np.sqrt(grouped["savings__count"])
    )
    grouped = grouped.reset_index()
    grouped = grouped.merge(
        summary[["protocol_id", "duration_s"]], on="protocol_id"
    )
    grouped = grouped.sort_values("weight_erasure__mean", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax2 = ax.twinx()
    ax.grid(False); ax2.grid(False)

    rank = grouped["rank"].to_numpy()
    we = grouped["weight_erasure__mean"].to_numpy()
    we_err = grouped["weight_erasure__sem"].to_numpy()
    sv = grouped["savings__mean"].to_numpy()
    sv_err = grouped["savings__sem"].to_numpy()
    dur = grouped["duration_s"].to_numpy()
    colors = [DURATION_COLOR[d] for d in dur]

    ax.scatter(rank, we, c=colors, s=20, edgecolor="none", zorder=3)
    ax.errorbar(rank, we, yerr=we_err, fmt="none", ecolor="0.7", elinewidth=0.4, alpha=0.5, zorder=2)
    ax2.scatter(rank, sv, marker="v", s=14, facecolor="none", edgecolor="#2C3E50",
                alpha=0.55, linewidths=0.7, zorder=4)
    ax2.errorbar(rank, sv, yerr=sv_err, fmt="none", ecolor="0.85", elinewidth=0.3, alpha=0.5, zorder=1)

    ax.axhline(0.0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax2.axhline(0.0, color="#2C3E50", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Protocol  (n = 540, sorted by weight erasure)")
    ax.set_ylabel("Weight erasure  (1 = full erase,  <0 = drift away from naive)")
    ax2.set_ylabel("Savings  (▽; 1 = perfect retention,  <0 = interference)", color="#2C3E50")
    ax2.tick_params(axis="y", colors="#2C3E50")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DURATION_COLOR[d],
                   markersize=8, label=f"{d:g} s")
        for d in [0.75, 1.5, 3.0]
    ]
    handles.append(
        plt.Line2D([0], [0], marker="v", color="#2C3E50", markerfacecolor="none",
                   linestyle="none", markersize=7, label="savings  (right axis)")
    )
    ax.legend(
        handles=handles, frameon=False, loc="lower left", fontsize=8,
        title="protocol duration  (color)", title_fontsize=8, ncol=2,
    )
    fig.tight_layout()
    save(fig, out_dir, "grid_sweep")


def figure_grid_beta_effect(inp: Inputs, out_dir: Path) -> None:
    """Weight erasure as a function of noise spectral slope beta."""
    raw = inp.raw.copy()
    betas = sorted(raw["beta"].unique())
    data = [raw.loc[raw["beta"] == b, "weight_erasure"].to_numpy() for b in betas]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bp = ax.boxplot(
        data,
        positions=range(len(betas)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black"),
    )
    for patch, b in zip(bp["boxes"], betas):
        patch.set_facecolor(plt.cm.coolwarm((b + 2) / 4))
        patch.set_alpha(0.75)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:+g}" for b in betas])
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Noise spectral slope  β   (violet → red)")
    ax.set_ylabel("Weight erasure")
    fig.tight_layout()
    save(fig, out_dir, "grid_beta_effect")


def figure_grid_burden_scaling(inp: Inputs, out_dir: Path) -> None:
    """Weight erasure against stimulation burden (duration × current)."""
    raw = inp.raw.copy()
    raw["burden"] = raw["duration_s"] * raw["current_uA"]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for spatial, marker in zip(
        ["shared", "independent", "correlated", "phase_shifted"],
        ["o", "s", "^", "D"],
    ):
        sub = raw[raw["spatial_mode"] == spatial]
        ax.scatter(
            sub["burden"], sub["weight_erasure"],
            s=14, marker=marker, alpha=0.5, label=spatial,
        )
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Stimulation burden   (duration × current,  s·µA)")
    ax.set_ylabel("Weight erasure")
    ax.legend(title="spatial mode", frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "grid_burden_scaling")


def figure_grid_pareto_front(inp: Inputs, out_dir: Path) -> None:
    """Pareto-nondominated protocols in (weight erasure, trace AUC) plane."""
    summary = inp.summary.copy()
    pareto = inp.pareto.copy()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(
        summary["weight_erasure"], summary["trace_auc_proxy"],
        s=10, color="0.7", alpha=0.6, label="screened (540)",
    )
    ax.scatter(
        pareto["weight_erasure"], pareto["trace_auc_proxy"],
        s=80, facecolor="none", edgecolor="#C44E52", lw=1.5, label="Pareto front",
    )
    for _, row in pareto.iterrows():
        ax.annotate(
            row["protocol_id"].split("_independent_")[-1],
            xy=(row["weight_erasure"], row["trace_auc_proxy"]),
            xytext=(8, -3), textcoords="offset points",
            fontsize=8, color="#C44E52",
        )
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(0.5, color="black", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Weight erasure")
    ax.set_ylabel("Trace AUC proxy   (0.5 = chance)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "grid_pareto_front")


# ---------------------------------------------------------------------------
# Weight-change figures (from trained_controls.csv + weight_snapshots.npz)
# ---------------------------------------------------------------------------

def figure_weights_distance_scatter(inp: Inputs, out_dir: Path) -> None:
    """Distance-to-naive vs distance-to-trained for post-protocol weights."""
    trained = inp.trained.copy()

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for protocol in PROTOCOL_ORDER:
        for mode, marker, edge in zip(
            ["reset", "no_reset"], ["o", "D"], ["black", "0.4"]
        ):
            sub = trained[
                (trained["source_protocol_id"] == protocol)
                & (trained["control_mode"] == mode)
            ]
            ax.scatter(
                sub["post_delta_norm"],
                sub["post_minus_trained_norm"],
                color=PROTOCOL_COLOR[protocol],
                marker=marker,
                s=60 if mode == "reset" else 30,
                alpha=0.85,
                label=f"{PROTOCOL_LABEL_SHORT[protocol]} – {mode}",
                edgecolor=edge,
                linewidths=0.5,
            )
        sub = trained[
            (trained["source_protocol_id"] == protocol)
            & (trained["control_mode"] == "reset")
        ]
        ax.scatter(
            sub["trained_delta_norm"], np.zeros(len(sub)),
            color=PROTOCOL_COLOR[protocol], marker="*", s=130,
            edgecolor="black", linewidths=0.5,
        )
    ax.set_xlabel("‖W$_\\text{post}$ − W$_\\text{naive}$‖   (distance to naive)")
    ax.set_ylabel("‖W$_\\text{post}$ − W$_\\text{trained}$‖   (distance to trained)")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls="--", alpha=0.7)
    ax.text(hi, hi, "  equidistant", color="0.55", fontsize=7, ha="left", va="top")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, "weights_distance_scatter")


def figure_weights_delta_histogram(inp: Inputs, out_dir: Path) -> None:
    """Per-synapse Δw density for each protocol (reset arm)."""
    weights = inp.weights
    if not weights:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for protocol in PROTOCOL_ORDER:
        deltas = []
        for post_key in [
            k for k in weights
            if f"reset__{protocol}__seed" in k and k.endswith("__post")
        ]:
            trained_key = post_key.replace("__post", "__trained")
            if trained_key in weights:
                deltas.append(weights[post_key] - weights[trained_key])
        if not deltas:
            continue
        delta_vec = np.concatenate(deltas)
        ax.hist(
            delta_vec, bins=140, histtype="step",
            color=PROTOCOL_COLOR[protocol],
            label=PROTOCOL_LABEL_SHORT[protocol],
            density=True, linewidth=1.4,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Δw  =  w$_\\text{post}$ − w$_\\text{trained}$   (per synapse)")
    ax.set_ylabel("density  (log scale)")
    ax.legend(fontsize=8, frameon=False, title="protocol")
    fig.tight_layout()
    save(fig, out_dir, "weights_delta_histogram")


def figure_weights_pre_vs_delta(inp: Inputs, out_dir: Path) -> None:
    """Pre-stimulation weight vs Δw, low-burden protocol (reset arm)."""
    weights = inp.weights
    if not weights:
        return
    pre_all, dw_all = [], []
    for trained_key in [
        k for k in weights
        if "reset__low_burden_0.75s__seed" in k and k.endswith("__trained")
    ]:
        post_key = trained_key.replace("__trained", "__post")
        if post_key not in weights:
            continue
        pre_all.append(weights[trained_key])
        dw_all.append(weights[post_key] - weights[trained_key])
    if not pre_all:
        return
    pre_vec = np.concatenate(pre_all)
    dw_vec = np.concatenate(dw_all)
    rng = np.random.default_rng(0)
    if len(pre_vec) > 100_000:
        idx = rng.choice(len(pre_vec), 100_000, replace=False)
        pre_vec = pre_vec[idx]
        dw_vec = dw_vec[idx]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    hb = ax.hexbin(pre_vec, dw_vec, gridsize=60, cmap="magma", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("log₁₀ count", fontsize=8)
    ax.axhline(0, color="white", lw=0.6, ls="--", alpha=0.7)
    ax.set_xlabel("w$_\\text{trained}$   (per synapse)")
    ax.set_ylabel("Δw  =  w$_\\text{post}$ − w$_\\text{trained}$")
    fig.tight_layout()
    save(fig, out_dir, "weights_pre_vs_delta")


def figure_weights_norm_bars(inp: Inputs, out_dir: Path) -> None:
    """Weight-norm budget per protocol, reset vs no-reset arm."""
    trained = inp.trained.copy()
    width = 0.22
    metrics = ["trained_delta_norm", "post_delta_norm", "post_minus_trained_norm"]
    metric_labels = [
        "‖W$_\\text{trained}$ − W$_\\text{naive}$‖",
        "‖W$_\\text{post}$ − W$_\\text{naive}$‖",
        "‖W$_\\text{post}$ − W$_\\text{trained}$‖",
    ]
    x_base = np.arange(len(PROTOCOL_ORDER))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, mode in enumerate(["reset", "no_reset"]):
        offset = (j - 0.5) * (width * len(metrics) + 0.08)
        for i, metric in enumerate(metrics):
            heights, errs = [], []
            for protocol in PROTOCOL_ORDER:
                sub = trained[
                    (trained["source_protocol_id"] == protocol)
                    & (trained["control_mode"] == mode)
                ]
                heights.append(sub[metric].mean())
                errs.append(sub[metric].std() / np.sqrt(max(len(sub), 1)))
            xs = x_base + offset + (i - 1) * width
            color = (
                plt.cm.Reds(0.4 + 0.2 * i) if mode == "reset"
                else plt.cm.Greys(0.35 + 0.25 * i)
            )
            ax.bar(
                xs, heights, width=width * 0.95, yerr=errs, capsize=2,
                color=color, edgecolor="black", linewidth=0.4,
            )
    ax.set_xticks(x_base)
    ax.set_xticklabels([PROTOCOL_LABEL_SHORT[p] for p in PROTOCOL_ORDER])
    ax.set_ylabel("Weight-vector norm")

    arm_handles = [
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Reds(0.5), label="reset arm"),
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Greys(0.5), label="no-reset arm"),
    ]
    metric_handles = [
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Reds(0.4 + 0.2 * i), label=metric_labels[i])
        for i in range(len(metric_labels))
    ]
    leg1 = ax.legend(handles=arm_handles, loc="upper left", frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(
        handles=metric_handles, loc="upper right", frameon=False, fontsize=7,
        title="bar shade →", title_fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, out_dir, "weights_norm_bars")


# ---------------------------------------------------------------------------
# Control-arm figures
# ---------------------------------------------------------------------------

def figure_control_reset_vs_noreset(inp: Inputs, out_dir: Path) -> None:
    """Reset − no-reset deltas across seven metrics."""
    reset_vs = inp.reset_vs.copy()
    delta_cols = [
        "weight_erasure_delta",
        "residual_performance_delta",
        "savings_delta",
        "trace_auc_proxy_delta",
        "health_delta",
        "post_delta_norm_delta",
        "reset_window_neuron_spikes_delta",
    ]
    nice_labels = [
        "weight erasure",
        "residual perf.",
        "savings",
        "trace AUC",
        "health",
        "‖W$_\\text{post}$−W$_\\text{naive}$‖",
        "neuron spikes",
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, protocol in enumerate(PROTOCOL_ORDER):
        sub = reset_vs[reset_vs["source_protocol_id"] == protocol]
        x_pos = np.arange(len(delta_cols)) + (i - 1) * 0.22
        values = []
        for col in delta_cols:
            v = sub[col].values
            if "spikes" in col:
                norm = np.max(np.abs(reset_vs[col].values)) or 1.0
                v = v / norm
            values.append(v)
        means = [float(np.mean(v)) for v in values]
        sems = [float(np.std(v) / np.sqrt(max(len(v), 1))) for v in values]
        ax.errorbar(
            x_pos, means, yerr=sems, fmt="o",
            color=PROTOCOL_COLOR[protocol], label=PROTOCOL_LABEL_SHORT[protocol],
            markersize=6, capsize=3, elinewidth=0.7,
        )
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(range(len(delta_cols)))
    ax.set_xticklabels(nice_labels, rotation=30, ha="right")
    ax.set_ylabel("Reset − No-reset Δ   (spikes normalized)")
    ax.legend(frameon=False, fontsize=8, title="protocol")
    fig.tight_layout()
    save(fig, out_dir, "control_reset_vs_noreset")


def figure_control_untrained_drift(inp: Inputs, out_dir: Path) -> None:
    """Naive-network drift vs window duration, reset vs no-reset."""
    untrained = inp.untrained.copy()
    pivot = (
        untrained.groupby(["source_protocol_id", "control_mode"], as_index=False)
        .agg(
            drift_mean=("weight_drift_rel_to_baseline", "mean"),
            drift_sem=(
                "weight_drift_rel_to_baseline",
                lambda s: s.std() / np.sqrt(max(len(s), 1)),
            ),
            duration_s=("duration_s", "first"),
        )
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for mode, marker in zip(["reset", "no_reset"], ["o", "x"]):
        sub = pivot[pivot["control_mode"] == mode].sort_values("duration_s")
        ax.errorbar(
            sub["duration_s"], sub["drift_mean"], yerr=sub["drift_sem"],
            marker=marker, label=mode,
            color="#C44E52" if mode == "reset" else "0.4",
            markersize=8, capsize=3, linestyle="-",
        )
    ax.set_xlabel("Protocol-window duration   (s)")
    ax.set_ylabel("Weight drift   ‖ΔW‖ / ‖W$_\\text{baseline}$‖")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "control_untrained_drift")


def figure_control_noise_actuator(inp: Inputs, out_dir: Path) -> None:
    """Colored-noise generator: estimated vs target spectral slope."""
    noise = inp.noise.copy()
    noise_per_beta = (
        noise.groupby("beta", as_index=False)
        .agg(
            target=("target_power_slope", "first"),
            est_mean=("estimated_power_slope_mean", "first"),
            est_std=("estimated_power_slope_std", "first"),
        )
        .sort_values("beta")
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(
        noise_per_beta["target"], noise_per_beta["est_mean"],
        yerr=noise_per_beta["est_std"],
        fmt="o", markersize=8, color="#4C72B0", capsize=3, elinewidth=0.8,
    )
    lo = min(noise_per_beta["target"].min(), noise_per_beta["est_mean"].min())
    hi = max(noise_per_beta["target"].max(), noise_per_beta["est_mean"].max())
    pad = 0.2
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", ls="--", lw=0.8)
    for _, row in noise_per_beta.iterrows():
        ax.annotate(
            f"β={int(row['beta']):+d}",
            xy=(row["target"], row["est_mean"]),
            xytext=(8, -3), textcoords="offset points",
            fontsize=9, color="#4C72B0",
        )
    ax.set_xlabel("Target spectral slope  −β")
    ax.set_ylabel("Estimated slope   (mean ± std, n=32)")
    fig.tight_layout()
    save(fig, out_dir, "control_noise_actuator")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

FIGURE_BUILDERS = [
    figure_grid_sweep,
    figure_grid_beta_effect,
    figure_grid_burden_scaling,
    figure_grid_pareto_front,
    figure_weights_distance_scatter,
    figure_weights_delta_histogram,
    figure_weights_pre_vs_delta,
    figure_weights_norm_bars,
    figure_control_reset_vs_noreset,
    figure_control_untrained_drift,
    figure_control_noise_actuator,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=None,
                        help="Path to a control_checks_10k_* directory. Default: latest.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory. Default: results/figures_<date>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controls_dir = args.controls or latest_controls_dir()
    out_dir = args.out or (
        RESULTS_ROOT / f"figures_{controls_dir.name.split('_')[-1]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    inp = load_inputs(controls_dir)

    for builder in FIGURE_BUILDERS:
        builder(inp, out_dir)

    print(f"controls_dir={controls_dir}")
    print(f"out_dir={out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
