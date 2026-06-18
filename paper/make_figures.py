"""Generate manuscript figures from committed, validated SNN reset results.

The invalidated 2026-06-02 calibrated grid is intentionally not read here.
Figures are pinned to the validated/relearning/manifold runs committed under
``experiments/snn_reset/results`` and each save as both PDF and PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ROOT = HERE.parent
RESULTS = ROOT / "experiments" / "snn_reset" / "results"

RELEARNING = RESULTS / "snn_reset_relearning_exhaustive_20260606T070543Z"
RELEARNING_ANALYSIS = RELEARNING / "relearning_analysis"
RAW_RELEARNING = RELEARNING_ANALYSIS / "raw_relearning_trials.csv"
TASK_PROTOCOLS = RELEARNING_ANALYSIS / "task_protocol_relearning_summary.csv"
PROTOCOLS = RELEARNING_ANALYSIS / "protocol_relearning_summary.csv"
MANIFOLD = RESULTS / "manifold_analysis_20260613T224052Z"
INTERPOLATION = MANIFOLD / "interpolation.csv"
ABLATIONS = MANIFOLD / "ablations.csv"
BREAKING_POINTS = MANIFOLD / "breaking_points.csv"

TASK_LABELS = {
    "conditioned_electrode_association": "Association",
    "pattern_discrimination": "Pattern",
}
TASK_ORDER = ["Association", "Pattern"]
TASK_COLORS = {"Association": "#0072B2", "Pattern": "#E69F00"}
PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "purple": "#7B3F98",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "black": "#111827",
}
SCHEDULE_LABELS = {
    "static": "static",
    "alternating_blue_red": "blue-red",
    "epoch_pause": "epoch-pause",
    "chirp": "chirp",
    "gated_burst": "gated burst",
}
SCHEDULE_ORDER = ["static", "blue-red", "epoch-pause", "chirp", "gated burst"]
BETA_ORDER = [-2.0, 0.0, 2.0]
BETA_LABELS = {-2.0: "-2", 0.0: "0", 2.0: "2"}
STATE_LABELS = {
    "baseline_score": "baseline",
    "trained_score": "trained",
    "naive_weight_control_score": "naive-weight\ncontrol",
}
DIRECTION_LABELS = {
    "axis": "toward naive\nmemory axis",
    "random": "matched random\ndirection",
}
DIRECTION_COLORS = {
    "toward naive\nmemory axis": PALETTE["vermillion"],
    "matched random\ndirection": PALETTE["gray"],
}
CONDITION_ORDER = [
    "trained",
    "anti_training_delta",
    "random_matched",
    "shuffle_magnitudes",
    "zero_input_outgoing",
    "zero_input_to_target_path",
    "zero_target_incoming",
    "zero_all",
    "naive_replacement",
]
CONDITION_LABELS = {
    "trained": "trained",
    "anti_training_delta": "anti-training delta",
    "random_matched": "random matched",
    "shuffle_magnitudes": "shuffle magnitudes",
    "zero_input_outgoing": "zero input outgoing",
    "zero_input_to_target_path": "zero input-to-target path",
    "zero_target_incoming": "zero target incoming",
    "zero_all": "zero all",
    "naive_replacement": "naive replacement",
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "legend.title_fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFBFC",
            "figure.facecolor": "white",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def label_tasks(df: pd.DataFrame, column: str = "task_name") -> pd.DataFrame:
    out = df[df[column].isin(TASK_LABELS)].copy()
    out["task"] = pd.Categorical(out[column].map(TASK_LABELS), categories=TASK_ORDER, ordered=True)
    return out


def label_protocols(df: pd.DataFrame) -> pd.DataFrame:
    out = label_tasks(df)
    out["schedule_label"] = pd.Categorical(
        out["schedule"].map(SCHEDULE_LABELS),
        categories=SCHEDULE_ORDER,
        ordered=True,
    )
    out["beta_label"] = pd.Categorical(out["beta"].astype(float).map(BETA_LABELS), categories=["-2", "0", "2"])
    displacement = out["reset_minus_no_reset_weight_norm"].astype(float)
    projection = out["erasure_projection_reset_vs_no_reset"].astype(float)
    out["orthogonal_displacement"] = np.sqrt(np.maximum(displacement.to_numpy() ** 2 - projection.to_numpy() ** 2, 0.0))
    out["axis_fraction_pct"] = 100.0 * projection / displacement.replace(0.0, np.nan)
    out["reset_spike_delta_k"] = out["reset_window_neuron_spikes_delta"] / 1_000.0
    out["reset_spike_delta_log10"] = np.log10(out["reset_spike_delta_k"].clip(lower=0.0) + 1.0)
    out["displacement_log10"] = np.log10(displacement.clip(lower=0.0) + 1.0)
    out["orthogonal_log10"] = np.log10(out["orthogonal_displacement"].clip(lower=0.0) + 1.0)
    out["current_label"] = out["current_uA"].map(lambda value: f"{value:g} uA")
    return out


def raw_relearning() -> pd.DataFrame:
    return label_protocols(pd.read_csv(RAW_RELEARNING))


def task_protocols() -> pd.DataFrame:
    return label_protocols(pd.read_csv(TASK_PROTOCOLS))


def protocol_summary() -> pd.DataFrame:
    out = pd.read_csv(PROTOCOLS).copy()
    out["schedule_label"] = pd.Categorical(
        out["schedule"].map(SCHEDULE_LABELS),
        categories=SCHEDULE_ORDER,
        ordered=True,
    )
    out["beta_label"] = pd.Categorical(out["beta"].astype(float).map(BETA_LABELS), categories=["-2", "0", "2"])
    return out


def training_history_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row in raw.drop_duplicates(["task_name", "seed"]).itertuples(index=False):
        scores = [float(value) for value in str(row.training_history).split("|") if value != ""]
        for step, score in enumerate(scores):
            rows.append(
                {
                    "task": row.task,
                    "seed": int(row.seed),
                    "step": step,
                    "score": score,
                    "criterion": float(row.criterion_score),
                }
            )
    return pd.DataFrame(rows)


def state_score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row in raw.drop_duplicates(["task_name", "seed"]).itertuples(index=False):
        values = row._asdict()
        for column, state in STATE_LABELS.items():
            rows.append({"task": row.task, "seed": int(row.seed), "state": state, "score": float(values[column])})
    return pd.DataFrame(rows)


def relearning_trial_frame(raw: pd.DataFrame) -> pd.DataFrame:
    initial = raw.drop_duplicates(["task_name", "seed"])[["task", "seed", "initial_trials_to_criterion"]].copy()
    initial = initial.rename(columns={"initial_trials_to_criterion": "trials"})
    initial["phase"] = "initial learning"

    relearn = (
        raw.groupby(["task", "protocol_id"], observed=True, as_index=False)["relearn_trials"]
        .mean()
        .rename(columns={"relearn_trials": "trials"})
    )
    relearn["phase"] = "after reset"
    return pd.concat([initial[["task", "phase", "trials"]], relearn[["task", "phase", "trials"]]], ignore_index=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        weight="bold",
        color=PALETTE["black"],
    )


def task_current_handles() -> list[Line2D]:
    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TASK_COLORS[task],
            markeredgecolor="white",
            label=task,
            markersize=5.8,
        )
        for task in TASK_ORDER
    ]
    current_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=PALETTE["gray"], label="50 uA", markersize=5.2),
        Line2D([0], [0], marker="X", linestyle="", color=PALETTE["gray"], label="10 uA", markersize=5.2),
    ]
    return task_handles + current_handles


def strip_and_bar(ax: plt.Axes, data: pd.DataFrame, *, x: str, y: str, order: list[str], ylabel: str) -> None:
    sns.barplot(
        data,
        x=x,
        y=y,
        hue="task",
        order=order,
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        errorbar="sd",
        capsize=0.12,
        err_kws={"linewidth": 0.8},
        alpha=0.72,
        ax=ax,
    )
    sns.stripplot(
        data,
        x=x,
        y=y,
        hue="task",
        order=order,
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        dodge=True,
        jitter=0.16,
        s=3.8,
        linewidth=0.35,
        edgecolor="white",
        alpha=0.9,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, title="")


def fig1_design() -> None:
    raw = raw_relearning()
    history = training_history_frame(raw)
    states = state_score_frame(raw)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.3, 3.05),
        gridspec_kw={"width_ratios": [1.28, 1.0]},
        constrained_layout=True,
    )
    sns.lineplot(
        history,
        x="step",
        y="score",
        hue="task",
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        estimator="mean",
        errorbar="sd",
        linewidth=2.0,
        ax=axes[0],
    )
    for task, criterion in history.groupby("task", observed=True)["criterion"].first().items():
        axes[0].axhline(criterion, color=TASK_COLORS[str(task)], linestyle="--", linewidth=0.85, alpha=0.62)
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].set_xlabel("training repetition")
    axes[0].set_ylabel("task score")
    axes[0].set_title("Training reaches criterion")
    axes[0].legend(frameon=False, title="")
    panel_label(axes[0], "A")

    strip_and_bar(axes[1], states, x="state", y="score", order=list(STATE_LABELS.values()), ylabel="task score")
    axes[1].set_ylim(-0.08, 1.08)
    axes[1].set_title("Readout depends on learned weights")
    axes[1].tick_params(axis="x", rotation=10)
    panel_label(axes[1], "B")
    fig.suptitle("Validated task assay before reset", fontsize=11, weight="bold")
    save(fig, "fig1_design")


def fig2_readout() -> None:
    summary = task_protocols()
    heat = (
        summary.groupby(["task", "schedule_label", "beta_label"], observed=True, as_index=False)["forgetting_score"]
        .mean()
        .copy()
    )
    vmax = max(0.35, float(heat["forgetting_score"].max()))
    cmap = sns.light_palette(PALETTE["vermillion"], as_cmap=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.55, 3.2),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]},
        constrained_layout=True,
    )
    for ax, task in zip(axes[:2], TASK_ORDER, strict=True):
        mat = (
            heat[heat["task"] == task]
            .pivot(index="schedule_label", columns="beta_label", values="forgetting_score")
            .reindex(index=SCHEDULE_ORDER, columns=["-2", "0", "2"])
        )
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            linewidths=0.7,
            linecolor="white",
            cbar=ax is axes[1],
            cbar_kws={"label": "mean immediate forgetting"},
            ax=ax,
        )
        ax.set_title(task)
        ax.set_xlabel("temporal color beta")
        ax.set_ylabel("schedule" if ax is axes[0] else "")
        if ax is axes[1]:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        panel_label(ax, "A" if ax is axes[0] else "B")

    scatter = summary.copy()
    sns.scatterplot(
        scatter,
        x="reset_spike_delta_log10",
        y="forgetting_score",
        hue="task",
        hue_order=TASK_ORDER,
        style="current_label",
        markers={"50 uA": "o", "10 uA": "X"},
        size="energy_cost_uC",
        sizes=(22, 145),
        palette=TASK_COLORS,
        alpha=0.76,
        linewidth=0.35,
        edgecolor="white",
        legend=False,
        ax=axes[2],
    )
    axes[2].axhline(0.0, color=PALETTE["black"], linestyle=":", linewidth=0.85, alpha=0.62)
    axes[2].set_xlabel("log10(extra reset-window spikes / 1k + 1)")
    axes[2].set_ylabel("immediate forgetting")
    axes[2].set_title("Burden is not forgetting")
    axes[2].legend(handles=task_current_handles(), frameon=False, loc="upper left")
    axes[2].text(
        0.98,
        0.04,
        "point size = charge",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["gray"],
    )
    panel_label(axes[2], "C")
    fig.suptitle("Protocol screen: stronger stimulation dents score but never crosses criterion", fontsize=11, weight="bold")
    save(fig, "fig2_readout")


def fig3_dissociation() -> None:
    summary = task_protocols()
    raw = raw_relearning()
    relearn = relearning_trial_frame(raw)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.15), constrained_layout=True)
    sns.scatterplot(
        summary,
        x="erasure_projection_reset_vs_no_reset",
        y="orthogonal_log10",
        hue="task",
        hue_order=TASK_ORDER,
        style="schedule_label",
        size="forgetting_score",
        sizes=(22, 140),
        palette=TASK_COLORS,
        alpha=0.76,
        linewidth=0.35,
        edgecolor="white",
        ax=axes[0],
    )
    axes[0].axvline(0.0, color=PALETTE["black"], linestyle=":", linewidth=0.9, alpha=0.65)
    axes[0].set_xlabel("memory-axis movement")
    axes[0].set_ylabel("log10(orthogonal displacement + 1)")
    axes[0].set_title("Reset motion is off-axis")
    axes[0].legend([], [], frameon=False)
    panel_label(axes[0], "A")

    sns.scatterplot(
        summary,
        x="displacement_log10",
        y="reset_score",
        hue="task",
        hue_order=TASK_ORDER,
        style="current_label",
        size="forgetting_score",
        sizes=(20, 135),
        palette=TASK_COLORS,
        alpha=0.78,
        linewidth=0.35,
        edgecolor="white",
        ax=axes[1],
    )
    for task, criterion in summary.groupby("task", observed=True)["criterion_score"].first().items():
        axes[1].axhline(criterion, color=TASK_COLORS[str(task)], linestyle="--", linewidth=0.82, alpha=0.62)
    axes[1].set_xlabel("log10(reset-induced displacement + 1)")
    axes[1].set_ylabel("post-reset task score")
    axes[1].set_ylim(0.25, 1.05)
    axes[1].set_title("Large shifts preserve function")
    axes[1].legend([], [], frameon=False)
    panel_label(axes[1], "B")

    phase_order = ["initial learning", "after reset"]
    strip_and_bar(axes[2], relearn, x="phase", y="trials", order=phase_order, ylabel="trials to criterion")
    axes[2].set_title("Reset does not reopen learning")
    axes[2].tick_params(axis="x", rotation=8)
    panel_label(axes[2], "C")
    fig.suptitle("Functional memory survives the reset sweep", fontsize=11, weight="bold")
    save(fig, "fig3_dissociation")


def fig4_breaking_point() -> None:
    interp = pd.read_csv(INTERPOLATION).copy()
    bp = label_tasks(pd.read_csv(BREAKING_POINTS), column="task")
    interp = label_tasks(interp, column="task")
    interp["direction_label"] = interp["direction"].map(DIRECTION_LABELS)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9.45, 3.25),
        gridspec_kw={"width_ratios": [1.1, 1.1, 0.78]},
        constrained_layout=True,
    )
    for ax, task, label in zip(axes[:2], TASK_ORDER, ["A", "B"], strict=True):
        sub = interp[interp["task"] == task]
        sns.lineplot(
            sub,
            x="alpha",
            y="score",
            hue="direction_label",
            hue_order=list(DIRECTION_LABELS.values()),
            palette=DIRECTION_COLORS,
            estimator="mean",
            errorbar="sd",
            marker="o",
            markersize=4,
            linewidth=1.9,
            ax=ax,
        )
        criterion = float(sub["criterion"].iloc[0])
        astar = float(bp.loc[bp["task"] == task, "alpha_star_axis"].mean())
        ax.axhline(criterion, color=PALETTE["black"], linestyle="--", linewidth=0.9, alpha=0.72)
        ax.axvline(astar, color=PALETTE["vermillion"], linestyle=":", linewidth=1.0, alpha=0.82)
        ax.text(astar + 0.025, 0.06, f"alpha*={astar:.2f}", color=PALETTE["vermillion"], fontsize=7.4)
        ax.set_xlabel("fraction moved from trained toward endpoint")
        ax.set_ylabel("task score")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(task)
        ax.legend(frameon=False, title="")
        panel_label(ax, label)

    sns.stripplot(
        bp,
        y="task",
        x="alpha_star_axis",
        order=TASK_ORDER,
        hue="task",
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        s=5.0,
        jitter=0.13,
        alpha=0.85,
        legend=False,
        ax=axes[2],
    )
    sns.pointplot(
        bp,
        y="task",
        x="alpha_star_axis",
        order=TASK_ORDER,
        hue="task",
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        errorbar="sd",
        markers="D",
        linestyles="",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_xlim(0.0, 0.18)
    axes[2].set_xlabel("axis breaking point")
    axes[2].set_ylabel("")
    axes[2].set_title("Tiny axis move erases")
    axes[2].grid(axis="y", visible=False)
    panel_label(axes[2], "C")
    fig.suptitle("Memory is direction-specific, not displacement-magnitude specific", fontsize=11, weight="bold")
    save(fig, "fig4_breaking_point")


def fig5_ablations() -> None:
    abl = pd.read_csv(ABLATIONS).copy()
    abl = label_tasks(abl, column="task")
    abl["condition_label"] = pd.Categorical(
        abl["condition"].map(CONDITION_LABELS),
        categories=[CONDITION_LABELS[name] for name in CONDITION_ORDER],
        ordered=True,
    )
    targeted = abl[abl["edges_changed_fraction"].notna()].copy()
    targeted["edges_changed_pct"] = 100.0 * targeted["edges_changed_fraction"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.45, 4.2),
        gridspec_kw={"width_ratios": [1.65, 0.82]},
        constrained_layout=True,
    )
    sns.barplot(
        abl,
        y="condition_label",
        x="score",
        hue="task",
        order=[CONDITION_LABELS[name] for name in CONDITION_ORDER],
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        errorbar="sd",
        capsize=0.12,
        err_kws={"linewidth": 0.8},
        alpha=0.72,
        ax=axes[0],
    )
    sns.stripplot(
        abl,
        y="condition_label",
        x="score",
        hue="task",
        order=[CONDITION_LABELS[name] for name in CONDITION_ORDER],
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        dodge=True,
        jitter=0.18,
        s=3.9,
        linewidth=0.35,
        edgecolor="white",
        alpha=0.92,
        legend=False,
        ax=axes[0],
    )
    for task, criterion in abl.groupby("task", observed=True)["criterion"].first().items():
        axes[0].axvline(criterion, color=TASK_COLORS[str(task)], linestyle="--", linewidth=0.85, alpha=0.62)
    axes[0].set_xlabel("task score")
    axes[0].set_ylabel("")
    axes[0].set_title("Surgical perturbations localize the memory")
    axes[0].legend(frameon=False, title="")
    panel_label(axes[0], "A")

    targeted_order = [
        CONDITION_LABELS["zero_input_to_target_path"],
        CONDITION_LABELS["zero_input_outgoing"],
        CONDITION_LABELS["zero_target_incoming"],
    ]
    sns.barplot(
        targeted,
        y="condition_label",
        x="edges_changed_pct",
        hue="task",
        order=targeted_order,
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        errorbar="sd",
        capsize=0.12,
        err_kws={"linewidth": 0.8},
        alpha=0.72,
        ax=axes[1],
    )
    sns.stripplot(
        targeted,
        y="condition_label",
        x="edges_changed_pct",
        hue="task",
        order=targeted_order,
        hue_order=TASK_ORDER,
        palette=TASK_COLORS,
        dodge=True,
        jitter=0.14,
        s=3.8,
        linewidth=0.35,
        edgecolor="white",
        alpha=0.92,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("edges changed (%)")
    axes[1].set_ylabel("")
    axes[1].set_title("Small targeted cuts are enough")
    axes[1].legend([], [], frameon=False)
    panel_label(axes[1], "B")
    fig.suptitle("The learned association is pathway-specific", fontsize=11, weight="bold")
    save(fig, "fig5_ablations")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    fig1_design()
    fig2_readout()
    fig3_dissociation()
    fig4_breaking_point()
    fig5_ablations()
    outputs = sorted(path.name for path in FIG.glob("fig*.*"))
    print("wrote:", *outputs)


if __name__ == "__main__":
    main()
