from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cl1_snn_reset import ResetProtocol, coarse_protocol_grid, protocol_events

from figlib import (
    BRANCH_COLORS,
    PALETTE,
    SCHEDULE_COLORS,
    SCHEDULE_LABELS,
    SCHEDULE_ORDER,
    TASK_COLORS,
    TASK_ORDER,
    apply_style,
    default_output_dir,
    facet_stripplot,
    latest_suite_dir,
    load_raw,
    save,
    schedule_stripplot,
    task_protocol_summary,
    task_schedule_summary,
)
from figures_schematics import (
    figure_experiment_timeline,
    figure_metric_definitions,
    figure_network_architecture,
    figure_paired_clone_assay,
    figure_sample_structure,
    figure_task_schematics,
)


def parse_training_histories(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = raw.drop_duplicates(["task_label", "seed"])[["task_label", "seed", "criterion_score", "training_history"]]
    for record in seeds.to_dict(orient="records"):
        history = [
            float(value)
            for value in str(record["training_history"]).split("|")
            if value != "" and value.lower() != "nan"
        ]
        for trial_index, score in enumerate(history, start=1):
            rows.append(
                {
                    "task_label": record["task_label"],
                    "seed": int(record["seed"]),
                    "trial": trial_index,
                    "score": score,
                    "criterion_score": float(record["criterion_score"]),
                }
            )
    return pd.DataFrame(rows)


def figure_training_curves(raw: pd.DataFrame, out_dir: Path) -> None:
    histories = parse_training_histories(raw)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        task_hist = histories[histories["task_label"] == task_label]
        if task_hist.empty:
            ax.set_visible(False)
            continue
        sns.lineplot(task_hist, x="trial", y="score", hue="seed", palette="viridis", marker="o", ms=2.5, lw=1.2, ax=ax)
        criterion = float(task_hist["criterion_score"].iloc[0])
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("training repetition")
        ax.set_ylabel("task score")
        ax.legend(frameon=False, title="seed")
    save(fig, out_dir, "M7_R1_training_curves_to_criterion")


def figure_trained_margin_distribution(raw: pd.DataFrame, out_dir: Path) -> None:
    seeds = raw.drop_duplicates(["task_label", "seed"]).copy()
    seeds["trained_margin"] = seeds["trained_score"] - seeds["criterion_score"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sns.stripplot(
        seeds,
        x="task_label",
        y="trained_margin",
        hue="task_label",
        palette=TASK_COLORS,
        order=TASK_ORDER,
        jitter=0.12,
        s=7,
        ax=ax,
        legend=False,
    )
    for index, task_label in enumerate(TASK_ORDER):
        values = seeds.loc[seeds["task_label"] == task_label, "trained_margin"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mean = float(values.mean())
        half_width = 0.12
        ax.hlines(mean, index - half_width, index + half_width, color=PALETTE["black"], lw=1.4)
        if len(values) > 1:
            ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
            ax.vlines(index, mean - ci, mean + ci, color=PALETTE["black"], lw=1.0)
    ax.axhline(0.0, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.7)
    ax.set_xlabel("task")
    ax.set_ylabel("trained score - criterion")
    ax.set_title("Trained-state margin before reset")
    save(fig, out_dir, "M8_trained_state_margins")


def representative_protocols() -> list[ResetProtocol]:
    protocols = coarse_protocol_grid()
    wanted = []
    for schedule in ["static", "alternating_blue_red", "epoch_pause", "chirp", "gated_burst"]:
        for protocol in protocols:
            if (
                protocol.schedule == schedule
                and protocol.current_uA == 50.0
                and protocol.spatial_mode == "shared"
                and protocol.beta == 0
            ):
                wanted.append(protocol)
                break
    return wanted


def schedule_events(protocol: ResetProtocol, seed: int = 13):
    rng = np.random.default_rng(seed)
    return protocol_events(protocol, n_channels=64, rng=rng)


def figure_protocol_schedule_raster(out_dir: Path) -> None:
    protocols = representative_protocols()
    fig, axes = plt.subplots(len(protocols), 1, figsize=(8.4, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, protocol in zip(axes, protocols, strict=True):
        events = schedule_events(protocol)
        times = []
        channels = []
        for event in events:
            for channel in event.channels:
                times.append(event.time_us / 1_000_000.0)
                channels.append(channel)
        label = SCHEDULE_LABELS.get(protocol.schedule, protocol.schedule)
        ax.scatter(times, channels, s=3, color=SCHEDULE_COLORS.get(label, PALETTE["blue"]), alpha=0.75, linewidths=0)
        ax.set_ylabel(label)
        ax.set_ylim(-1, 64)
        ax.grid(axis="x", alpha=0.18)
    axes[-1].set_xlabel("protocol time (s)")
    fig.suptitle("Representative stimulation schedule rasters", y=1.01, fontsize=12, weight="bold")
    save(fig, out_dir, "M9_protocol_schedule_rasters")


def figure_protocol_dose_summary(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        raw.groupby(["schedule_label", "current_uA"], as_index=False)
        .agg(
            total_pulses=("total_pulses", "mean"),
            energy_cost_uC=("energy_cost_uC", "mean"),
            extra_spikes_per_neuron_s=("extra_spikes_per_neuron_s", "mean"),
            weight_norm=("reset_minus_no_reset_weight_norm", "mean"),
        )
        .sort_values("current_uA")
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), constrained_layout=True)
    specs = [
        ("total_pulses", "channel-pulses"),
        ("energy_cost_uC", "nominal charge (uC)"),
        ("extra_spikes_per_neuron_s", "extra spikes / neuron / s"),
        ("weight_norm", "||Wreset - Wno-reset||"),
    ]
    for ax, (metric, ylabel) in zip(axes.ravel(), specs, strict=True):
        sns.barplot(
            summary,
            x="schedule_label",
            y=metric,
            hue="current_uA",
            order=SCHEDULE_ORDER,
            palette=[PALETTE["sky"], PALETTE["vermillion"]],
            ax=ax,
        )
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, title="uA")
    save(fig, out_dir, "M10_protocol_dose_summary")


def figure_protocol_temporal_signatures(out_dir: Path) -> None:
    protocols = representative_protocols()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    bin_s = 0.05
    for protocol in protocols:
        label = SCHEDULE_LABELS.get(protocol.schedule, protocol.schedule)
        events = schedule_events(protocol)
        bins = np.arange(0, protocol.duration_s + bin_s, bin_s)
        event_times = np.array([event.time_us / 1_000_000.0 for event in events], dtype=float)
        counts, edges = np.histogram(event_times, bins=bins)
        rate = counts / bin_s
        centers = edges[:-1] + bin_s / 2.0
        color = SCHEDULE_COLORS.get(label, PALETTE["blue"])
        axes[0].plot(centers, rate, lw=1.3, label=label, color=color)
        spectrum = np.abs(np.fft.rfft(rate - rate.mean())) ** 2
        freqs = np.fft.rfftfreq(len(rate), d=bin_s)
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()
        axes[1].plot(freqs[1:], spectrum[1:], lw=1.3, label=label, color=color)
    axes[0].set_xlabel("protocol time (s)")
    axes[0].set_ylabel("events / s")
    axes[0].set_title("Temporal event burden")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("normalized power")
    axes[1].set_title("Schedule spectra")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, ncol=2)
    save(fig, out_dir, "M11_protocol_temporal_signatures")


def figure_learning_phase_summary(raw: pd.DataFrame, out_dir: Path) -> None:
    phases = (
        raw.drop_duplicates(["task_label", "seed"])
        .melt(
            id_vars=["task_label", "seed", "criterion_score"],
            value_vars=["baseline_score", "trained_score", "naive_weight_control_score"],
            var_name="phase",
            value_name="score",
        )
        .replace(
            {
                "baseline_score": "baseline",
                "trained_score": "trained",
                "naive_weight_control_score": "naive-weight control",
            }
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = phases[phases["task_label"] == task_label]
        sns.stripplot(
            frame,
            x="phase",
            y="score",
            order=["baseline", "trained", "naive-weight control"],
            color=TASK_COLORS.get(task_label, PALETTE["blue"]),
            jitter=0.12,
            s=6,
            ax=ax,
        )
        criterion = float(frame["criterion_score"].iloc[0])
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("")
        ax.set_ylabel("task score")
        ax.tick_params(axis="x", rotation=15)
    save(fig, out_dir, "R1_baseline_learning_success")


def figure_no_reset_stability(raw: pd.DataFrame, out_dir: Path) -> None:
    paired = raw.melt(
        id_vars=["task_label", "seed", "protocol_id", "schedule_label", "criterion_score"],
        value_vars=["trained_score", "no_reset_score"],
        var_name="branch",
        value_name="score",
    ).replace({"trained_score": "trained clone", "no_reset_score": "no-reset post"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        task = paired[paired["task_label"] == task_label]
        sns.stripplot(
            task,
            x="branch",
            y="score",
            hue="branch",
            palette={"trained clone": PALETTE["green"], "no-reset post": PALETTE["gray"]},
            jitter=0.18,
            alpha=0.45,
            s=3.2,
            ax=ax,
            legend=False,
        )
        criterion = float(task["criterion_score"].iloc[0])
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("")
        ax.set_ylabel("task score")
    save(fig, out_dir, "R2_no_reset_stability")


def figure_activity_perturbation(raw: pd.DataFrame, out_dir: Path) -> None:
    schedule_stripplot(
        raw,
        out_dir,
        y="extra_spikes_per_neuron_s",
        ylabel="extra reset-window spikes / neuron / s",
        title="Reset protocols evoke activity perturbations",
        name="R3_reset_evoked_activity_perturbation",
        zero_line=True,
    )


def figure_synaptic_displacement(raw: pd.DataFrame, out_dir: Path) -> None:
    schedule_stripplot(
        raw,
        out_dir,
        y="reset_minus_no_reset_weight_norm",
        ylabel="||Wreset - Wno-reset||",
        title="Stimulation-induced synaptic displacement",
        name="R6_synaptic_displacement_by_schedule",
    )


def figure_training_axis_projection(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    sns.stripplot(
        raw,
        x="schedule_label",
        y="erasure_projection_reset_vs_no_reset",
        hue="task_label",
        order=SCHEDULE_ORDER,
        palette=TASK_COLORS,
        dodge=True,
        jitter=0.2,
        alpha=0.7,
        s=3.8,
        ax=axes[0],
    )
    axes[0].axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.7)
    axes[0].set_xlabel("stimulation schedule")
    axes[0].set_ylabel("training-axis erasure fraction")
    axes[0].set_title("Projection onto training vector")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, title="")
    components = (
        raw.groupby(["task_label", "schedule_label"], as_index=False)
        .agg(parallel=("erasure_parallel_norm", "mean"), orthogonal=("erasure_orthogonal_norm", "mean"))
        .melt(
            id_vars=["task_label", "schedule_label"],
            value_vars=["parallel", "orthogonal"],
            var_name="component",
            value_name="norm",
        )
    )
    sns.barplot(
        components,
        x="schedule_label",
        y="norm",
        hue="component",
        order=SCHEDULE_ORDER,
        palette={"parallel": PALETTE["vermillion"], "orthogonal": PALETTE["gray"]},
        ax=axes[1],
    )
    axes[1].set_xlabel("stimulation schedule")
    axes[1].set_ylabel("component norm")
    axes[1].set_title("Parallel vs orthogonal displacement")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, title="")
    save(fig, out_dir, "R10_R11_training_axis_projection")


def figure_absolute_scores(raw: pd.DataFrame, out_dir: Path) -> None:
    scores = raw.melt(
        id_vars=["task_label", "schedule_label", "seed", "protocol_id", "criterion_score"],
        value_vars=["no_reset_score", "reset_score"],
        var_name="branch",
        value_name="score",
    ).replace({"no_reset_score": "no reset", "reset_score": "reset"})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = scores[scores["task_label"] == task_label]
        sns.stripplot(
            frame,
            x="schedule_label",
            y="score",
            hue="branch",
            order=SCHEDULE_ORDER,
            palette=BRANCH_COLORS,
            dodge=True,
            jitter=0.16,
            alpha=0.65,
            s=3.6,
            ax=ax,
        )
        criterion = float(frame["criterion_score"].iloc[0])
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("stimulation schedule")
        ax.set_ylabel("task score")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, title="")
    save(fig, out_dir, "R12_absolute_scores_vs_criterion")


def figure_score_dents(raw: pd.DataFrame, out_dir: Path) -> None:
    facet_stripplot(
        raw,
        out_dir,
        y="score_dent",
        ylabel="score dent: no-reset - reset",
        name="R13_paired_behavioral_score_dents",
    )


def figure_criterion_margins(raw: pd.DataFrame, out_dir: Path) -> None:
    facet_stripplot(
        raw,
        out_dir,
        y="criterion_margin",
        ylabel="reset score - criterion",
        name="R14_post_reset_criterion_margins",
    )


def figure_margin_consumed(raw: pd.DataFrame, out_dir: Path) -> None:
    facet_stripplot(
        raw,
        out_dir,
        y="margin_consumed",
        ylabel="fraction of no-reset margin consumed",
        name="R15_margin_consumed_by_stimulation",
        hlines=((1.0, "--", "black"), (0.0, ":", "gray")),
    )


def binomial_zero_upper(n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return np.nan
    return 1.0 - alpha ** (1.0 / float(n))


def figure_forgetting_incidence(raw: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for label, frame in list(raw.groupby("task_label")) + [("all tasks", raw)]:
        n = len(frame)
        observed = int(frame["criterion_forget"].sum())
        rows.append(
            {
                "group": label,
                "n": n,
                "observed": observed,
                "incidence": observed / n if n else np.nan,
                "upper95": binomial_zero_upper(n) if observed == 0 else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    sns.barplot(summary, x="group", y="incidence", color=PALETTE["vermillion"], ax=ax)
    ax.errorbar(
        np.arange(len(summary)),
        summary["incidence"],
        yerr=summary["upper95"].fillna(0.0),
        fmt="none",
        ecolor=PALETTE["black"],
        capsize=4,
        lw=1.0,
    )
    for index, row in summary.iterrows():
        ax.text(index, 0.002, f"{int(row['observed'])}/{int(row['n'])}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, max(0.035, float(summary["upper95"].max()) * 1.4))
    ax.set_xlabel("")
    ax.set_ylabel("criterion-level forgetting incidence")
    ax.set_title("No reset branch crossed below criterion")
    save(fig, out_dir, "R16_criterion_forgetting_incidence")


def figure_relearning_burden(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    sns.stripplot(
        raw,
        x="task_label",
        y="relearn_trials",
        hue="task_label",
        order=TASK_ORDER,
        palette=TASK_COLORS,
        jitter=0.20,
        alpha=0.7,
        s=4.0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel("task")
    axes[0].set_ylabel("relearning trials to criterion")
    axes[0].set_title("Relearning burden")
    axes[0].set_ylim(-0.08, max(1.0, raw["relearn_trials"].max() + 0.4))
    counts = (
        raw.groupby("task_label", as_index=False)
        .agg(n=("relearn_trials", "size"), zero_trials=("relearn_trials", lambda values: int((values == 0).sum())))
        .sort_values("task_label")
    )
    sns.barplot(counts, x="task_label", y="zero_trials", hue="task_label", palette=TASK_COLORS, order=TASK_ORDER, ax=axes[1], legend=False)
    for index, task_label in enumerate(TASK_ORDER):
        row = counts[counts["task_label"] == task_label]
        if row.empty:
            continue
        n = int(row["n"].iloc[0])
        zero_trials = int(row["zero_trials"].iloc[0])
        axes[1].text(index, zero_trials + max(n * 0.02, 2.0), f"{zero_trials}/{n}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xlabel("task")
    axes[1].set_ylabel("evaluations with zero relearning trials")
    axes[1].set_title("All reset branches start above criterion")
    axes[1].set_ylim(0, max(float(counts["n"].max()) * 1.15, 1.0))
    save(fig, out_dir, "R17_relearning_burden_zero")


def figure_positive_control_naive_weight(raw: pd.DataFrame, out_dir: Path) -> None:
    controls = (
        raw.drop_duplicates(["task_label", "seed"])
        .melt(
            id_vars=["task_label", "seed", "criterion_score"],
            value_vars=["baseline_score", "trained_score", "reset_score", "naive_weight_control_score"],
            var_name="condition",
            value_name="score",
        )
        .replace(
            {
                "baseline_score": "baseline",
                "trained_score": "trained",
                "reset_score": "stimulation reset",
                "naive_weight_control_score": "naive-weight control",
            }
        )
    )
    order = ["baseline", "trained", "stimulation reset", "naive-weight control"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = controls[controls["task_label"] == task_label]
        sns.stripplot(frame, x="condition", y="score", order=order, color=TASK_COLORS[task_label], jitter=0.12, s=6, ax=ax)
        criterion = float(frame["criterion_score"].iloc[0])
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("")
        ax.set_ylabel("task score")
        ax.tick_params(axis="x", rotation=25)
    save(fig, out_dir, "R37_naive_weight_positive_control")


def figure_outcome_summary_matrix(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = task_schedule_summary(raw)
    summary["row"] = summary["task_label"] + " | " + summary["schedule_label"]
    metrics = {
        "score_dent": "mean score dent",
        "min_criterion_margin": "min margin",
        "criterion_forget": "criterion failures",
        "relearn_trials": "relearn trials",
        "reset_minus_no_reset_weight_norm": "weight displacement",
        "extra_spikes_per_neuron_s": "extra spikes/neuron/s",
    }
    table = summary.set_index("row")[list(metrics)].rename(columns=metrics)
    scaled = table.copy()
    for column in scaled.columns:
        values = scaled[column].astype(float)
        span = values.max() - values.min()
        scaled[column] = 0.0 if span <= 1e-12 else (values - values.min()) / span
    annotations = table.copy()
    for column in annotations.columns:
        annotations[column] = annotations[column].map(lambda value: f"{value:.3g}")
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    sns.heatmap(
        scaled,
        cmap="viridis",
        annot=annotations,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "column-scaled value"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Behavioral and perturbation outcome summary")
    save(fig, out_dir, "R19_behavioral_outcome_summary_matrix")


def figure_weight_behavior_dissociation(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    sns.scatterplot(
        raw,
        x="log_weight_effect",
        y="score_dent",
        hue="task_label",
        style="schedule_label",
        palette=TASK_COLORS,
        s=34,
        alpha=0.78,
        ax=axes[0],
    )
    axes[0].axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.65)
    axes[0].set_xlabel("log10(1 + ||Wreset - Wno-reset||)")
    axes[0].set_ylabel("score dent")
    axes[0].set_title("Displacement vs score dent")
    sns.scatterplot(
        raw,
        x="log_weight_effect",
        y="criterion_margin",
        hue="task_label",
        style="schedule_label",
        palette=TASK_COLORS,
        s=34,
        alpha=0.78,
        ax=axes[1],
        legend=False,
    )
    axes[1].axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.65)
    axes[1].set_xlabel("log10(1 + ||Wreset - Wno-reset||)")
    axes[1].set_ylabel("reset score - criterion")
    axes[1].set_title("Displacement vs erasure boundary")
    sns.scatterplot(
        raw,
        x="log_weight_effect",
        y="relearn_trials",
        hue="task_label",
        style="schedule_label",
        palette=TASK_COLORS,
        s=34,
        alpha=0.78,
        ax=axes[2],
        legend=False,
    )
    axes[2].set_xlabel("log10(1 + ||Wreset - Wno-reset||)")
    axes[2].set_ylabel("relearning trials")
    axes[2].set_title("Displacement vs relearning burden")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=3, title="")
    save(fig, out_dir, "R20_R21_R22_weight_behavior_dissociation")


def figure_activity_dissociation(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    sns.scatterplot(
        raw,
        x="extra_spikes_per_neuron_s",
        y="score_dent",
        hue="task_label",
        style="schedule_label",
        palette=TASK_COLORS,
        s=36,
        alpha=0.78,
        ax=axes[0],
    )
    axes[0].axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.65)
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xlabel("extra spikes / neuron / s")
    axes[0].set_ylabel("score dent")
    axes[0].set_title("Activity burden vs score dent")
    sns.scatterplot(
        raw,
        x="extra_spikes_per_neuron_s",
        y="reset_minus_no_reset_weight_norm",
        hue="task_label",
        style="schedule_label",
        palette=TASK_COLORS,
        s=36,
        alpha=0.78,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xlabel("extra spikes / neuron / s")
    axes[1].set_ylabel("||Wreset - Wno-reset||")
    axes[1].set_title("Activity burden vs weight movement")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=3, title="")
    save(fig, out_dir, "R23_R24_activity_weight_behavior_dissociation")


def figure_seed_effects(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = raw[raw["task_label"] == task_label]
        sns.stripplot(
            frame,
            x="seed",
            y="score_dent",
            hue="schedule_label",
            palette=SCHEDULE_COLORS,
            dodge=True,
            jitter=0.15,
            alpha=0.7,
            s=3.6,
            ax=ax,
        )
        ax.axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.65)
        ax.set_title(task_label)
        ax.set_xlabel("trained network seed")
        ax.set_ylabel("score dent")
        ax.legend(frameon=False, title="", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "R42_per_network_paired_effects")


def figure_criterion_sensitivity(raw: pd.DataFrame, out_dir: Path) -> None:
    thresholds = np.linspace(0.0, 1.0, 81)
    rows = []
    for (task_label, schedule_label), frame in raw.groupby(["task_label", "schedule_label"]):
        for threshold in thresholds:
            rows.append(
                {
                    "task_label": task_label,
                    "schedule_label": schedule_label,
                    "threshold": threshold,
                    "reset_below_threshold": float((frame["reset_score"] < threshold).mean()),
                    "no_reset_below_threshold": float((frame["no_reset_score"] < threshold).mean()),
                }
            )
    sensitivity = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = sensitivity[sensitivity["task_label"] == task_label]
        sns.lineplot(
            frame,
            x="threshold",
            y="reset_below_threshold",
            hue="schedule_label",
            hue_order=SCHEDULE_ORDER,
            palette=SCHEDULE_COLORS,
            ax=ax,
        )
        chosen = float(raw.loc[raw["task_label"] == task_label, "criterion_score"].iloc[0])
        ax.axvline(chosen, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label)
        ax.set_xlabel("hypothetical criterion threshold")
        ax.set_ylabel("fraction reset branches below threshold")
        ax.legend(frameon=False, title="")
    save(fig, out_dir, "R44_criterion_sensitivity_analysis")


def figure_dose_response(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        raw.groupby(["task_label", "schedule_label", "current_uA"], as_index=False)
        .agg(
            score_dent=("score_dent", "mean"),
            criterion_margin=("criterion_margin", "mean"),
            weight_norm=("reset_minus_no_reset_weight_norm", "mean"),
            extra_spikes_per_neuron_s=("extra_spikes_per_neuron_s", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    specs = [
        ("score_dent", "mean score dent"),
        ("criterion_margin", "mean reset margin"),
        ("weight_norm", "mean weight displacement"),
    ]
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        sns.lineplot(
            summary,
            x="current_uA",
            y=metric,
            hue="schedule_label",
            style="task_label",
            hue_order=SCHEDULE_ORDER,
            palette=SCHEDULE_COLORS,
            markers=True,
            dashes=False,
            ax=ax,
        )
        if metric == "criterion_margin":
            ax.axhline(0.0, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_xlabel("stimulation current (uA)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3, title="")
    axes[1].legend_.remove()
    axes[2].legend_.remove()
    save(fig, out_dir, "R48_protocol_dose_response")


def figure_top_protocols(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = task_protocol_summary(raw)
    top = (
        summary.sort_values("score_dent", ascending=False)
        .groupby("task_label", group_keys=False)
        .head(6)
        .copy()
    )
    top["protocol_short"] = (
        top["beta_label"]
        + " | "
        + top["schedule_label"]
        + " | "
        + top["spatial_mode"]
        + " | "
        + top["current_uA"].map(lambda value: f"{value:g} uA")
    )
    top["row_label"] = top["task_label"] + " | " + top["protocol_short"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.0), constrained_layout=True)
    sns.barplot(top, y="row_label", x="score_dent", hue="task_label", dodge=False, palette=TASK_COLORS, ax=axes[0])
    axes[0].set_xlabel("score dent")
    axes[0].set_ylabel("")
    axes[0].set_title("Largest score dents")
    axes[0].legend(frameon=False, title="")
    sns.barplot(top, y="row_label", x="criterion_margin", hue="task_label", dodge=False, palette=TASK_COLORS, ax=axes[1])
    axes[1].axvline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.7)
    axes[1].set_xlabel("reset score - criterion")
    axes[1].set_ylabel("")
    axes[1].set_title("Margins remain positive")
    axes[1].legend_.remove()
    sns.barplot(
        top,
        y="row_label",
        x="reset_minus_no_reset_weight_norm",
        hue="task_label",
        dodge=False,
        palette=TASK_COLORS,
        ax=axes[2],
    )
    axes[2].set_xlabel("||Wreset - Wno-reset||")
    axes[2].set_ylabel("")
    axes[2].set_title("Weight movement in same protocols")
    axes[2].legend_.remove()
    save(fig, out_dir, "top_score_dent_protocols")


def write_index(out_dir: Path, suite_dir: Path, raw: pd.DataFrame) -> None:
    metadata = json.loads((suite_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = []
    for pdf in sorted(out_dir.glob("*.pdf")):
        rows.append(
            {
                "figure": pdf.stem,
                "pdf": str(pdf),
                "png": str(pdf.with_suffix(".png")),
                "source_suite": str(suite_dir),
                "source_git_commit": metadata.get("git_commit", "unknown"),
                "n_rows": len(raw),
                "n_tasks": raw["task_label"].nunique(),
                "n_protocols": raw["protocol_id"].nunique(),
                "n_seeds": raw["seed"].nunique(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "figure_index.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-facing figures for a modular SNN reset grid result.")
    parser.add_argument("--suite-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir or latest_suite_dir()
    out_dir = args.out or default_output_dir(suite_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    raw = load_raw(suite_dir)
    figure_paired_clone_assay(out_dir)
    figure_task_schematics(out_dir)
    figure_network_architecture(out_dir)
    figure_experiment_timeline(out_dir)
    figure_training_curves(raw, out_dir)
    figure_trained_margin_distribution(raw, out_dir)
    figure_protocol_schedule_raster(out_dir)
    figure_protocol_dose_summary(raw, out_dir)
    figure_protocol_temporal_signatures(out_dir)
    figure_metric_definitions(out_dir)
    figure_sample_structure(raw, out_dir)
    figure_learning_phase_summary(raw, out_dir)
    figure_no_reset_stability(raw, out_dir)
    figure_activity_perturbation(raw, out_dir)
    figure_synaptic_displacement(raw, out_dir)
    figure_training_axis_projection(raw, out_dir)
    figure_absolute_scores(raw, out_dir)
    figure_score_dents(raw, out_dir)
    figure_criterion_margins(raw, out_dir)
    figure_margin_consumed(raw, out_dir)
    figure_forgetting_incidence(raw, out_dir)
    figure_relearning_burden(raw, out_dir)
    figure_positive_control_naive_weight(raw, out_dir)
    figure_outcome_summary_matrix(raw, out_dir)
    figure_weight_behavior_dissociation(raw, out_dir)
    figure_activity_dissociation(raw, out_dir)
    figure_seed_effects(raw, out_dir)
    figure_criterion_sensitivity(raw, out_dir)
    figure_dose_response(raw, out_dir)
    figure_top_protocols(raw, out_dir)
    write_index(out_dir, suite_dir, raw)
    print(f"suite_dir={suite_dir}")
    print(f"out_dir={out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()

