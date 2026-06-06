from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
TASK_LABELS = {
    "evoked_channel_response": "Evoked",
    "conditioned_electrode_association": "Association",
    "delayed_conditioned_response": "Delayed",
    "pattern_discrimination": "Pattern",
    "temporal_order_discrimination": "Order",
}
PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#6B7280",
    "black": "#111827",
}


def latest_suite_dir() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and (path / "all_task_summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No modular SNN reset suite output with all_task_summary.csv found.")
    return sorted(candidates)[-1]


def default_output_dir(suite_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return Path.home() / "Desktop" / f"cl1_snn_reset_figures_{suite_dir.name}_{stamp}"


def apply_style() -> None:
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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=220)
    plt.close(fig)


def validate_suite_dir(suite_dir: Path) -> None:
    metadata_path = suite_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Suite metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_results = metadata.get("tasks", [])
    if metadata.get("status", "complete") == "failed":
        raise ValueError(f"Suite is marked failed: {suite_dir}")
    failed_tasks = [task for task in task_results if int(task.get("returncode", 1)) != 0]
    if failed_tasks:
        raise ValueError(f"Suite contains failed task scripts: {failed_tasks}")
    for task in task_results:
        task_output_dir = Path(task["output_dir"])
        if not task_output_dir.is_absolute() and not task_output_dir.exists():
            task_output_dir = suite_dir / str(task["task"])
        progress_path = task_output_dir / "progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"Task progress metadata is missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_jobs", -1))
        expected = int(progress.get("job_count", -2))
        pending = int(progress.get("pending_jobs", -1))
        if completed != expected or pending != 0:
            raise ValueError(f"Task output is incomplete: {progress_path}")


def load_raw(suite_dir: Path) -> pd.DataFrame:
    validate_suite_dir(suite_dir)
    frames = []
    for raw_path in sorted(suite_dir.glob("*/raw_trials.csv")):
        frame = pd.read_csv(raw_path).copy()
        task_output_dir = pd.Series(
            [str(raw_path.parent)] * len(frame),
            name="task_output_dir",
        )
        frames.append(pd.concat([frame, task_output_dir], axis=1))
    if not frames:
        raise FileNotFoundError(f"No task raw_trials.csv files found under {suite_dir}")
    raw = pd.concat(frames, ignore_index=True).copy()
    abs_weight_effect = raw["reset_minus_no_reset_weight_norm"].abs()
    return raw.assign(
        task_label=raw["task_name"].map(TASK_LABELS).fillna(raw["task_name"]),
        abs_reset_score_effect=raw["reset_minus_no_reset_score"].abs(),
        abs_weight_effect=abs_weight_effect,
        log_weight_effect=np.log10(1.0 + abs_weight_effect),
        extra_spikes_per_s=raw["reset_window_neuron_spikes_delta"] / raw["duration_s"].clip(lower=1e-9),
    )


def figure_task_learning(raw: pd.DataFrame, out_dir: Path) -> None:
    metrics = raw.groupby(["task_name", "task_label"], as_index=False).agg(
        baseline=("baseline_score", "mean"),
        trained=("trained_score", "mean"),
        naive_weight_control=("naive_weight_control_score", "mean"),
        no_reset=("no_reset_score", "mean"),
        reset=("reset_score", "mean"),
        criterion_rate=("training_reached_criterion", "mean"),
    )
    melted = metrics.melt(
        id_vars=["task_name", "task_label"],
        value_vars=["baseline", "trained", "naive_weight_control", "no_reset", "reset"],
        var_name="phase",
        value_name="task_score",
    )
    phase_order = ["baseline", "trained", "naive_weight_control", "no_reset", "reset"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    sns.barplot(
        melted,
        x="task_label",
        y="task_score",
        hue="phase",
        hue_order=phase_order,
        palette=[
            PALETTE["gray"],
            PALETTE["blue"],
            PALETTE["vermillion"],
            PALETTE["green"],
            PALETTE["orange"],
        ],
        ax=axes[0],
    )
    axes[0].axhline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.65)
    axes[0].set_xlabel("task")
    axes[0].set_ylabel("positive - negative probe score")
    axes[0].set_title("Task score by phase")
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["baseline", "trained", "naive-weight", "no reset", "reset"]
    axes[0].legend(handles, labels, frameon=False, title="", loc="lower left")
    sns.barplot(
        metrics,
        x="task_label",
        y="criterion_rate",
        color=PALETTE["sky"],
        ax=axes[1],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("task")
    axes[1].set_ylabel("fraction of protocol x seed jobs")
    axes[1].set_title("Training criterion reached")
    save(fig, out_dir, "task_learning_summary")


def figure_reset_effect_scatter(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.scatterplot(
        raw,
        x="log_weight_effect",
        y="reset_minus_no_reset_score",
        hue="task_label",
        style="schedule",
        s=34,
        alpha=0.78,
        ax=ax,
    )
    ax.axhline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("log10(1 + ||Wreset - Wno-reset||)")
    ax.set_ylabel("reset - no-reset task score")
    ax.set_title("Behavioral effect versus paired weight displacement")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="")
    save(fig, out_dir, "reset_effect_scatter")


def figure_activity_weight_coupling(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sns.scatterplot(
        raw,
        x="extra_spikes_per_s",
        y="reset_minus_no_reset_weight_norm",
        hue="task_label",
        style="schedule",
        s=34,
        alpha=0.78,
        ax=ax,
    )
    ax.axhline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axvline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_xscale("symlog", linthresh=10.0)
    ax.set_xlabel("reset - no-reset spikes / s")
    ax.set_ylabel("||Wreset - Wno-reset||")
    ax.set_title("Activity perturbation and weight displacement")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="")
    save(fig, out_dir, "activity_weight_coupling")


def figure_schedule_summary(raw: pd.DataFrame, out_dir: Path) -> None:
    grouped = raw.groupby(["task_label", "schedule"], as_index=False).agg(
        score_effect=("reset_minus_no_reset_score", "mean"),
        weight_effect=("reset_minus_no_reset_weight_norm", "mean"),
        spike_effect=("reset_window_neuron_spikes_delta", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    sns.barplot(grouped, x="schedule", y="score_effect", hue="task_label", ax=axes[0])
    axes[0].axhline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.65)
    axes[0].set_title("Task score effect")
    axes[0].set_xlabel("schedule")
    axes[0].set_ylabel("reset - no-reset")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(grouped, x="schedule", y="weight_effect", hue="task_label", ax=axes[1])
    axes[1].set_title("Weight displacement")
    axes[1].set_xlabel("schedule")
    axes[1].set_ylabel("||Wreset - Wno-reset||")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend_.remove()
    sns.barplot(grouped, x="schedule", y="spike_effect", hue="task_label", ax=axes[2])
    axes[2].set_title("Reset-window spikes")
    axes[2].set_xlabel("schedule")
    axes[2].set_ylabel("reset - no-reset")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend_.remove()
    axes[0].legend(frameon=False, title="")
    save(fig, out_dir, "schedule_summary")


def figure_top_protocols(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = raw.groupby(["task_label", "protocol_id"], as_index=False).agg(
        score_effect=("reset_minus_no_reset_score", "mean"),
        weight_effect=("reset_minus_no_reset_weight_norm", "mean"),
        spike_effect=("reset_window_neuron_spikes_delta", "mean"),
    )
    rows = []
    for _, frame in summary.groupby("task_label"):
        rows.append(frame.assign(rank_metric=frame["score_effect"].abs()).nlargest(5, "rank_metric"))
    top = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if top.empty:
        return
    top["protocol_short"] = top["protocol_id"].str.replace("_", " ").str.replace("b", "b=", n=1)
    top["row_label"] = top["task_label"] + " | " + top["protocol_short"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    sns.barplot(
        top,
        y="row_label",
        x="score_effect",
        hue="task_label",
        dodge=False,
        ax=axes[0],
    )
    axes[0].axvline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.65)
    axes[0].set_xlabel("reset - no-reset task score")
    axes[0].set_ylabel("")
    axes[0].set_title("Largest behavioral effects")
    axes[0].legend(frameon=False, title="")
    sns.barplot(
        top,
        y="row_label",
        x="weight_effect",
        hue="task_label",
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("||Wreset - Wno-reset||")
    axes[1].set_ylabel("")
    axes[1].set_title("Weight displacement for same protocols")
    axes[1].legend_.remove()
    save(fig, out_dir, "top_protocols")


def write_index(out_dir: Path, suite_dir: Path) -> None:
    rows = []
    for pdf in sorted(out_dir.glob("*.pdf")):
        rows.append(
            {
                "figure": pdf.stem,
                "pdf": str(pdf),
                "png": str(pdf.with_suffix(".png")),
                "source_suite": str(suite_dir),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "figure_index.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for a modular SNN reset grid result.")
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
    figure_task_learning(raw, out_dir)
    figure_reset_effect_scatter(raw, out_dir)
    figure_activity_weight_coupling(raw, out_dir)
    figure_schedule_summary(raw, out_dir)
    figure_top_protocols(raw, out_dir)
    write_index(out_dir, suite_dir)
    print(f"suite_dir={suite_dir}")
    print(f"out_dir={out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
