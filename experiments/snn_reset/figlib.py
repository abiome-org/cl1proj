"""Shared figure support for the SNN reset suite: palette, style, suite IO, and
the loaders/summaries used by both ``figures.py`` and ``relearning_analysis.py``.

This is the only place the figure-generation scripts share state. Keeping it
here (rather than importing one script from another) is what lets each script
stay a thin, single-purpose entrypoint.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
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
TASK_ORDER = ["Association", "Pattern"]
SCHEDULE_LABELS = {
    "static": "static",
    "alternating_blue_red": "blue-red",
    "epoch_pause": "epoch-pause",
    "chirp": "chirp",
    "gated_burst": "gated burst",
}
SCHEDULE_ORDER = ["static", "blue-red", "epoch-pause", "chirp", "gated burst"]
BETA_LABELS = {-2.0: "beta=-2", 0.0: "beta=0", 2.0: "beta=2"}
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
TASK_COLORS = {"Association": PALETTE["blue"], "Pattern": PALETTE["orange"]}
BRANCH_COLORS = {"no reset": PALETTE["gray"], "reset": PALETTE["vermillion"]}
SCHEDULE_COLORS = {
    "static": PALETTE["blue"],
    "blue-red": PALETTE["sky"],
    "epoch-pause": PALETTE["green"],
    "chirp": PALETTE["orange"],
    "gated burst": PALETTE["vermillion"],
}


def latest_suite_dir() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and (path / "all_task_summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No modular SNN reset suite output with all_task_summary.csv found.")
    relearning = [path for path in candidates if (path / "relearning_analysis" / "raw_relearning_trials.csv").exists()]
    return sorted(relearning or candidates)[-1]


def default_output_dir(suite_dir: Path, kind: str = "paper_figures") -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return Path.home() / "Desktop" / f"cl1_snn_reset_{kind}_{suite_dir.name}_{stamp}"


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=240)
    plt.close(fig)


def energy_cost_uC(frame: pd.DataFrame) -> pd.Series:
    """Nominal injected charge per evaluation (uC), summed over both pulse phases."""
    return (
        frame["current_uA"].abs()
        * frame["pulse_width_us"].astype(float)
        * 1e-6
        * frame["total_pulses"].astype(float)
        * 2.0
    )


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


def suite_neuron_count(suite_dir: Path) -> int:
    for metadata_path in sorted(suite_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        if "neurons" in config:
            return int(config["neurons"])
    metadata = json.loads((suite_dir / "metadata.json").read_text(encoding="utf-8"))
    argv = list(metadata.get("argv", []))
    if "--neurons" in argv:
        index = argv.index("--neurons")
        if index + 1 < len(argv):
            return int(argv[index + 1])
    return 10_000


def load_raw(suite_dir: Path) -> pd.DataFrame:
    validate_suite_dir(suite_dir)
    frames = []
    for raw_path in sorted(suite_dir.glob("*/raw_trials.csv")):
        frame = pd.read_csv(raw_path).copy()
        task_output_dir = pd.Series([str(raw_path.parent)] * len(frame), name="task_output_dir")
        frames.append(pd.concat([frame, task_output_dir], axis=1))
    if not frames:
        raise FileNotFoundError(f"No task raw_trials.csv files found under {suite_dir}")
    raw = pd.concat(frames, ignore_index=True).copy()
    n_neurons = suite_neuron_count(suite_dir)
    raw["task_label"] = raw["task_name"].map(TASK_LABELS).fillna(raw["task_name"])
    raw["schedule_label"] = raw["schedule"].map(SCHEDULE_LABELS).fillna(raw["schedule"])
    raw["beta_label"] = raw["beta"].astype(float).map(BETA_LABELS).fillna(raw["beta"].map(lambda value: f"beta={value:g}"))
    raw["score_dent"] = raw["no_reset_score"] - raw["reset_score"]
    raw["criterion_margin"] = raw["reset_score"] - raw["criterion_score"]
    raw["no_reset_margin"] = raw["no_reset_score"] - raw["criterion_score"]
    raw["margin_consumed"] = np.where(
        raw["no_reset_margin"] > 1e-12,
        raw["score_dent"] / raw["no_reset_margin"],
        np.nan,
    )
    raw["abs_weight_effect"] = raw["reset_minus_no_reset_weight_norm"].abs()
    raw["log_weight_effect"] = np.log10(1.0 + raw["abs_weight_effect"])
    raw["extra_spikes_per_s"] = raw["reset_window_neuron_spikes_delta"] / raw["duration_s"].clip(lower=1e-9)
    raw["extra_spikes_per_neuron_s"] = raw["extra_spikes_per_s"] / max(float(n_neurons), 1.0)
    raw["energy_cost_uC"] = energy_cost_uC(raw)
    raw["erasure_parallel_norm"] = (raw["erasure_projection_reset_vs_no_reset"] * raw["trained_delta_norm"]).abs()
    parallel_sq = np.square(raw["erasure_parallel_norm"].fillna(0.0))
    total_sq = np.square(raw["reset_minus_no_reset_weight_norm"].fillna(0.0))
    raw["erasure_orthogonal_norm"] = np.sqrt(np.maximum(total_sq - parallel_sq, 0.0))
    raw["relearn_trials"] = raw["relearn_trials"].fillna(0.0)
    raw["relearn_savings"] = raw["relearn_savings"].fillna(1.0)
    raw["criterion_forget"] = raw["criterion_forget"].astype(bool)
    raw["score_drop"] = raw["score_drop"].astype(bool)
    raw["n_neurons"] = n_neurons
    return raw


def task_protocol_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["task_name", "task_label", "protocol_id"], as_index=False)
        .agg(
            beta=("beta", "first"),
            beta_label=("beta_label", "first"),
            schedule=("schedule", "first"),
            schedule_label=("schedule_label", "first"),
            spatial_mode=("spatial_mode", "first"),
            duration_s=("duration_s", "first"),
            current_uA=("current_uA", "first"),
            pulse_width_us=("pulse_width_us", "first"),
            total_pulses=("total_pulses", "mean"),
            energy_cost_uC=("energy_cost_uC", "mean"),
            baseline_score=("baseline_score", "mean"),
            trained_score=("trained_score", "mean"),
            no_reset_score=("no_reset_score", "mean"),
            reset_score=("reset_score", "mean"),
            score_dent=("score_dent", "mean"),
            criterion_score=("criterion_score", "first"),
            criterion_margin=("criterion_margin", "mean"),
            min_criterion_margin=("criterion_margin", "min"),
            margin_consumed=("margin_consumed", "mean"),
            score_drop=("score_drop", "mean"),
            criterion_forget=("criterion_forget", "mean"),
            relearn_trials=("relearn_trials", "mean"),
            relearn_savings=("relearn_savings", "mean"),
            reset_minus_no_reset_weight_norm=("reset_minus_no_reset_weight_norm", "mean"),
            erasure_projection_reset_vs_no_reset=("erasure_projection_reset_vs_no_reset", "mean"),
            erasure_parallel_norm=("erasure_parallel_norm", "mean"),
            erasure_orthogonal_norm=("erasure_orthogonal_norm", "mean"),
            reset_window_neuron_spikes_delta=("reset_window_neuron_spikes_delta", "mean"),
            extra_spikes_per_neuron_s=("extra_spikes_per_neuron_s", "mean"),
            replicates=("seed", "count"),
        )
        .sort_values(["task_label", "score_dent"], ascending=[True, False])
        .reset_index(drop=True)
    )


def task_schedule_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["task_label", "schedule_label"], as_index=False)
        .agg(
            score_dent=("score_dent", "mean"),
            reset_score=("reset_score", "mean"),
            no_reset_score=("no_reset_score", "mean"),
            criterion_score=("criterion_score", "first"),
            criterion_margin=("criterion_margin", "mean"),
            min_criterion_margin=("criterion_margin", "min"),
            margin_consumed=("margin_consumed", "mean"),
            criterion_forget=("criterion_forget", "mean"),
            relearn_trials=("relearn_trials", "mean"),
            reset_minus_no_reset_weight_norm=("reset_minus_no_reset_weight_norm", "mean"),
            erasure_projection_reset_vs_no_reset=("erasure_projection_reset_vs_no_reset", "mean"),
            erasure_parallel_norm=("erasure_parallel_norm", "mean"),
            erasure_orthogonal_norm=("erasure_orthogonal_norm", "mean"),
            reset_window_neuron_spikes_delta=("reset_window_neuron_spikes_delta", "mean"),
            extra_spikes_per_neuron_s=("extra_spikes_per_neuron_s", "mean"),
            total_pulses=("total_pulses", "mean"),
            energy_cost_uC=("energy_cost_uC", "mean"),
            replicates=("seed", "count"),
        )
        .reset_index(drop=True)
    )


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#D1D5DB",
    fontsize: float = 8.5,
    weight: str = "regular",
) -> None:
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=PALETTE["black"],
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#4B5563") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=2, shrinkB=2),
    )


def facet_stripplot(
    raw: pd.DataFrame,
    out_dir: Path,
    *,
    y: str,
    ylabel: str,
    name: str,
    hue: str = "current_uA",
    palette=None,
    legend_title: str = "uA",
    hlines: tuple[tuple[float, str, str], ...] = ((0.0, "--", "black"),),
    figsize: tuple[float, float] = (9.2, 4.0),
) -> None:
    """One stripplot per learned task (``TASK_ORDER``), x = stimulation schedule.

    Collapses the family of paired schedule panels (score dent, criterion margin,
    margin consumed, ...) that differ only in the y column, label, and guide lines.
    """
    palette = palette or [PALETTE["sky"], PALETTE["vermillion"]]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
    for ax, task_label in zip(axes, TASK_ORDER, strict=False):
        frame = raw[raw["task_label"] == task_label]
        sns.stripplot(
            frame,
            x="schedule_label",
            y=y,
            hue=hue,
            order=SCHEDULE_ORDER,
            palette=palette,
            dodge=True,
            jitter=0.18,
            alpha=0.72,
            s=3.8,
            ax=ax,
        )
        for level, style, color in hlines:
            ax.axhline(level, color=PALETTE[color], ls=style, lw=0.85, alpha=0.72)
        ax.set_title(task_label)
        ax.set_xlabel("stimulation schedule")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, title=legend_title)
    save(fig, out_dir, name)


def schedule_stripplot(
    raw: pd.DataFrame,
    out_dir: Path,
    *,
    y: str,
    ylabel: str,
    title: str,
    name: str,
    zero_line: bool = False,
) -> None:
    """Single-axis stripplot, x = schedule, hue = task, y = one perturbation metric."""
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    sns.stripplot(
        raw,
        x="schedule_label",
        y=y,
        hue="task_label",
        order=SCHEDULE_ORDER,
        palette=TASK_COLORS,
        dodge=True,
        jitter=0.20,
        alpha=0.68,
        s=3.8,
        ax=ax,
    )
    if zero_line:
        ax.axhline(0.0, color=PALETTE["black"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("stimulation schedule")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, title="")
    save(fig, out_dir, name)
