from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figures import PALETTE, TASK_LABELS, apply_style, latest_suite_dir, save, validate_suite_dir


REQUIRED_COLUMNS = {
    "forgetting_score",
    "reset_score",
    "no_reset_score",
    "relearn_trials",
    "relearn_savings",
    "relearn_reached_criterion",
}


def default_output_dir(suite_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return Path.home() / "Desktop" / f"cl1_snn_reset_relearning_{suite_dir.name}_{stamp}"


def pareto_mask(
    frame: pd.DataFrame,
    *,
    maximize: list[str],
    minimize: list[str],
) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    objective_columns = []
    for column in maximize:
        objective_columns.append(frame[column].to_numpy(dtype=float))
    for column in minimize:
        objective_columns.append(-frame[column].to_numpy(dtype=float))
    objectives = np.column_stack(objective_columns)
    keep = np.ones(len(frame), dtype=bool)
    for index, row in enumerate(objectives):
        if not keep[index]:
            continue
        dominates = np.all(objectives >= row, axis=1) & np.any(objectives > row, axis=1)
        if np.any(dominates):
            keep[index] = False
    return keep


def place_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="")


def load_relearning_raw(suite_dir: Path) -> pd.DataFrame:
    validate_suite_dir(suite_dir)
    frames = []
    for raw_path in sorted(suite_dir.glob("*/raw_trials.csv")):
        frame = pd.read_csv(raw_path).copy()
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            continue
        metadata = json.loads((raw_path.parent / "metadata.json").read_text(encoding="utf-8"))
        criterion_score = float(metadata["task_regime"]["criterion_score"])
        frames.append(
            frame.assign(
                task_output_dir=str(raw_path.parent),
                criterion_score=criterion_score,
            )
        )
    if not frames:
        raise FileNotFoundError(f"No relearning raw_trials.csv files found under {suite_dir}")
    raw = pd.concat(frames, ignore_index=True).copy()
    if "forgetting_score" not in raw.columns:
        raw["forgetting_score"] = raw["no_reset_score"] - raw["reset_score"]
    energy_cost_uC = (
        raw["current_uA"].abs()
        * raw["pulse_width_us"].astype(float)
        * 1e-6
        * raw["total_pulses"].astype(float)
        * 2.0
    )
    return raw.assign(
        task_label=raw["task_name"].map(TASK_LABELS).fillna(raw["task_name"]),
        energy_cost_uC=energy_cost_uC,
        residual_performance=raw["reset_score"],
        score_drop=raw["forgetting_score"] > 0.0,
        criterion_forget=(raw["no_reset_score"] >= raw["criterion_score"]) & (raw["reset_score"] < raw["criterion_score"]),
        made_forget=(raw["no_reset_score"] >= raw["criterion_score"]) & (raw["reset_score"] < raw["criterion_score"]),
    )


def summarize_task_protocols(raw: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "beta": "first",
        "schedule": "first",
        "spatial_mode": "first",
        "duration_s": "first",
        "current_uA": "first",
        "pulse_width_us": "first",
        "criterion_score": "first",
        "total_pulses": "mean",
        "baseline_score": "mean",
        "trained_score": "mean",
        "no_reset_score": "mean",
        "reset_score": "mean",
        "residual_performance": "mean",
        "forgetting_score": "mean",
        "score_drop": "mean",
        "criterion_forget": "mean",
        "made_forget": "mean",
        "relearn_measured": "mean",
        "relearn_trials": "mean",
        "relearn_score": "mean",
        "relearn_reached_criterion": "mean",
        "relearn_savings": "mean",
        "energy_cost_uC": "mean",
        "reset_minus_no_reset_weight_norm": "mean",
        "erasure_projection_reset_vs_no_reset": "mean",
        "reset_window_neuron_spikes_delta": "mean",
        "seed": "count",
    }
    columns = {column: op for column, op in aggregations.items() if column in raw.columns}
    return (
        raw.groupby(["task_name", "task_label", "protocol_id"], as_index=False)
        .agg(columns)
        .rename(columns={"seed": "replicates"})
        .sort_values(["task_name", "forgetting_score", "relearn_savings"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def summarize_protocols(task_summary: pd.DataFrame) -> pd.DataFrame:
    first_fields = ["beta", "schedule", "spatial_mode", "duration_s", "current_uA", "pulse_width_us"]
    mean_fields = [
        "criterion_score",
        "total_pulses",
        "trained_score",
        "no_reset_score",
        "reset_score",
        "residual_performance",
        "forgetting_score",
        "score_drop",
        "criterion_forget",
        "made_forget",
        "relearn_measured",
        "relearn_trials",
        "relearn_score",
        "relearn_reached_criterion",
        "relearn_savings",
        "energy_cost_uC",
        "reset_minus_no_reset_weight_norm",
        "erasure_projection_reset_vs_no_reset",
        "reset_window_neuron_spikes_delta",
    ]
    aggregations = {field: "first" for field in first_fields if field in task_summary.columns}
    aggregations.update({field: "mean" for field in mean_fields if field in task_summary.columns})
    aggregations["task_name"] = "count"
    return (
        task_summary.groupby("protocol_id", as_index=False)
        .agg(aggregations)
        .rename(columns={"task_name": "task_count"})
        .sort_values(["forgetting_score", "relearn_savings"], ascending=[False, True])
        .reset_index(drop=True)
    )


def compute_fronts(
    summary: pd.DataFrame,
    *,
    forgetting_threshold: float,
    eligibility_column: str,
) -> dict[str, pd.DataFrame]:
    eligible = summary[
        (summary["forgetting_score"] > forgetting_threshold)
        & (summary[eligibility_column] > 0.0)
    ].copy()
    if eligible.empty:
        return {
            "eligible": eligible,
            "forgetting_front": eligible,
            "forget_savings_front": eligible,
            "direct_forget_savings_front": eligible,
        }
    forgetting_front = eligible[
        pareto_mask(
            eligible,
            maximize=["forgetting_score"],
            minimize=["residual_performance", "energy_cost_uC"],
        )
    ].copy()
    forget_savings_front = forgetting_front[
        pareto_mask(
            forgetting_front,
            maximize=["forgetting_score"],
            minimize=["relearn_savings", "relearn_trials", "energy_cost_uC"],
        )
    ].copy()
    direct_forget_savings_front = eligible[
        pareto_mask(
            eligible,
            maximize=["forgetting_score"],
            minimize=["relearn_savings", "relearn_trials", "energy_cost_uC"],
        )
    ].copy()
    return {
        "eligible": eligible,
        "forgetting_front": forgetting_front,
        "forget_savings_front": forget_savings_front,
        "direct_forget_savings_front": direct_forget_savings_front,
    }


def write_tables(out_dir: Path, raw: pd.DataFrame, task_summary: pd.DataFrame, protocol_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    task_fronts = compute_fronts(
        task_summary,
        forgetting_threshold=0.0,
        eligibility_column="criterion_forget",
    )
    protocol_fronts = compute_fronts(
        protocol_summary,
        forgetting_threshold=0.0,
        eligibility_column="criterion_forget",
    )
    task_score_drop_fronts = compute_fronts(
        task_summary,
        forgetting_threshold=0.0,
        eligibility_column="score_drop",
    )
    protocol_score_drop_fronts = compute_fronts(
        protocol_summary,
        forgetting_threshold=0.0,
        eligibility_column="score_drop",
    )
    tables: dict[str, pd.DataFrame] = {
        "raw_relearning_trials": raw,
        "task_protocol_relearning_summary": task_summary,
        "protocol_relearning_summary": protocol_summary,
        **{f"task_{name}": frame for name, frame in task_fronts.items()},
        **{f"protocol_{name}": frame for name, frame in protocol_fronts.items()},
        **{f"task_score_drop_{name}": frame for name, frame in task_score_drop_fronts.items()},
        **{f"protocol_score_drop_{name}": frame for name, frame in protocol_score_drop_fronts.items()},
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def figure_forgetting_savings(task_summary: pd.DataFrame, front: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.scatterplot(
        task_summary,
        x="forgetting_score",
        y="relearn_savings",
        hue="task_label",
        style="schedule",
        size="energy_cost_uC",
        sizes=(28, 140),
        alpha=0.72,
        ax=ax,
    )
    if not front.empty:
        sns.scatterplot(
            front,
            x="forgetting_score",
            y="relearn_savings",
            marker="X",
            s=130,
            color=PALETTE["black"],
            label="front of forgetting front",
            ax=ax,
        )
    ax.axvline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axhline(0.0, color=PALETTE["black"], linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("immediate forgetting score (no-reset - reset)")
    ax.set_ylabel("relearning savings (lower = less retained)")
    ax.set_title("Forgetting versus savings")
    place_legend(ax)
    save(fig, out_dir, "forgetting_vs_savings")


def figure_front_protocols(front: pd.DataFrame, out_dir: Path) -> None:
    if front.empty:
        return
    plot_frame = front.copy()
    plot_frame["label"] = plot_frame["task_label"] + " | " + plot_frame["protocol_id"]
    plot_frame = plot_frame.sort_values(["forgetting_score", "relearn_savings"], ascending=[True, False])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, max(3.2, 0.34 * len(plot_frame))), constrained_layout=True)
    sns.barplot(
        plot_frame,
        y="label",
        x="forgetting_score",
        color=PALETTE["vermillion"],
        ax=axes[0],
    )
    axes[0].set_xlabel("immediate forgetting")
    axes[0].set_ylabel("")
    axes[0].set_title("Front protocols erase task score")
    sns.barplot(
        plot_frame,
        y="label",
        x="relearn_savings",
        color=PALETTE["blue"],
        ax=axes[1],
    )
    axes[1].axvline(0.0, color=PALETTE["black"], linestyle=":", linewidth=0.8, alpha=0.55)
    axes[1].set_xlabel("relearning savings")
    axes[1].set_ylabel("")
    axes[1].set_title("Same protocols minimize savings")
    save(fig, out_dir, "forget_savings_front")


def figure_protocol_level(protocol_summary: pd.DataFrame, front: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sns.scatterplot(
        protocol_summary,
        x="forgetting_score",
        y="relearn_savings",
        hue="schedule",
        size="energy_cost_uC",
        sizes=(30, 140),
        alpha=0.74,
        ax=ax,
    )
    if not front.empty:
        sns.scatterplot(
            front,
            x="forgetting_score",
            y="relearn_savings",
            marker="X",
            s=130,
            color=PALETTE["black"],
            label="protocol front",
            ax=ax,
        )
    ax.axvline(0.0, color=PALETTE["black"], linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axhline(0.0, color=PALETTE["black"], linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("mean immediate forgetting across learned tasks")
    ax.set_ylabel("mean relearning savings")
    ax.set_title("Protocol-level forgetting and relearning")
    place_legend(ax)
    save(fig, out_dir, "protocol_forgetting_vs_savings")


def write_index(out_dir: Path, suite_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    payload = {
        "suite_dir": str(suite_dir),
        "output_dir": str(out_dir),
        "tables": {
            name: {
                "path": str(out_dir / f"{name}.csv"),
                "rows": int(len(frame)),
            }
            for name, frame in sorted(tables.items())
        },
        "figures": [
            str(path)
            for path in sorted(out_dir.glob("*.pdf"))
        ],
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze relearning/savings for an SNN reset grid.")
    parser.add_argument("--suite-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir or latest_suite_dir()
    out_dir = args.out or default_output_dir(suite_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    raw = load_relearning_raw(suite_dir)
    task_summary = summarize_task_protocols(raw)
    protocol_summary = summarize_protocols(task_summary)
    tables = write_tables(out_dir, raw, task_summary, protocol_summary)
    figure_forgetting_savings(
        task_summary,
        tables["task_forget_savings_front"],
        out_dir,
    )
    figure_front_protocols(tables["task_forget_savings_front"], out_dir)
    figure_protocol_level(
        protocol_summary,
        tables["protocol_forget_savings_front"],
        out_dir,
    )
    write_index(out_dir, suite_dir, tables)
    print(f"suite_dir={suite_dir}")
    print(f"out_dir={out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
