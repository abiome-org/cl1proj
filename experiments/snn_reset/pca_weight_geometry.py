"""PCA and orthogonality summaries for SNN reset weight geometry.

This script consumes the validated n=50 manifold and reset-sweep artifacts. It
uses the fixed 64x64 channel-pair representation saved by ``manifold_analysis``
for cross-seed PCA, and the per-seed reset-sweep scalar weight metrics for
reset-to-memory-axis orthogonality.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import RESULTS_DIR, git_commit
from figlib import PALETTE, TASK_COLORS, TASK_LABELS, TASK_ORDER, apply_style, save

DEFAULT_MANIFOLD_DIR = RESULTS_DIR / "manifold_analysis_final_n50_modal_20260620T004937Z_combined"
DEFAULT_GRID_DIR = RESULTS_DIR / "snn_reset_final_n50_all_tasks_20260620T022002Z"
THRESHOLDS = (0.5, 0.8, 0.9, 0.95)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " "))


def delta_files(manifold_dir: Path) -> list[Path]:
    files = sorted((manifold_dir / "delta_vectors").glob("*_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"No delta vector files found in {manifold_dir / 'delta_vectors'}")
    return files


def parse_delta_name(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_seed(\d+)\.npz", path.name)
    if match is None:
        raise ValueError(f"Unexpected delta vector filename: {path.name}")
    return match.group(1), int(match.group(2))


def load_delta_tables(manifold_dir: Path, tasks: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    records: dict[str, list[dict[str, Any]]] = {}
    vectors: dict[str, list[np.ndarray]] = {}
    selected = set(tasks) if tasks else None
    for path in delta_files(manifold_dir):
        task, seed = parse_delta_name(path)
        if selected is not None and task not in selected:
            continue
        with np.load(path) as payload:
            key = "channel_delta" if "channel_delta" in payload.files else "delta"
            vector = np.asarray(payload[key], dtype=np.float64)
        records.setdefault(task, []).append({"task": task, "seed": seed, "delta_path": str(path)})
        vectors.setdefault(task, []).append(vector)

    if not vectors:
        raise ValueError("No selected delta vectors were loaded.")

    tables: dict[str, pd.DataFrame] = {}
    matrices: dict[str, np.ndarray] = {}
    for task in sorted(vectors):
        # records and vectors are appended together; sorting by seed needs the same permutation.
        sorted_pairs = sorted(zip(records[task], vectors[task], strict=True), key=lambda pair: int(pair[0]["seed"]))
        tables[task] = pd.DataFrame([pair[0] for pair in sorted_pairs]).reset_index(drop=True)
        matrices[task] = np.stack([pair[1] for pair in sorted_pairs], axis=0)
        if matrices[task].ndim != 2:
            raise ValueError(f"Expected 2-D matrix for {task}, got {matrices[task].shape}")
    return tables, matrices


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=np.float64), where=norms > 0.0)


def explained_from_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(values.shape, np.nan, dtype=np.float64)
    return values / total


def pca_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    explained = explained_from_values(variance)

    _, raw_singular, raw_vt = np.linalg.svd(matrix, full_matrices=False)
    raw_variance = np.square(raw_singular)
    raw_explained = explained_from_values(raw_variance)

    scores = centered @ vt.T
    raw_scores = matrix @ raw_vt.T
    return {
        "mean": mean,
        "centered": centered,
        "singular": singular,
        "vt": vt,
        "variance": variance,
        "explained": explained,
        "scores": scores,
        "raw_singular": raw_singular,
        "raw_vt": raw_vt,
        "raw_variance": raw_variance,
        "raw_explained": raw_explained,
        "raw_scores": raw_scores,
    }


def participation_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    denom = float(np.sum(np.square(values)))
    if total <= 0.0 or denom <= 0.0:
        return float("nan")
    return float(total * total / denom)


def components_for_threshold(explained: np.ndarray, threshold: float) -> int:
    explained = np.asarray(explained, dtype=np.float64)
    if explained.size == 0 or not np.isfinite(explained).any():
        return 0
    cumulative = np.nancumsum(explained)
    hits = np.flatnonzero(cumulative >= float(threshold))
    return int(hits[0] + 1) if hits.size else int(explained.size)


def upper_triangle_values(square: np.ndarray) -> np.ndarray:
    if square.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    indices = np.triu_indices(square.shape[0], k=1)
    return square[indices]


def angle_degrees(cosine: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(clipped))


def summarize_vector(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_sd": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q025": float("nan"),
            f"{prefix}_q975": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q025": float(np.quantile(values, 0.025)),
        f"{prefix}_q975": float(np.quantile(values, 0.975)),
    }


def bootstrap_pca_metrics(matrix: np.ndarray, *, rng: np.random.Generator, replicates: int) -> dict[str, float]:
    if replicates <= 0 or matrix.shape[0] < 3:
        return {}

    rows: list[dict[str, float]] = []
    n = matrix.shape[0]
    for _ in range(int(replicates)):
        sample = matrix[rng.integers(0, n, size=n)]
        pca = pca_from_matrix(sample)
        explained = pca["explained"]
        row = {
            "pca_top1": float(np.nansum(explained[:1])),
            "pca_top2": float(np.nansum(explained[:2])),
            "pca_top5": float(np.nansum(explained[:5])),
            "pca_participation_ratio": participation_ratio(pca["variance"]),
        }
        for threshold in THRESHOLDS:
            row[f"pc_count_{int(threshold * 100)}"] = float(components_for_threshold(explained, threshold))
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=np.float64)
        summary[f"{column}_bootstrap_mean"] = float(np.mean(values))
        summary[f"{column}_bootstrap_q025"] = float(np.quantile(values, 0.025))
        summary[f"{column}_bootstrap_q975"] = float(np.quantile(values, 0.975))
    return summary


def pca_summaries(
    tables: dict[str, pd.DataFrame],
    matrices: dict[str, np.ndarray],
    *,
    bootstrap: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    summary_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    paper: dict[str, Any] = {}

    for task, matrix in matrices.items():
        seeds = tables[task]["seed"].astype(int).to_numpy()
        pca = pca_from_matrix(matrix)
        explained = pca["explained"]
        raw_explained = pca["raw_explained"]
        norms = np.linalg.norm(matrix, axis=1)
        normalized = normalize_rows(matrix)
        pair_cosines = upper_triangle_values(normalized @ normalized.T)
        pair_angles = angle_degrees(pair_cosines)

        mean_axis = pca["mean"]
        mean_axis_norm = float(np.linalg.norm(mean_axis))
        if mean_axis_norm > 0.0:
            cos_to_mean = matrix @ (mean_axis / mean_axis_norm) / np.maximum(norms, 1e-12)
        else:
            cos_to_mean = np.full(matrix.shape[0], np.nan)
        angle_to_mean = angle_degrees(cos_to_mean)

        row: dict[str, Any] = {
            "task": task,
            "task_label": task_label(task),
            "seeds": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "delta_norm_mean": float(np.mean(norms)),
            "delta_norm_sd": float(np.std(norms, ddof=1)) if matrix.shape[0] > 1 else 0.0,
            "mean_axis_norm": mean_axis_norm,
            "mean_axis_fraction_of_mean_norm": float(mean_axis_norm / max(float(np.mean(norms)), 1e-12)),
            "mean_axis_cosine_mean": float(np.nanmean(cos_to_mean)),
            "mean_axis_angle_deg_mean": float(np.nanmean(angle_to_mean)),
            "pca_top1_variance_explained": float(np.nansum(explained[:1])),
            "pca_top2_variance_explained": float(np.nansum(explained[:2])),
            "pca_top3_variance_explained": float(np.nansum(explained[:3])),
            "pca_top5_variance_explained": float(np.nansum(explained[:5])),
            "pca_participation_ratio": participation_ratio(pca["variance"]),
            "svd_top1_variance_explained": float(np.nansum(raw_explained[:1])),
            "svd_top2_variance_explained": float(np.nansum(raw_explained[:2])),
            "svd_top5_variance_explained": float(np.nansum(raw_explained[:5])),
            "svd_participation_ratio": participation_ratio(pca["raw_variance"]),
            "centered_singular_values_top10": "|".join(f"{value:.6g}" for value in pca["singular"][:10]),
            "uncentered_singular_values_top10": "|".join(f"{value:.6g}" for value in pca["raw_singular"][:10]),
        }
        for threshold in THRESHOLDS:
            row[f"pca_components_{int(threshold * 100)}"] = components_for_threshold(explained, threshold)
        row.update(summarize_vector(pair_cosines, "pairwise_cosine"))
        row.update(summarize_vector(pair_angles, "pairwise_angle_deg"))
        row.update(bootstrap_pca_metrics(matrix, rng=rng, replicates=bootstrap))
        summary_rows.append(row)

        for pc_index, fraction in enumerate(explained, start=1):
            variance_rows.append(
                {
                    "task": task,
                    "task_label": task_label(task),
                    "component": pc_index,
                    "variance_explained": float(fraction),
                    "cumulative_variance_explained": float(np.nansum(explained[:pc_index])),
                    "singular_value": float(pca["singular"][pc_index - 1]),
                }
            )

        scores = pca["scores"]
        raw_scores = pca["raw_scores"]
        for index, seed in enumerate(seeds):
            score_rows.append(
                {
                    "task": task,
                    "task_label": task_label(task),
                    "seed": int(seed),
                    "delta_norm": float(norms[index]),
                    "cosine_to_task_mean": float(cos_to_mean[index]),
                    "angle_to_task_mean_deg": float(angle_to_mean[index]),
                    "pc1": float(scores[index, 0]) if scores.shape[1] > 0 else np.nan,
                    "pc2": float(scores[index, 1]) if scores.shape[1] > 1 else np.nan,
                    "pc3": float(scores[index, 2]) if scores.shape[1] > 2 else np.nan,
                    "uncentered_pc1": float(raw_scores[index, 0]) if raw_scores.shape[1] > 0 else np.nan,
                }
            )

        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                cosine = float(normalized[i] @ normalized[j])
                pair_rows.append(
                    {
                        "task": task,
                        "task_label": task_label(task),
                        "seed_i": int(seeds[i]),
                        "seed_j": int(seeds[j]),
                        "cosine": cosine,
                        "angle_deg": float(angle_degrees(cosine)),
                    }
                )

        paper[task] = {
            "task_label": task_label(task),
            "seeds": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "pca_top1": row["pca_top1_variance_explained"],
            "pca_top5": row["pca_top5_variance_explained"],
            "pca_components_80": row["pca_components_80"],
            "pca_components_90": row["pca_components_90"],
            "pca_participation_ratio": row["pca_participation_ratio"],
            "svd_top1": row["svd_top1_variance_explained"],
            "pairwise_angle_deg_median": row["pairwise_angle_deg_median"],
            "mean_axis_angle_deg_mean": row["mean_axis_angle_deg_mean"],
        }

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(variance_rows),
        paper,
    )


def raw_trial_paths(grid_dir: Path, tasks: list[str] | None = None) -> list[Path]:
    selected = set(tasks) if tasks else None
    paths = []
    for path in sorted(grid_dir.glob("*/raw_trials.csv")):
        if selected is not None and path.parent.name not in selected:
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No raw_trials.csv files found in {grid_dir}")
    return paths


def load_reset_rows(grid_dir: Path, tasks: list[str] | None = None) -> pd.DataFrame:
    needed_columns = {
        "task_name",
        "seed",
        "protocol_id",
        "beta",
        "schedule",
        "spatial_mode",
        "duration_s",
        "current_uA",
        "forgetting_score",
        "reset_score",
        "no_reset_score",
        "relearn_savings",
        "reset_minus_no_reset_weight_norm",
        "trained_delta_norm",
        "erasure_projection_reset_vs_no_reset",
    }
    frames = []
    for path in raw_trial_paths(grid_dir, tasks):
        frame = pd.read_csv(path, usecols=lambda column: column in needed_columns)
        frame = frame.assign(task_output_dir=str(path.parent))
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True).copy()
    task_label_values = raw["task_name"].map(TASK_LABELS).fillna(raw["task_name"])
    reset_norm = raw["reset_minus_no_reset_weight_norm"].astype(float)
    trace_norm = raw["trained_delta_norm"].astype(float)
    projection = raw["erasure_projection_reset_vs_no_reset"].astype(float)
    reset_to_erasure_axis_cosine = np.divide(
        projection * trace_norm,
        reset_norm,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=reset_norm.to_numpy() > 1e-12,
    )
    reset_to_erasure_axis_cosine = np.clip(reset_to_erasure_axis_cosine, -1.0, 1.0)
    reset_to_erasure_axis_abs_cosine = np.abs(reset_to_erasure_axis_cosine)
    raw = raw.assign(
        task_label=task_label_values,
        reset_to_erasure_axis_cosine=reset_to_erasure_axis_cosine,
        reset_to_erasure_axis_abs_cosine=reset_to_erasure_axis_abs_cosine,
        reset_to_erasure_axis_angle_deg=angle_degrees(reset_to_erasure_axis_cosine),
        reset_to_erasure_axis_unsigned_angle_deg=angle_degrees(reset_to_erasure_axis_abs_cosine),
        reset_orthogonal_fraction=np.sqrt(
            np.maximum(1.0 - np.square(np.nan_to_num(reset_to_erasure_axis_cosine)), 0.0)
        ),
        reset_norm_over_trace_norm=np.divide(
            reset_norm,
            trace_norm,
            out=np.full(len(raw), np.nan, dtype=np.float64),
            where=trace_norm.to_numpy() > 1e-12,
        ),
        has_weight_motion=reset_norm > 1e-12,
    )
    return raw


def reset_summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_columns = [
        "reset_minus_no_reset_weight_norm",
        "reset_norm_over_trace_norm",
        "erasure_projection_reset_vs_no_reset",
        "reset_to_erasure_axis_cosine",
        "reset_to_erasure_axis_abs_cosine",
        "reset_to_erasure_axis_angle_deg",
        "reset_to_erasure_axis_unsigned_angle_deg",
        "reset_orthogonal_fraction",
        "forgetting_score",
        "reset_score",
        "no_reset_score",
        "relearn_savings",
    ]

    rows = []
    for task, group in raw.groupby("task_name"):
        moving = group[group["has_weight_motion"]]
        row: dict[str, Any] = {
            "task": task,
            "task_label": task_label(task),
            "protocol_seed_trials": int(len(group)),
            "moving_trials": int(len(moving)),
            "protocols": int(group["protocol_id"].nunique()),
            "seeds": int(group["seed"].nunique()),
        }
        for column in metric_columns:
            row.update(summarize_vector(moving[column].to_numpy(dtype=np.float64), column))
        rows.append(row)
    task_summary = pd.DataFrame(rows)

    protocol_summary = (
        raw.groupby(["task_name", "task_label", "protocol_id"], as_index=False)
        .agg(
            beta=("beta", "first"),
            schedule=("schedule", "first"),
            spatial_mode=("spatial_mode", "first"),
            duration_s=("duration_s", "first"),
            current_uA=("current_uA", "first"),
            seeds=("seed", "nunique"),
            forgetting_score_mean=("forgetting_score", "mean"),
            forgetting_score_sd=("forgetting_score", "std"),
            reset_score_mean=("reset_score", "mean"),
            reset_minus_no_reset_weight_norm_mean=("reset_minus_no_reset_weight_norm", "mean"),
            reset_norm_over_trace_norm_mean=("reset_norm_over_trace_norm", "mean"),
            erasure_projection_reset_vs_no_reset_mean=("erasure_projection_reset_vs_no_reset", "mean"),
            reset_to_erasure_axis_cosine_mean=("reset_to_erasure_axis_cosine", "mean"),
            reset_to_erasure_axis_abs_cosine_mean=("reset_to_erasure_axis_abs_cosine", "mean"),
            reset_to_erasure_axis_angle_deg_mean=("reset_to_erasure_axis_angle_deg", "mean"),
            reset_orthogonal_fraction_mean=("reset_orthogonal_fraction", "mean"),
            relearn_savings_mean=("relearn_savings", "mean"),
        )
        .reset_index(drop=True)
    )
    best = (
        protocol_summary.sort_values(["task_name", "forgetting_score_mean"], ascending=[True, False])
        .groupby("task_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    paper = {
        row.task_name: {
            "task_label": row.task_label,
            "protocol_id": row.protocol_id,
            "forgetting_score_mean": safe_float(row.forgetting_score_mean),
            "forgetting_score_sd": safe_float(row.forgetting_score_sd),
            "reset_norm_over_trace_norm_mean": safe_float(row.reset_norm_over_trace_norm_mean),
            "reset_to_erasure_axis_cosine_mean": safe_float(row.reset_to_erasure_axis_cosine_mean),
            "reset_to_erasure_axis_abs_cosine_mean": safe_float(row.reset_to_erasure_axis_abs_cosine_mean),
            "reset_to_erasure_axis_angle_deg_mean": safe_float(row.reset_to_erasure_axis_angle_deg_mean),
            "reset_orthogonal_fraction_mean": safe_float(row.reset_orthogonal_fraction_mean),
            "relearn_savings_mean": safe_float(row.relearn_savings_mean),
        }
        for row in best.itertuples(index=False)
    }
    return task_summary, protocol_summary, best, paper


def ordered_task_labels(frame: pd.DataFrame) -> list[str]:
    labels = list(frame["task_label"].drop_duplicates())
    ordered = [label for label in TASK_ORDER if label in labels]
    ordered.extend(label for label in labels if label not in ordered)
    return ordered


def figure_pca(summary: pd.DataFrame, variance: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path) -> None:
    labels = ordered_task_labels(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), constrained_layout=True)

    plot_summary = summary.copy()
    sns.barplot(
        data=plot_summary,
        x="task_label",
        y="pca_participation_ratio",
        order=labels,
        ax=axes[0],
        color=PALETTE["blue"],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("PCA effective dimension")
    axes[0].tick_params(axis="x", rotation=30)

    limited = variance[variance["component"] <= 20].copy()
    for task_label_value in labels:
        sub = limited[limited["task_label"] == task_label_value]
        if sub.empty:
            continue
        axes[1].plot(
            sub["component"],
            sub["cumulative_variance_explained"],
            marker="o",
            ms=2.6,
            lw=1.2,
            color=TASK_COLORS.get(task_label_value, PALETTE["gray"]),
            label=task_label_value,
        )
    axes[1].axhline(0.8, color=PALETTE["gray"], ls="--", lw=0.8)
    axes[1].axhline(0.9, color=PALETTE["gray"], ls=":", lw=0.8)
    axes[1].set_xlabel("principal components")
    axes[1].set_ylabel("cumulative variance explained")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False, ncol=2, loc="lower right")

    sns.boxplot(
        data=pairwise,
        x="task_label",
        y="angle_deg",
        order=labels,
        ax=axes[2],
        color=PALETTE["light_gray"],
        fliersize=1.5,
        linewidth=0.8,
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("pairwise training-delta angle (deg)")
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle("Training-induced weight changes form reproducible but multi-dimensional channel-pair structure")
    save(fig, out_dir, "F_weight_pca_dimensionality")


def figure_scores(scores: pd.DataFrame, out_dir: Path) -> None:
    labels = ordered_task_labels(scores)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for task_label_value in labels:
        sub = scores[scores["task_label"] == task_label_value]
        if sub.empty:
            continue
        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            s=20,
            alpha=0.78,
            color=TASK_COLORS.get(task_label_value, PALETTE["gray"]),
            label=task_label_value,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.axhline(0.0, color=PALETTE["light_gray"], lw=0.8)
    ax.axvline(0.0, color=PALETTE["light_gray"], lw=0.8)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.legend(frameon=False, ncol=2)
    fig.suptitle("Seed-wise training deltas in task-specific PCA coordinates")
    save(fig, out_dir, "F_weight_pca_scores")


def figure_reset(task_summary: pd.DataFrame, reset_rows: pd.DataFrame, best: pd.DataFrame, out_dir: Path) -> None:
    labels = ordered_task_labels(task_summary)
    moving = reset_rows[reset_rows["has_weight_motion"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.35), constrained_layout=True)

    sns.boxplot(
        data=moving,
        x="task_label",
        y="reset_to_erasure_axis_abs_cosine",
        order=labels,
        ax=axes[0],
        color=PALETTE["light_gray"],
        fliersize=1.0,
        linewidth=0.8,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("|cosine to erasure axis|")
    axes[0].tick_params(axis="x", rotation=30)

    sns.scatterplot(
        data=moving.sample(min(len(moving), 3000), random_state=17) if len(moving) > 3000 else moving,
        x="reset_norm_over_trace_norm",
        y="reset_to_erasure_axis_cosine",
        hue="task_label",
        hue_order=labels,
        palette={label: TASK_COLORS.get(label, PALETTE["gray"]) for label in labels},
        ax=axes[1],
        s=14,
        alpha=0.45,
        linewidth=0,
        legend=False,
    )
    axes[1].axhline(0.0, color=PALETTE["gray"], ls="--", lw=0.8)
    axes[1].set_xlabel("reset displacement / memory-trace norm")
    axes[1].set_ylabel("signed cosine to erasure axis")

    best_plot = best.set_index("task_label").reindex(labels).reset_index()
    x_positions = np.arange(len(best_plot))
    axes[2].scatter(
        x_positions,
        best_plot["reset_to_erasure_axis_angle_deg_mean"],
        s=48,
        color=PALETTE["vermillion"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[2].axhline(90.0, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(best_plot["task_label"])
    axes[2].set_xlabel("")
    axes[2].set_ylabel("angle from erasure direction (deg)")
    y_min = float(np.nanmin(best_plot["reset_to_erasure_axis_angle_deg_mean"]))
    axes[2].set_ylim(max(0.0, y_min - 0.6), 90.35)
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle("Open-loop reset displacement is large but nearly orthogonal to the erasure direction")
    save(fig, out_dir, "F_reset_axis_orthogonality")


def write_metadata(out_dir: Path, args: argparse.Namespace, outputs: list[str]) -> None:
    metadata = {
        "run_id": out_dir.name,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA and orthogonality analysis for SNN reset weight geometry.")
    parser.add_argument("--manifold-dir", type=Path, default=DEFAULT_MANIFOLD_DIR)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=20260629)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or RESULTS_DIR / f"pca_weight_geometry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    tables, matrices = load_delta_tables(args.manifold_dir, args.tasks)
    pca_summary, pca_scores, pairwise, variance, pca_paper = pca_summaries(
        tables,
        matrices,
        bootstrap=max(int(args.bootstrap), 0),
        random_seed=int(args.random_seed),
    )
    reset_rows = load_reset_rows(args.grid_dir, list(matrices))
    reset_task_summary, reset_protocol_summary, best_protocols, reset_paper = reset_summaries(reset_rows)

    pca_summary.to_csv(out_dir / "pca_task_summary.csv", index=False)
    pca_scores.to_csv(out_dir / "pca_seed_scores.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_delta_cosines.csv", index=False)
    variance.to_csv(out_dir / "pca_variance_curves.csv", index=False)
    reset_rows.to_csv(out_dir / "reset_orthogonality.csv", index=False)
    reset_task_summary.to_csv(out_dir / "reset_orthogonality_task_summary.csv", index=False)
    reset_protocol_summary.to_csv(out_dir / "reset_orthogonality_protocol_summary.csv", index=False)
    best_protocols.to_csv(out_dir / "reset_orthogonality_best_protocols.csv", index=False)

    figure_pca(pca_summary, variance, pairwise, out_dir)
    figure_scores(pca_scores, out_dir)
    figure_reset(reset_task_summary, reset_rows, best_protocols, out_dir)

    paper_numbers = {
        "pca": pca_paper,
        "reset_best_protocols": reset_paper,
        "notes": {
            "pca_representation": "64x64 channel-pair mean weight-delta vectors from manifold_analysis channel_delta arrays",
            "reset_cosine_definition": "dot(reset_minus_no_reset, W_naive - W_trained) / (norm(reset_minus_no_reset) * norm(W_trained - W_naive))",
            "bootstrap_replicates": int(args.bootstrap),
        },
    }
    (out_dir / "paper_numbers.json").write_text(json.dumps(paper_numbers, indent=2, sort_keys=True), encoding="utf-8")

    outputs = [
        "metadata.json",
        "paper_numbers.json",
        "pca_task_summary.csv",
        "pca_seed_scores.csv",
        "pairwise_delta_cosines.csv",
        "pca_variance_curves.csv",
        "reset_orthogonality.csv",
        "reset_orthogonality_task_summary.csv",
        "reset_orthogonality_protocol_summary.csv",
        "reset_orthogonality_best_protocols.csv",
        "F_weight_pca_dimensionality.pdf",
        "F_weight_pca_dimensionality.png",
        "F_weight_pca_scores.pdf",
        "F_weight_pca_scores.png",
        "F_reset_axis_orthogonality.pdf",
        "F_reset_axis_orthogonality.png",
    ]
    write_metadata(out_dir, args, outputs)
    print(json.dumps({"event": "complete", "out_dir": str(out_dir), "outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
