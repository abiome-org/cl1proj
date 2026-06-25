"""Task-generic memory-geometry probes for the validated SNN reset regime.

The script trains each selected task and then runs three simulation-only probes:

1. Interpolation along the training-defined memory axis, with a matched random
   orthogonal displacement control.
2. Surgical ablations of task-relevant input-to-target pathways, plus global and
   matched-random controls.
3. PCA/SVD dimensionality summaries of seed-wise training-induced weight deltas.

Outputs CSVs, figures, and per-seed delta vectors under
``results/manifold_analysis_<stamp>/``. Use ``--resume`` for long final runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cl1_snn_reset import CultureConfig, TaskRegime, build_network, evaluate_regime, train_regime

from common import RESULTS_DIR, add_common_task_args, culture_from_args, git_commit
from figlib import PALETTE, TASK_LABELS, apply_style, save
from tasks import TASK_BUILDERS

DEFAULT_TASKS = [
    "conditioned_electrode_association",
    "pattern_discrimination",
    "overlapping_shared_target_association",
    "overlapping_shared_input_association",
    "multi_association_mapping",
    "xor_electrode_classification",
]
DEFAULT_ALPHAS = np.round(np.linspace(0.0, 1.0, 16), 4)


def safe_name(value: str) -> str:
    """Return a stable file/condition-safe identifier."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " "))


def delta_path(out_dir: Path, task: str, seed: int) -> Path:
    return out_dir / "delta_vectors" / f"{safe_name(task)}_seed{int(seed)}.npz"


def event_channels(events) -> tuple[int, ...]:
    channels = {int(channel) for event in events for channel in event.channels}
    return tuple(sorted(channels))


def positive_mappings(regime: TaskRegime) -> list[dict[str, Any]]:
    """Infer task-relevant input-target mappings from positive probes."""
    mappings = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for probe in regime.probes:
        if not probe.is_positive:
            continue
        inputs = event_channels(probe.events)
        targets = tuple(sorted(int(channel) for channel in probe.target_channels))
        key = (inputs, targets)
        if not inputs or key in seen:
            continue
        seen.add(key)
        mappings.append({"name": probe.name, "input_channels": inputs, "target_channels": targets})
    return mappings


def train_state(
    culture: CultureConfig,
    regime: TaskRegime,
    seed: int,
    *,
    warmup_s: float,
    training_repetitions: int | None,
    eval_repetitions: int | None,
    stop_at_criterion: bool,
):
    net = build_network(culture, seed=seed)
    net.advance(warmup_s * 1000.0, [], plasticity=False, record=False)
    naive = net.weights_vector()
    train_reps, trained_eval, history = train_regime(
        net,
        regime,
        max_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=stop_at_criterion,
    )
    trained = net.weights_vector()
    return net, naive, trained, train_reps, trained_eval.score, history


def score_weights(net, regime: TaskRegime, weights: np.ndarray, *, reps: int | None) -> tuple[float, np.ndarray]:
    net.set_weights(weights)
    effective = net.weights_vector()
    score = evaluate_regime(net, regime, repetitions=reps).score
    return float(score), effective


def interpolation_rows(
    net,
    regime: TaskRegime,
    naive: np.ndarray,
    trained: np.ndarray,
    *,
    task: str,
    seed: int,
    alpha_values: np.ndarray,
    eval_repetitions: int | None,
) -> list[dict[str, Any]]:
    axis = naive - trained
    distance = float(np.linalg.norm(axis))
    rng = np.random.default_rng(7919 * int(seed) + len(task))
    random_dir = rng.standard_normal(axis.shape)
    axis_norm_sq = float(axis @ axis)
    if axis_norm_sq > 0.0:
        random_dir -= (random_dir @ axis) / axis_norm_sq * axis
    random_dir /= np.linalg.norm(random_dir) + 1e-12

    rows = []
    for alpha in alpha_values:
        for direction, target in (
            ("axis", trained + float(alpha) * axis),
            ("random", trained + float(alpha) * distance * random_dir),
        ):
            score, effective = score_weights(net, regime, target, reps=eval_repetitions)
            rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "direction": direction,
                    "alpha": float(alpha),
                    "score": score,
                    "criterion": float(regime.criterion_score),
                    "distance_from_naive": float(np.linalg.norm(effective - naive)),
                    "distance_from_trained": float(np.linalg.norm(effective - trained)),
                }
            )
    return rows


def channel_edge_mask(net, source_channels: tuple[int, ...], target_channels: tuple[int, ...]) -> np.ndarray:
    channel_of = net.electrodes.nearest_channel
    source_neuron = np.isin(channel_of, list(source_channels))
    target_neuron = np.isin(channel_of, list(target_channels))
    return source_neuron[net.sources] & target_neuron[net.targets]


def channel_delta_vector(net, delta: np.ndarray) -> np.ndarray:
    """Aggregate edge deltas into a fixed channel-pair representation."""
    channel_count = int(net.electrodes.channel_count)
    channel_of = net.electrodes.nearest_channel
    source_channels = channel_of[net.sources].astype(np.int64, copy=False)
    target_channels = channel_of[net.targets].astype(np.int64, copy=False)
    flat_index = source_channels * channel_count + target_channels
    sums = np.zeros(channel_count * channel_count, dtype=np.float64)
    counts = np.zeros(channel_count * channel_count, dtype=np.float64)
    np.add.at(sums, flat_index, delta.astype(np.float64, copy=False))
    np.add.at(counts, flat_index, 1.0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0.0)


def ablation_rows(
    net,
    regime: TaskRegime,
    naive: np.ndarray,
    trained: np.ndarray,
    *,
    task: str,
    seed: int,
    eval_repetitions: int | None,
) -> list[dict[str, Any]]:
    mappings = positive_mappings(regime)
    rng = np.random.default_rng(104729 * int(seed) + len(task))
    axis = naive - trained
    axis_norm = float(np.linalg.norm(axis))

    def zeroed(mask: np.ndarray) -> np.ndarray:
        weights = trained.copy()
        weights[mask] = 0.0
        return weights

    def random_displacement(distance: float) -> np.ndarray:
        random_dir = rng.standard_normal(trained.shape)
        axis_norm_sq = float(axis @ axis)
        if axis_norm_sq > 0.0:
            random_dir -= (random_dir @ axis) / axis_norm_sq * axis
        random_dir /= np.linalg.norm(random_dir) + 1e-12
        return trained + float(distance) * random_dir

    mapping_masks = {
        f"zero_path_{safe_name(mapping['name'])}": channel_edge_mask(
            net,
            mapping["input_channels"],
            mapping["target_channels"],
        )
        for mapping in mappings
    }
    if mapping_masks:
        task_path_union = np.logical_or.reduce(tuple(mapping_masks.values()))
    else:
        task_path_union = np.zeros_like(trained, dtype=bool)

    all_input_channels = tuple(sorted({channel for mapping in mappings for channel in mapping["input_channels"]}))
    all_target_channels = tuple(sorted({channel for mapping in mappings for channel in mapping["target_channels"]}))
    input_outgoing = channel_edge_mask(net, all_input_channels, tuple(range(net.electrodes.channel_count)))
    target_incoming = channel_edge_mask(net, tuple(range(net.electrodes.channel_count)), all_target_channels)
    union_distance = float(np.linalg.norm(trained - zeroed(task_path_union)))
    edge_count = int(np.count_nonzero(task_path_union))
    random_edge_mask = np.zeros_like(task_path_union, dtype=bool)
    if edge_count:
        random_edge_mask[rng.choice(trained.size, size=edge_count, replace=False)] = True

    conditions: dict[str, np.ndarray] = {
        "trained": trained,
        "naive_replacement": naive,
        "zero_all": np.zeros_like(trained),
        "shuffle_magnitudes": np.abs(trained)[rng.permutation(trained.size)] * np.sign(trained),
        "anti_training_delta": 2.0 * trained - naive,
        "random_matched_full_trace": random_displacement(axis_norm),
        "zero_task_path_union": zeroed(task_path_union),
        "zero_all_task_inputs_outgoing": zeroed(input_outgoing),
        "zero_all_task_targets_incoming": zeroed(target_incoming),
        "random_edge_count_matched_task_path_union": zeroed(random_edge_mask),
        "random_distance_matched_task_path_union": random_displacement(union_distance),
    }
    for name, mask in mapping_masks.items():
        conditions[name] = zeroed(mask)

    edge_fraction = {
        "zero_task_path_union": float(np.mean(task_path_union)),
        "zero_all_task_inputs_outgoing": float(np.mean(input_outgoing)),
        "zero_all_task_targets_incoming": float(np.mean(target_incoming)),
        "random_edge_count_matched_task_path_union": float(np.mean(random_edge_mask)),
    }
    edge_fraction.update({name: float(np.mean(mask)) for name, mask in mapping_masks.items()})

    rows = []
    for name, weights in conditions.items():
        score, effective = score_weights(net, regime, weights, reps=eval_repetitions)
        rows.append(
            {
                "task": task,
                "seed": int(seed),
                "condition": name,
                "score": score,
                "criterion": float(regime.criterion_score),
                "distance_from_naive": float(np.linalg.norm(effective - naive)),
                "distance_from_trained": float(np.linalg.norm(effective - trained)),
                "edges_changed_fraction": edge_fraction.get(name, np.nan),
                "mapping_count": int(len(mappings)),
            }
        )
    return rows


def geometry_job(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        culture,
        regime,
        task,
        seed,
        alpha_values,
        warmup_s,
        training_repetitions,
        eval_repetitions,
        stop_at_criterion,
    ) = job
    net, naive, trained, train_reps, trained_score, history = train_state(
        culture,
        regime,
        int(seed),
        warmup_s=float(warmup_s),
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=bool(stop_at_criterion),
    )
    work = copy.deepcopy(net)
    interp = interpolation_rows(
        work,
        regime,
        naive,
        trained,
        task=task,
        seed=int(seed),
        alpha_values=alpha_values,
        eval_repetitions=eval_repetitions,
    )
    abl = ablation_rows(
        work,
        regime,
        naive,
        trained,
        task=task,
        seed=int(seed),
        eval_repetitions=eval_repetitions,
    )
    return {
        "task": task,
        "seed": int(seed),
        "interpolation_rows": interp,
        "ablation_rows": abl,
        "delta": (trained - naive).astype(np.float32, copy=False),
        "channel_delta": channel_delta_vector(net, trained - naive).astype(np.float32, copy=False),
        "trained_delta_norm": float(np.linalg.norm(trained - naive)),
        "training_repetitions": int(train_reps),
        "trained_score": float(trained_score),
        "training_history": "|".join(f"{score:.6g}" for score in history),
    }


def rows_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def row_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["task"]), int(row["seed"])) for row in rows if "task" in row and "seed" in row}


def completed_row_keys(interp_rows_existing: list[dict[str, Any]], abl_rows_existing: list[dict[str, Any]]):
    return row_keys(interp_rows_existing) & row_keys(abl_rows_existing)


def filter_completed_rows(rows: list[dict[str, Any]], completed: set[tuple[str, int]]) -> list[dict[str, Any]]:
    return [row for row in rows if (str(row["task"]), int(row["seed"])) in completed]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def breaking_points(interp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    axis_rows = interp[interp["direction"] == "axis"]
    for (task, seed), group in axis_rows.groupby(["task", "seed"]):
        criterion = float(group["criterion"].iloc[0])
        below = group[group["score"] < criterion].sort_values("alpha")
        alpha_star = float(below["alpha"].iloc[0]) if not below.empty else np.nan
        random = interp[(interp.task == task) & (interp.seed == seed) & (interp.direction == "random")]
        random_breaks = bool((random["score"] < criterion).any())
        rows.append(
            {
                "task": task,
                "seed": int(seed),
                "alpha_star_axis": alpha_star,
                "random_ever_breaks": random_breaks,
            }
        )
    return pd.DataFrame(rows)


def load_delta_matrix(out_dir: Path, task: str, seeds: list[int]) -> np.ndarray:
    vectors = []
    for seed in seeds:
        path = delta_path(out_dir, task, seed)
        if not path.exists():
            continue
        with np.load(path) as payload:
            if "channel_delta" not in payload.files:
                continue
            vectors.append(payload["channel_delta"].astype(np.float64, copy=False))
    if not vectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(vectors, axis=0)


def variance_fraction(values: np.ndarray, count: int) -> float:
    total = float(np.sum(values))
    if total <= 0.0:
        return np.nan
    return float(np.sum(values[: min(count, values.size)]) / total)


def dimensionality_rows(out_dir: Path, tasks: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        matrix = load_delta_matrix(out_dir, task, seeds)
        if matrix.size == 0:
            continue
        norms = np.linalg.norm(matrix, axis=1)
        mean_axis = matrix.mean(axis=0)
        mean_axis_norm = float(np.linalg.norm(mean_axis))
        if mean_axis_norm > 0.0:
            unit_mean_axis = mean_axis / mean_axis_norm
            axis_fractions = np.square(matrix @ unit_mean_axis) / np.maximum(np.square(norms), 1e-12)
            mean_axis_alignment_mean = float(np.nanmean(axis_fractions))
            mean_axis_alignment_sd = float(np.nanstd(axis_fractions, ddof=1)) if matrix.shape[0] > 1 else 0.0
        else:
            axis_fractions = np.full(matrix.shape[0], np.nan)
            mean_axis_alignment_mean = np.nan
            mean_axis_alignment_sd = np.nan

        centered = matrix - mean_axis
        centered_singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        centered_variance = np.square(centered_singular)
        uncentered_singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        uncentered_variance = np.square(uncentered_singular)
        centered_total = float(np.sum(centered_variance))
        participation_ratio = (
            float(centered_total**2 / np.sum(np.square(centered_variance)))
            if centered_total > 0.0 and np.sum(np.square(centered_variance)) > 0.0
            else np.nan
        )
        rows.append(
            {
                "task": task,
                "seeds": int(matrix.shape[0]),
                "edges": int(matrix.shape[1]),
                "delta_norm_mean": float(np.mean(norms)),
                "delta_norm_sd": float(np.std(norms, ddof=1)) if matrix.shape[0] > 1 else 0.0,
                "mean_axis_alignment_mean": mean_axis_alignment_mean,
                "mean_axis_alignment_sd": mean_axis_alignment_sd,
                "pca_top1_variance_explained": variance_fraction(centered_variance, 1),
                "pca_top2_variance_explained": variance_fraction(centered_variance, 2),
                "pca_top5_variance_explained": variance_fraction(centered_variance, 5),
                "pca_participation_ratio": participation_ratio,
                "svd_top1_variance_explained": variance_fraction(uncentered_variance, 1),
                "svd_top2_variance_explained": variance_fraction(uncentered_variance, 2),
                "svd_top5_variance_explained": variance_fraction(uncentered_variance, 5),
                "centered_singular_values": "|".join(f"{value:.6g}" for value in centered_singular[:10]),
                "uncentered_singular_values": "|".join(f"{value:.6g}" for value in uncentered_singular[:10]),
            }
        )
    return rows


def subplot_grid(count: int, *, width: float, height: float):
    cols = min(3, max(count, 1))
    rows = int(np.ceil(max(count, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[count:]:
        ax.set_axis_off()
    return fig, axes[:count]


def figure_breaking_point(interp: pd.DataFrame, bp: pd.DataFrame, out_dir: Path) -> None:
    tasks = list(interp["task"].drop_duplicates())
    fig, axes = subplot_grid(len(tasks), width=4.3, height=3.4)
    colors = {"axis": PALETTE["vermillion"], "random": PALETTE["gray"]}
    labels = {"axis": "toward naive (training-defined axis)", "random": "random matched direction"}
    for ax, task in zip(axes, tasks, strict=True):
        sub = interp[interp["task"] == task]
        criterion = float(sub["criterion"].iloc[0])
        for direction in ("axis", "random"):
            d = sub[sub["direction"] == direction].groupby("alpha")["score"].agg(["mean", "std"]).reset_index()
            ax.plot(d["alpha"], d["mean"], color=colors[direction], lw=1.6, marker="o", ms=2.5, label=labels[direction])
            ax.fill_between(
                d["alpha"],
                d["mean"] - d["std"].fillna(0.0),
                d["mean"] + d["std"].fillna(0.0),
                color=colors[direction],
                alpha=0.18,
            )
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        astar = bp.loc[bp["task"] == task, "alpha_star_axis"].mean()
        if np.isfinite(astar):
            ax.axvline(astar, color=PALETTE["vermillion"], ls=":", lw=1.0, alpha=0.8)
            ax.text(astar + 0.02, 0.02, f"a*={astar:.2f}", color=PALETTE["vermillion"], fontsize=8)
        ax.set_title(task_label(task))
        ax.set_xlabel("fraction moved along direction (a)")
        ax.set_ylabel("task score")
    if len(axes):
        axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Performance collapse is direction-specific, not displacement-magnitude specific", fontsize=11)
    save(fig, out_dir, "F1_memory_axis_breaking_point")


def figure_ablations(abl: pd.DataFrame, out_dir: Path) -> None:
    summary = abl.groupby(["task", "condition"], as_index=False)["score"].mean()
    tasks = list(summary["task"].drop_duplicates())
    order = [
        "trained",
        "anti_training_delta",
        "random_matched_full_trace",
        "random_distance_matched_task_path_union",
        "random_edge_count_matched_task_path_union",
        "shuffle_magnitudes",
        "zero_task_path_union",
        "zero_all_task_inputs_outgoing",
        "zero_all_task_targets_incoming",
        "zero_all",
        "naive_replacement",
    ]
    fig, axes = subplot_grid(len(tasks), width=5.1, height=4.1)
    for ax, task in zip(axes, tasks, strict=True):
        task_summary = summary[summary["task"] == task].set_index("condition")
        present_order = [condition for condition in order if condition in task_summary.index]
        sub = task_summary.reindex(present_order).reset_index()
        criterion = float(abl.loc[abl["task"] == task, "criterion"].iloc[0])
        colors = [PALETTE["vermillion"] if value < criterion else PALETTE["blue"] for value in sub["score"]]
        ax.barh(sub["condition"], sub["score"], color=colors)
        ax.axvline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task_label(task))
        ax.set_xlabel("task score")
        ax.invert_yaxis()
    fig.suptitle("Task-path ablations versus matched random and global controls", fontsize=11)
    save(fig, out_dir, "F2_surgical_ablations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-geometry probes for SNN reset tasks.")
    add_common_task_args(parser)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_BUILDERS), default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alpha-count", type=int, default=len(DEFAULT_ALPHAS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    started_at = datetime.now(timezone.utc)
    out_dir = args.out or RESULTS_DIR / f"manifold_analysis_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_vectors").mkdir(exist_ok=True)
    apply_style()

    alpha_values = np.round(np.linspace(0.0, 1.0, max(int(args.alpha_count), 2)), 4)
    tasks = list(args.tasks)
    seeds = [int(seed) for seed in args.seeds]
    interp_path = out_dir / "interpolation.csv"
    abl_path = out_dir / "ablations.csv"

    interp_rows_existing = rows_from_csv(interp_path) if args.resume else []
    abl_rows_existing = rows_from_csv(abl_path) if args.resume else []
    completed_from_rows = completed_row_keys(interp_rows_existing, abl_rows_existing)
    completed = {
        (task, seed)
        for task in tasks
        for seed in seeds
        if (task, seed) in completed_from_rows and delta_path(out_dir, task, seed).exists()
    }
    interp_rows_existing = filter_completed_rows(interp_rows_existing, completed)
    abl_rows_existing = filter_completed_rows(abl_rows_existing, completed)
    if args.resume:
        write_rows(interp_path, interp_rows_existing)
        write_rows(abl_path, abl_rows_existing)

    regimes = {task: TASK_BUILDERS[task](args) for task in tasks}
    culture = culture_from_args(args)
    pending_jobs = [
        (
            culture,
            regimes[task],
            task,
            seed,
            alpha_values,
            float(args.warmup_s),
            args.training_repetitions,
            args.eval_repetitions,
            bool(args.stop_at_criterion),
        )
        for task in tasks
        for seed in seeds
        if (task, seed) not in completed
    ]

    metadata = {
        "run_id": out_dir.name,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "tasks": tasks,
        "seeds": seeds,
        "alphas": alpha_values.tolist(),
        "started_at_utc": started_at.isoformat(),
        "completed_jobs": len(completed),
        "pending_jobs": len(pending_jobs),
        "status": "running",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    total_jobs = len(tasks) * len(seeds)
    completed_count = len(completed)
    if pending_jobs:
        executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=int(args.workers)) as executor:
            futures = {executor.submit(geometry_job, job): (job[2], int(job[3])) for job in pending_jobs}
            for future in as_completed(futures):
                task, seed = futures[future]
                result = future.result()
                np.savez_compressed(
                    delta_path(out_dir, task, seed),
                    delta=result["delta"],
                    channel_delta=result["channel_delta"],
                    trained_delta_norm=np.array(result["trained_delta_norm"], dtype=np.float64),
                )
                append_rows(interp_path, result["interpolation_rows"])
                append_rows(abl_path, result["ablation_rows"])
                completed_count += 1
                elapsed = perf_counter() - started
                jobs_per_s = (completed_count - len(completed)) / max(elapsed, 1e-9)
                remaining = max(total_jobs - completed_count, 0)
                metadata.update(
                    {
                        "completed_jobs": int(completed_count),
                        "pending_jobs": int(remaining),
                        "elapsed_s": float(elapsed),
                        "jobs_per_s": float(jobs_per_s),
                        "eta_s": float(remaining / max(jobs_per_s, 1e-9)),
                        "last_completed_task": task,
                        "last_completed_seed": int(seed),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                (out_dir / "progress.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                if completed_count % max(int(args.progress_interval), 1) == 0 or completed_count == total_jobs:
                    print(
                        json.dumps(
                            {
                                "event": "geometry_progress",
                                "completed": completed_count,
                                "total": total_jobs,
                                "last_task": task,
                                "last_seed": int(seed),
                                "eta_s": metadata["eta_s"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    interp = pd.read_csv(interp_path)
    abl = pd.read_csv(abl_path)
    bp = breaking_points(interp)
    dim = pd.DataFrame(dimensionality_rows(out_dir, tasks, seeds))
    bp.to_csv(out_dir / "breaking_points.csv", index=False)
    dim.to_csv(out_dir / "dimensionality.csv", index=False)
    figure_breaking_point(interp, bp, out_dir)
    figure_ablations(abl, out_dir)

    metadata.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": perf_counter() - started,
            "completed_jobs": int(total_jobs),
            "pending_jobs": 0,
            "outputs": [
                "metadata.json",
                "progress.json",
                "interpolation.csv",
                "ablations.csv",
                "breaking_points.csv",
                "dimensionality.csv",
                "delta_vectors/",
                "F1_memory_axis_breaking_point.png",
                "F1_memory_axis_breaking_point.pdf",
                "F2_surgical_ablations.png",
                "F2_surgical_ablations.pdf",
            ],
            "status": "complete",
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "progress.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "complete", "out_dir": str(out_dir), "elapsed_s": metadata["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
