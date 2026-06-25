"""Association-task anisotropy and pathway erasure probe.

Train conditioned electrode association to criterion, define the memory axis as
trained weights minus naive weights, then compare matched-distance moves toward
naive against random orthogonal moves. The same run silences the candidate
input-to-target pathway and compares it with magnitude-matched random synaptic
ablations. Optional decoder rows ask whether association content is readable
from pathway/direction-isolated activity or from the remaining network.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SNN_RESET_DIR = ROOT / "experiments" / "snn_reset"
for path in (SRC_DIR, SNN_RESET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cl1_snn_reset import (  # noqa: E402
    TaskRegime,
    build_network,
    conditioned_electrode_association,
    evaluate_regime,
    train_regime,
)
from common import add_common_task_args, culture_from_args, git_commit  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DISTANCE_FRACTIONS = (
    0.0,
    0.02,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
)


def task_accuracy(evaluation) -> float:
    return float(
        0.5
        * (
            evaluation.positive_response_probability
            + (1.0 - evaluation.negative_response_probability)
        )
    )


def criterion_accuracy(regime: TaskRegime) -> float:
    return float(0.5 * (1.0 + regime.criterion_score))


def association_regime(args: argparse.Namespace) -> TaskRegime:
    return conditioned_electrode_association(
        input_channel=int(args.input_channel),
        target_channel=int(args.target_channel),
        input_current_uA=float(args.input_current_uA),
        target_current_uA=float(args.target_current_uA),
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.4,
        max_training_repetitions=args.training_repetitions or 80,
        eval_repetitions=args.eval_repetitions or 16,
    )


def event_channels(events) -> tuple[int, ...]:
    return tuple(sorted({int(channel) for event in events for channel in event.channels}))


def positive_pathway(regime: TaskRegime) -> dict[str, Any]:
    positive = [probe for probe in regime.probes if probe.is_positive]
    inputs = tuple(sorted({channel for probe in positive for channel in event_channels(probe.events)}))
    targets = tuple(sorted({int(channel) for probe in positive for channel in probe.target_channels}))
    if not inputs or not targets:
        raise ValueError("Association pathway could not be inferred from positive probes.")
    return {
        "probe_names": tuple(probe.name for probe in positive),
        "input_channels": inputs,
        "target_channels": targets,
        "pathway_channels": tuple(sorted(set(inputs) | set(targets))),
    }


def channel_edge_mask(net, source_channels: tuple[int, ...], target_channels: tuple[int, ...]) -> np.ndarray:
    channel_of = net.electrodes.nearest_channel
    source_neuron = np.isin(channel_of, list(source_channels))
    target_neuron = np.isin(channel_of, list(target_channels))
    return source_neuron[net.sources] & target_neuron[net.targets]


def legalize_weights(net, weights: np.ndarray) -> np.ndarray:
    signs = np.asarray(net.signs, dtype=np.float64)
    magnitudes = np.clip(np.abs(np.asarray(weights, dtype=np.float64)), 0.0, float(net.cfg.w_max))
    return magnitudes * signs


def score_weights(
    base_net,
    regime: TaskRegime,
    weights: np.ndarray,
    *,
    eval_repetitions: int | None,
):
    work = copy.deepcopy(base_net)
    work.set_weights(legalize_weights(base_net, weights))
    effective = work.weights_vector()
    evaluation = evaluate_regime(work, regime, repetitions=eval_repetitions)
    return evaluation, effective


def random_unit_orthogonal(axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    axis_norm_sq = float(axis @ axis)
    if axis_norm_sq <= 0.0:
        raise ValueError("Cannot sample an orthogonal direction for a zero memory axis.")
    for _ in range(100):
        direction = rng.standard_normal(axis.shape)
        direction -= float(direction @ axis) / axis_norm_sq * axis
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            return direction / norm
    raise RuntimeError("Failed to sample a nonzero orthogonal direction.")


def evaluation_row(
    *,
    seed: int,
    block: str,
    condition: str,
    replicate: int,
    evaluation,
    effective: np.ndarray,
    trained: np.ndarray,
    axis_norm: float,
    regime: TaskRegime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distance = float(np.linalg.norm(effective - trained))
    row: dict[str, Any] = {
        "seed": int(seed),
        "block": block,
        "condition": condition,
        "replicate": int(replicate),
        "score": float(evaluation.score),
        "task_accuracy": task_accuracy(evaluation),
        "criterion_score": float(regime.criterion_score),
        "criterion_accuracy": criterion_accuracy(regime),
        "positive_response_probability": float(evaluation.positive_response_probability),
        "negative_response_probability": float(evaluation.negative_response_probability),
        "effective_distance_from_trained": distance,
        "effective_distance_fraction_of_axis": distance / max(axis_norm, 1e-12),
        "collapsed_by_score": bool(evaluation.score < regime.criterion_score),
    }
    if extra:
        row.update(extra)
    return row


def displacement_sweep_rows(
    base_net,
    regime: TaskRegime,
    naive: np.ndarray,
    trained: np.ndarray,
    *,
    seed: int,
    distance_fractions: tuple[float, ...],
    orthogonal_repeats: int,
    eval_repetitions: int | None,
) -> list[dict[str, Any]]:
    axis = trained - naive
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Training produced a zero memory axis.")

    rng = np.random.default_rng(1_000_003 * int(seed) + 17)
    rows: list[dict[str, Any]] = []
    for fraction in distance_fractions:
        distance = float(fraction) * axis_norm
        parallel_target = trained - float(fraction) * axis
        evaluation, effective = score_weights(
            base_net,
            regime,
            parallel_target,
            eval_repetitions=eval_repetitions,
        )
        rows.append(
            evaluation_row(
                seed=seed,
                block="displacement",
                condition="parallel_erasure_axis",
                replicate=0,
                evaluation=evaluation,
                effective=effective,
                trained=trained,
                axis_norm=axis_norm,
                regime=regime,
                extra={
                    "distance_fraction": float(fraction),
                    "intended_distance": distance,
                    "orthogonal_dot_axis": np.nan,
                },
            )
        )

        for replicate in range(int(orthogonal_repeats)):
            direction = random_unit_orthogonal(axis, rng)
            target = trained + distance * direction
            evaluation, effective = score_weights(
                base_net,
                regime,
                target,
                eval_repetitions=eval_repetitions,
            )
            rows.append(
                evaluation_row(
                    seed=seed,
                    block="displacement",
                    condition="orthogonal_random",
                    replicate=replicate,
                    evaluation=evaluation,
                    effective=effective,
                    trained=trained,
                    axis_norm=axis_norm,
                    regime=regime,
                    extra={
                        "distance_fraction": float(fraction),
                        "intended_distance": distance,
                        "orthogonal_dot_axis": float(direction @ axis),
                    },
                )
            )
    return rows


def random_edge_count_mask(
    available: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros_like(available, dtype=bool)
    candidates = np.flatnonzero(available)
    if count <= 0 or candidates.size == 0:
        return mask
    chosen = rng.choice(candidates, size=min(int(count), candidates.size), replace=False)
    mask[chosen] = True
    return mask


def random_magnitude_matched_mask(
    weights: np.ndarray,
    available: np.ndarray,
    target_distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros_like(available, dtype=bool)
    candidates = np.flatnonzero(available)
    if target_distance <= 0.0 or candidates.size == 0:
        return mask

    order = rng.permutation(candidates)
    squared = np.square(weights[order].astype(np.float64, copy=False))
    cumulative = np.cumsum(squared)
    target_sq = float(target_distance) ** 2
    stop = int(np.searchsorted(cumulative, target_sq, side="left"))
    if stop >= order.size:
        chosen = order
    elif stop == 0:
        chosen = order[:1]
    else:
        previous_error = abs(float(cumulative[stop - 1]) - target_sq)
        current_error = abs(float(cumulative[stop]) - target_sq)
        chosen = order[: stop if previous_error <= current_error else stop + 1]
    mask[chosen] = True
    return mask


def zeroed(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = weights.copy()
    result[mask] = 0.0
    return result


def pathway_ablation_rows(
    base_net,
    regime: TaskRegime,
    naive: np.ndarray,
    trained: np.ndarray,
    *,
    seed: int,
    path_mask: np.ndarray,
    control_repeats: int,
    eval_repetitions: int | None,
) -> list[dict[str, Any]]:
    axis_norm = float(np.linalg.norm(trained - naive))
    path_edge_count = int(np.count_nonzero(path_mask))
    path_distance = float(np.linalg.norm(trained - zeroed(trained, path_mask)))
    available = ~path_mask
    rng = np.random.default_rng(2_000_003 * int(seed) + 29)
    rows: list[dict[str, Any]] = []

    empty_mask = np.zeros_like(path_mask, dtype=bool)
    full_reset_touched = np.abs(trained - naive) > 1e-12
    fixed_conditions = (
        ("trained", 0, trained, empty_mask, empty_mask, "reference"),
        ("full_naive_reset", 0, naive, full_reset_touched, empty_mask, "full_reset"),
        (
            "silence_candidate_input_to_target_pathway",
            0,
            zeroed(trained, path_mask),
            path_mask,
            path_mask,
            "candidate_pathway",
        ),
    )
    for condition, replicate, weights, touched_mask, silenced_mask, family in fixed_conditions:
        evaluation, effective = score_weights(
            base_net,
            regime,
            weights,
            eval_repetitions=eval_repetitions,
        )
        rows.append(
            evaluation_row(
                seed=seed,
                block="ablation",
                condition=condition,
                replicate=replicate,
                evaluation=evaluation,
                effective=effective,
                trained=trained,
                axis_norm=axis_norm,
                regime=regime,
                extra={
                    "control_family": family,
                    "edges_touched": int(np.count_nonzero(touched_mask)),
                    "edges_touched_fraction": float(np.mean(touched_mask)),
                    "edges_silenced": int(np.count_nonzero(silenced_mask)),
                    "edges_silenced_fraction": float(np.mean(silenced_mask)),
                    "path_edges": path_edge_count,
                    "path_edge_fraction": float(np.mean(path_mask)),
                    "path_ablation_distance": path_distance,
                    "ablation_match_error": np.nan,
                },
            )
        )

    for replicate in range(int(control_repeats)):
        for condition, mask in (
            (
                "random_edge_count_matched_ablation",
                random_edge_count_mask(available, path_edge_count, rng),
            ),
            (
                "random_magnitude_matched_ablation",
                random_magnitude_matched_mask(trained, available, path_distance, rng),
            ),
        ):
            weights = zeroed(trained, mask)
            evaluation, effective = score_weights(
                base_net,
                regime,
                weights,
                eval_repetitions=eval_repetitions,
            )
            distance = float(np.linalg.norm(trained - weights))
            rows.append(
                evaluation_row(
                    seed=seed,
                    block="ablation",
                    condition=condition,
                    replicate=replicate,
                    evaluation=evaluation,
                    effective=effective,
                    trained=trained,
                    axis_norm=axis_norm,
                    regime=regime,
                    extra={
                        "control_family": "random_control",
                        "edges_touched": int(np.count_nonzero(mask)),
                        "edges_touched_fraction": float(np.mean(mask)),
                        "edges_silenced": int(np.count_nonzero(mask)),
                        "edges_silenced_fraction": float(np.mean(mask)),
                        "path_edges": path_edge_count,
                        "path_edge_fraction": float(np.mean(path_mask)),
                        "path_ablation_distance": path_distance,
                        "ablation_match_error": abs(distance - path_distance),
                    },
                )
            )
    return rows


def isolated_weight_conditions(
    naive: np.ndarray,
    trained: np.ndarray,
    path_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    pathway_only = naive.copy()
    pathway_only[path_mask] = trained[path_mask]
    rest_only = trained.copy()
    rest_only[path_mask] = naive[path_mask]
    return {
        "naive_full": naive,
        "trained_full": trained,
        "pathway_direction_only": pathway_only,
        "rest_of_network_direction_only": rest_only,
    }


def collect_probe_activity(
    base_net,
    regime: TaskRegime,
    weights: np.ndarray,
    *,
    repetitions: int,
    bin_ms: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    work = copy.deepcopy(base_net)
    work.set_weights(legalize_weights(base_net, weights))
    channel_count = int(work.electrodes.channel_count)
    features: list[np.ndarray] = []
    labels: list[int] = []
    probe_names: list[str] = []
    probes = tuple(regime.probes)

    for _ in range(max(int(repetitions), 1)):
        for probe in probes:
            activity = work.advance(probe.duration_ms, probe.events, plasticity=False, record=True)
            features.append(activity.binned_counts(bin_ms=bin_ms, channel_count=channel_count))
            labels.append(1 if probe.is_positive else 0)
            probe_names.append(probe.name)

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64), probe_names


def decode_binary(X: np.ndarray, y: np.ndarray, *, seed: int) -> dict[str, float | int]:
    class_counts = np.bincount(y, minlength=2)
    min_class = int(class_counts.min())
    if X.shape[1] == 0 or min_class < 2 or np.unique(y).size < 2:
        return {
            "decoder_accuracy": np.nan,
            "decoder_balanced_accuracy": np.nan,
            "decoder_splits": 0,
            "decoder_samples": int(y.size),
            "decoder_features": int(X.shape[1]),
        }

    splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=int(seed),
        ),
    )
    accuracies: list[float] = []
    balanced: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        predicted = model.predict(X[test_idx])
        accuracies.append(float(accuracy_score(y[test_idx], predicted)))
        balanced.append(float(balanced_accuracy_score(y[test_idx], predicted)))
    return {
        "decoder_accuracy": float(np.mean(accuracies)),
        "decoder_balanced_accuracy": float(np.mean(balanced)),
        "decoder_splits": int(splits),
        "decoder_samples": int(y.size),
        "decoder_features": int(X.shape[1]),
    }


def decoder_rows(
    base_net,
    regime: TaskRegime,
    naive: np.ndarray,
    trained: np.ndarray,
    *,
    seed: int,
    path_mask: np.ndarray,
    pathway_channels: tuple[int, ...],
    repetitions: int,
    bin_ms: float,
) -> list[dict[str, Any]]:
    channel_count = int(base_net.electrodes.channel_count)
    pathway_channel_mask = np.zeros(channel_count, dtype=bool)
    pathway_channel_mask[list(pathway_channels)] = True
    channel_views = {
        "all_channels": np.arange(channel_count),
        "pathway_channels": np.flatnonzero(pathway_channel_mask),
        "rest_channels": np.flatnonzero(~pathway_channel_mask),
    }

    rows: list[dict[str, Any]] = []
    for condition, weights in isolated_weight_conditions(naive, trained, path_mask).items():
        activity, labels, probe_names = collect_probe_activity(
            base_net,
            regime,
            weights,
            repetitions=repetitions,
            bin_ms=bin_ms,
        )
        for view, channels in channel_views.items():
            X = activity[:, :, channels].reshape(activity.shape[0], -1)
            metrics = decode_binary(X, labels, seed=3_000_007 * int(seed) + len(condition) + len(view))
            rows.append(
                {
                    "seed": int(seed),
                    "weight_condition": condition,
                    "feature_view": view,
                    "positive_probe_count": int(np.sum(labels == 1)),
                    "negative_probe_count": int(np.sum(labels == 0)),
                    "probe_names": "|".join(sorted(set(probe_names))),
                    "bin_ms": float(bin_ms),
                    "path_edges": int(np.count_nonzero(path_mask)),
                    **metrics,
                }
            )
    return rows


def threshold_rows(displacement: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parallel = displacement[displacement["condition"] == "parallel_erasure_axis"]
    for seed, group in parallel.groupby("seed", sort=True):
        ordered = group.sort_values("distance_fraction")
        criterion = float(ordered["criterion_score"].iloc[0])
        below = ordered[ordered["score"] < criterion]
        if below.empty:
            threshold = np.nan
            threshold_accuracy = np.nan
        else:
            first = below.iloc[0]
            idx = int(ordered.index.get_loc(first.name))
            if idx == 0:
                threshold = float(first["distance_fraction"])
                threshold_accuracy = float(first["task_accuracy"])
            else:
                prev = ordered.iloc[idx - 1]
                x0 = float(prev["distance_fraction"])
                x1 = float(first["distance_fraction"])
                y0 = float(prev["score"])
                y1 = float(first["score"])
                if abs(y1 - y0) <= 1e-12:
                    threshold = x1
                else:
                    threshold = x0 + (criterion - y0) * (x1 - x0) / (y1 - y0)
                threshold_accuracy = float(first["task_accuracy"])

        orthogonal = displacement[displacement["seed"].eq(seed) & displacement["condition"].eq("orthogonal_random")]
        criterion_acc = float(ordered["criterion_accuracy"].iloc[0])
        intact = orthogonal[orthogonal["task_accuracy"] >= criterion_acc]
        max_orthogonal_intact = (
            float(intact["distance_fraction"].max()) if not intact.empty else np.nan
        )
        rows.append(
            {
                "seed": int(seed),
                "parallel_threshold_fraction": threshold,
                "parallel_threshold_percent": threshold * 100.0 if np.isfinite(threshold) else np.nan,
                "parallel_threshold_first_below_accuracy": threshold_accuracy,
                "max_orthogonal_intact_fraction": max_orthogonal_intact,
                "orthogonal_to_parallel_intact_ratio": (
                    max_orthogonal_intact / threshold
                    if np.isfinite(max_orthogonal_intact) and np.isfinite(threshold) and threshold > 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def seed_summary_row(
    *,
    seed: int,
    train_repetitions: int,
    trained_eval,
    naive_eval,
    history: tuple[float, ...],
    naive: np.ndarray,
    trained: np.ndarray,
    path_mask: np.ndarray,
    regime: TaskRegime,
) -> dict[str, Any]:
    delta = trained - naive
    axis_norm = float(np.linalg.norm(delta))
    path_delta_norm = float(np.linalg.norm(delta[path_mask]))
    return {
        "seed": int(seed),
        "training_repetitions": int(train_repetitions),
        "trained_score": float(trained_eval.score),
        "trained_task_accuracy": task_accuracy(trained_eval),
        "naive_score_after_weight_reset": float(naive_eval.score),
        "naive_task_accuracy_after_weight_reset": task_accuracy(naive_eval),
        "criterion_score": float(regime.criterion_score),
        "criterion_accuracy": criterion_accuracy(regime),
        "reached_criterion": bool(trained_eval.score >= regime.criterion_score),
        "training_history": "|".join(f"{score:.6g}" for score in history),
        "axis_norm": axis_norm,
        "path_delta_norm": path_delta_norm,
        "path_delta_norm_fraction": path_delta_norm / max(axis_norm, 1e-12),
        "path_edges": int(np.count_nonzero(path_mask)),
        "synapse_count": int(path_mask.size),
        "path_edge_fraction": float(np.mean(path_mask)),
    }


def write_frame(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def plot_displacement(displacement: pd.DataFrame, thresholds: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    sns.lineplot(
        data=displacement,
        x="distance_fraction",
        y="task_accuracy",
        hue="condition",
        estimator="mean",
        errorbar="sd",
        marker="o",
        ax=ax,
    )
    criterion = float(displacement["criterion_accuracy"].iloc[0])
    ax.axhline(criterion, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    threshold = float(thresholds["parallel_threshold_fraction"].mean())
    if np.isfinite(threshold):
        ax.axvline(threshold, color="tab:red", linestyle=":", linewidth=1.2)
        ax.text(threshold, criterion + 0.02, f"{threshold * 100.0:.1f}%", color="tab:red")
    ax.set_xlabel("Displacement distance / trained-naive axis norm")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Matched-magnitude displacement sweep")
    ax.legend(frameon=False, title=None)
    fig.savefig(out_dir / "displacement_anisotropy.png", dpi=200)
    fig.savefig(out_dir / "displacement_anisotropy.pdf")
    plt.close(fig)


def plot_ablations(ablation: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "trained",
        "full_naive_reset",
        "silence_candidate_input_to_target_pathway",
        "random_edge_count_matched_ablation",
        "random_magnitude_matched_ablation",
    ]
    present = [condition for condition in order if condition in set(ablation["condition"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    sns.barplot(
        data=ablation,
        y="condition",
        x="task_accuracy",
        order=present,
        errorbar="sd",
        ax=ax,
    )
    criterion = float(ablation["criterion_accuracy"].iloc[0])
    ax.axvline(criterion, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Task accuracy")
    ax.set_ylabel("")
    ax.set_title("Candidate pathway silencing versus random ablations")
    fig.savefig(out_dir / "pathway_ablation_controls.png", dpi=200)
    fig.savefig(out_dir / "pathway_ablation_controls.pdf")
    plt.close(fig)


def plot_decoder(decoder: pd.DataFrame, out_dir: Path) -> None:
    if decoder.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    sns.barplot(
        data=decoder,
        x="weight_condition",
        y="decoder_balanced_accuracy",
        hue="feature_view",
        errorbar="sd",
        ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("")
    ax.set_ylabel("Balanced decoding accuracy")
    ax.set_title("Access-versus-content decoder")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, title=None)
    fig.savefig(out_dir / "access_content_decoder.png", dpi=200)
    fig.savefig(out_dir / "access_content_decoder.pdf")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run association-task anisotropy and pathway-specific erasure probes."
    )
    add_common_task_args(parser)
    parser.add_argument(
        "--distance-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCE_FRACTIONS),
        help="Distances as fractions of the trained-naive axis norm.",
    )
    parser.add_argument("--orthogonal-repeats", type=int, default=8)
    parser.add_argument("--ablation-control-repeats", type=int, default=32)
    parser.add_argument("--decode-repetitions", type=int, default=48)
    parser.add_argument("--decoder-bin-ms", type=float, default=10.0)
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--train-full-budget",
        action="store_true",
        help="Run the full training budget instead of stopping once criterion is reached.",
    )
    parser.add_argument(
        "--allow-unreached-criterion",
        action="store_true",
        help="Write diagnostic rows even if a seed does not reach criterion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or f"anisotropic_pathway_erasure_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = args.output_dir or RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    regime = association_regime(args)
    pathway = positive_pathway(regime)
    culture = culture_from_args(args)
    seeds = [int(seed) for seed in args.seeds]
    distance_fractions = tuple(sorted({float(value) for value in args.distance_fractions}))
    stop_at_criterion = not bool(args.train_full_budget)
    eval_repetitions = args.eval_repetitions or regime.eval_repetitions

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": {key: json_ready(value) for key, value in vars(args).items()},
        "task_regime": regime.to_metadata(),
        "candidate_pathway": pathway,
        "distance_fractions": distance_fractions,
        "stop_at_criterion": stop_at_criterion,
    }
    (out_dir / "metadata.json").write_text(json.dumps(json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")

    seed_rows: list[dict[str, Any]] = []
    displacement_rows_all: list[dict[str, Any]] = []
    ablation_rows_all: list[dict[str, Any]] = []
    decoder_rows_all: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        net = build_network(culture, seed=seed)
        if args.warmup_s:
            net.advance(float(args.warmup_s) * 1000.0, [], plasticity=False, record=False)
        naive = net.weights_vector()
        train_repetitions, trained_eval, history = train_regime(
            net,
            regime,
            max_repetitions=args.training_repetitions,
            eval_repetitions=eval_repetitions,
            stop_at_criterion=stop_at_criterion,
        )
        trained = net.weights_vector()
        path_mask = channel_edge_mask(net, pathway["input_channels"], pathway["target_channels"])
        naive_eval, _ = score_weights(net, regime, naive, eval_repetitions=eval_repetitions)

        if trained_eval.score < regime.criterion_score and not args.allow_unreached_criterion:
            raise RuntimeError(
                f"Seed {seed} did not reach criterion: score={trained_eval.score:.3f}, "
                f"criterion={regime.criterion_score:.3f}. Re-run with --allow-unreached-criterion "
                "to keep diagnostic rows."
            )

        seed_rows.append(
            seed_summary_row(
                seed=seed,
                train_repetitions=int(train_repetitions),
                trained_eval=trained_eval,
                naive_eval=naive_eval,
                history=history,
                naive=naive,
                trained=trained,
                path_mask=path_mask,
                regime=regime,
            )
        )
        displacement_rows_all.extend(
            displacement_sweep_rows(
                net,
                regime,
                naive,
                trained,
                seed=seed,
                distance_fractions=distance_fractions,
                orthogonal_repeats=int(args.orthogonal_repeats),
                eval_repetitions=eval_repetitions,
            )
        )
        ablation_rows_all.extend(
            pathway_ablation_rows(
                net,
                regime,
                naive,
                trained,
                seed=seed,
                path_mask=path_mask,
                control_repeats=int(args.ablation_control_repeats),
                eval_repetitions=eval_repetitions,
            )
        )
        if not args.skip_decoder:
            decoder_rows_all.extend(
                decoder_rows(
                    net,
                    regime,
                    naive,
                    trained,
                    seed=seed,
                    path_mask=path_mask,
                    pathway_channels=pathway["pathway_channels"],
                    repetitions=int(args.decode_repetitions),
                    bin_ms=float(args.decoder_bin_ms),
                )
            )

        progress = {
            **metadata,
            "completed_seeds": index,
            "pending_seeds": len(seeds) - index,
            "elapsed_s": perf_counter() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "progress.json").write_text(
            json.dumps(json_ready(progress), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "seed_complete",
                    "seed": int(seed),
                    "completed": index,
                    "total": len(seeds),
                    "trained_score": float(trained_eval.score),
                    "path_edges": int(np.count_nonzero(path_mask)),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    seed_summary = write_frame(out_dir / "seed_summary.csv", seed_rows)
    displacement = write_frame(out_dir / "displacement_sweep.csv", displacement_rows_all)
    ablation = write_frame(out_dir / "pathway_ablations.csv", ablation_rows_all)
    thresholds = threshold_rows(displacement)
    write_frame(out_dir / "parallel_thresholds.csv", thresholds)
    decoder = write_frame(out_dir / "access_content_decoder.csv", decoder_rows_all)

    ablation_summary = (
        ablation.groupby(["condition"], as_index=False)
        .agg(
            task_accuracy_mean=("task_accuracy", "mean"),
            task_accuracy_sd=("task_accuracy", "std"),
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            effective_distance_fraction_mean=("effective_distance_fraction_of_axis", "mean"),
            edges_touched_mean=("edges_touched", "mean"),
            edges_silenced_mean=("edges_silenced", "mean"),
        )
    )
    write_frame(out_dir / "pathway_ablation_summary.csv", ablation_summary)

    if not args.no_figures:
        plot_displacement(displacement, thresholds, out_dir)
        plot_ablations(ablation, out_dir)
        plot_decoder(decoder, out_dir)

    outputs = [
        "metadata.json",
        "progress.json",
        "seed_summary.csv",
        "displacement_sweep.csv",
        "parallel_thresholds.csv",
        "pathway_ablations.csv",
        "pathway_ablation_summary.csv",
        "access_content_decoder.csv",
    ]
    if not args.no_figures:
        outputs.extend(
            [
                "displacement_anisotropy.png",
                "displacement_anisotropy.pdf",
                "pathway_ablation_controls.png",
                "pathway_ablation_controls.pdf",
                "access_content_decoder.png",
                "access_content_decoder.pdf",
            ]
        )

    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": perf_counter() - started,
            "outputs": outputs,
            "threshold_percent_mean": float(thresholds["parallel_threshold_percent"].mean()),
            "path_edge_fraction_mean": float(seed_summary["path_edge_fraction"].mean()),
            "path_delta_norm_fraction_mean": float(seed_summary["path_delta_norm_fraction"].mean()),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "progress.json").write_text(json.dumps(json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "complete", "out_dir": str(out_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
