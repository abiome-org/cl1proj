from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..config import CultureConfig
from ..experiment import apply_reset_protocol
from ..metrics import residual_trace_correlation, savings_score, weight_erasure_score
from ..network import build_network
from ..protocols import ResetProtocol
from .execution import evaluate_regime, train_regime
from .specs import RegimeEvaluation, TaskRegime


def _protocol_row(protocol: ResetProtocol) -> dict[str, Any]:
    return {
        "protocol_id": protocol.id,
        "beta": float(protocol.beta),
        "schedule": protocol.schedule,
        "spatial_mode": protocol.spatial_mode,
        "duration_s": float(protocol.duration_s),
        "current_uA": float(protocol.current_uA),
        "pulse_width_us": int(protocol.pulse_width_us),
    }


def first_criterion_repetition(history: tuple[float, ...], criterion_score: float) -> int | None:
    for repetition, score in enumerate(history):
        if float(score) >= float(criterion_score):
            return int(repetition)
    return None


def _history_row(history: tuple[float, ...]) -> str:
    return "|".join(f"{value:.6g}" for value in history)


def forgetting_flags(*, reset_score: float, no_reset_score: float, criterion_score: float) -> dict[str, bool]:
    score_drop = float(reset_score) < float(no_reset_score)
    criterion_forget = score_drop and float(no_reset_score) >= float(criterion_score) > float(reset_score)
    return {
        "score_drop": bool(score_drop),
        "criterion_forget": bool(criterion_forget),
        "made_forget": bool(criterion_forget),
    }


def _weight_metrics(
    baseline_weights: np.ndarray,
    trained_weights: np.ndarray,
    reset_weights: np.ndarray,
    no_reset_weights: np.ndarray,
) -> dict[str, float]:
    reset_minus_no_reset = reset_weights - no_reset_weights
    trained_delta = trained_weights - baseline_weights
    trained_delta_norm_sq = float(np.dot(trained_delta, trained_delta))
    erasure_projection = 0.0
    if trained_delta_norm_sq > 1e-12:
        erasure_projection = float(-np.dot(reset_minus_no_reset, trained_delta) / trained_delta_norm_sq)
    return {
        "trained_delta_norm": float(np.linalg.norm(trained_delta)),
        "naive_weight_control_displacement_norm": float(np.linalg.norm(baseline_weights - trained_weights)),
        "reset_post_minus_trained_norm": float(np.linalg.norm(reset_weights - trained_weights)),
        "no_reset_post_minus_trained_norm": float(np.linalg.norm(no_reset_weights - trained_weights)),
        "reset_minus_no_reset_weight_norm": float(np.linalg.norm(reset_minus_no_reset)),
        "weight_erasure_reset": weight_erasure_score(baseline_weights, trained_weights, reset_weights),
        "weight_erasure_no_reset": weight_erasure_score(baseline_weights, trained_weights, no_reset_weights),
        "residual_trace_correlation_reset": residual_trace_correlation(
            baseline_weights,
            trained_weights,
            reset_weights,
        ),
        "residual_trace_correlation_no_reset": residual_trace_correlation(
            baseline_weights,
            trained_weights,
            no_reset_weights,
        ),
        "erasure_projection_reset_vs_no_reset": erasure_projection,
    }


def _evaluation_controls_row(
    *,
    baseline_eval: RegimeEvaluation,
    trained_eval: RegimeEvaluation,
    naive_weight_control_eval: RegimeEvaluation,
    no_reset_eval: RegimeEvaluation,
) -> dict[str, float]:
    return {
        "baseline_score": float(baseline_eval.score),
        "trained_score": float(trained_eval.score),
        "training_score_delta": float(trained_eval.score - baseline_eval.score),
        "naive_weight_control_score": float(naive_weight_control_eval.score),
        "naive_weight_control_minus_trained_score": float(
            naive_weight_control_eval.score - trained_eval.score
        ),
        "naive_weight_control_minus_no_reset_score": float(
            naive_weight_control_eval.score - no_reset_eval.score
        ),
        "naive_weight_control_positive_response_probability": float(
            naive_weight_control_eval.positive_response_probability
        ),
        "naive_weight_control_negative_response_probability": float(
            naive_weight_control_eval.negative_response_probability
        ),
    }


def prepare_regime_seed_state(
    culture: CultureConfig,
    regime: TaskRegime,
    *,
    seed: int,
    warmup_s: float = 0.5,
    consolidation_rest_s: float = 1.0,
    training_repetitions: int | None = None,
    eval_repetitions: int | None = None,
    stop_at_criterion: bool = False,
) -> dict[str, Any]:
    """Train one task/seed once, then cache the settled state for protocol clones."""
    regime.validate()
    net = build_network(replace(culture, build_workers=1), seed=int(seed))
    if warmup_s > 0.0:
        net.advance(float(warmup_s) * 1000.0, [], plasticity=False, record=False)

    baseline_weights = net.weights_vector()
    baseline_eval = evaluate_regime(net, regime, repetitions=eval_repetitions)
    training_started = perf_counter()
    trained_repetitions, _, training_history = train_regime(
        net,
        regime,
        max_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=stop_at_criterion,
    )
    consolidation_activity = net.advance(
        max(float(consolidation_rest_s), 0.0) * 1000.0,
        [],
        plasticity=True,
        record=True,
    )
    trained_weights = net.weights_vector()
    trained_eval = evaluate_regime(net, regime, repetitions=eval_repetitions)
    initial_trials_to_criterion = first_criterion_repetition(training_history, regime.criterion_score)
    initial_trials_for_savings = (
        int(initial_trials_to_criterion)
        if initial_trials_to_criterion is not None
        else int(trained_repetitions)
    )

    naive_weight_control_net = copy.deepcopy(net)
    naive_weight_control_net.set_weights(baseline_weights)
    naive_weight_control_eval = evaluate_regime(
        naive_weight_control_net,
        regime,
        repetitions=eval_repetitions,
    )
    return {
        "task_name": regime.name,
        "seed": int(seed),
        "regime": regime,
        "trained_net": net,
        "baseline_weights": baseline_weights,
        "trained_weights": trained_weights,
        "baseline_eval": baseline_eval,
        "trained_eval": trained_eval,
        "naive_weight_control_eval": naive_weight_control_eval,
        "training_repetitions": int(trained_repetitions),
        "initial_trials_to_criterion": (
            int(initial_trials_to_criterion)
            if initial_trials_to_criterion is not None
            else -1
        ),
        "initial_trials_for_savings": int(initial_trials_for_savings),
        "training_reached_criterion": bool(trained_eval.score >= regime.criterion_score),
        "training_stop_at_criterion": bool(stop_at_criterion),
        "training_history": _history_row(training_history),
        "training_elapsed_s": perf_counter() - training_started,
        "consolidation_rest_s": float(max(float(consolidation_rest_s), 0.0)),
        "consolidation_neuron_spikes": int(consolidation_activity.total_neuron_spikes),
    }


def evaluate_protocol_from_seed_state(
    seed_state: dict[str, Any],
    protocol: ResetProtocol,
    *,
    eval_repetitions: int | None = None,
    measure_relearning: bool = False,
    relearn_only_if_forgot: bool = False,
    relearn_repetitions: int | None = None,
) -> dict[str, Any]:
    """Run one reset/no-reset protocol pair from a cached trained seed state."""
    seed = int(seed_state["seed"])
    regime = seed_state["regime"]
    reset_net = copy.deepcopy(seed_state["trained_net"])
    no_reset_net = copy.deepcopy(seed_state["trained_net"])

    reset_activity, total_pulses = apply_reset_protocol(reset_net, protocol, seed=seed + 10_000)
    no_reset_activity = no_reset_net.advance(
        protocol.duration_s * 1000.0,
        [],
        plasticity=True,
        record=True,
    )

    reset_weights = reset_net.weights_vector()
    no_reset_weights = no_reset_net.weights_vector()
    reset_eval = evaluate_regime(reset_net, regime, repetitions=eval_repetitions)
    no_reset_eval = evaluate_regime(no_reset_net, regime, repetitions=eval_repetitions)
    baseline_eval = seed_state["baseline_eval"]
    trained_eval = seed_state["trained_eval"]
    naive_weight_control_eval = seed_state["naive_weight_control_eval"]
    relearning_row: dict[str, Any] = {}
    flags = forgetting_flags(
        reset_score=reset_eval.score,
        no_reset_score=no_reset_eval.score,
        criterion_score=regime.criterion_score,
    )
    should_relearn = measure_relearning and (flags["made_forget"] or not relearn_only_if_forgot)
    if measure_relearning and not should_relearn:
        relearning_row = {
            "relearn_measured": False,
            "relearn_skipped_reason": "did_not_forget",
            "relearn_trials": np.nan,
            "relearn_score": np.nan,
            "relearn_reached_criterion": np.nan,
            "relearn_savings": np.nan,
            "relearn_history": "",
            "relearn_elapsed_s": 0.0,
        }
    if should_relearn:
        relearning_started = perf_counter()
        relearn_net = copy.deepcopy(reset_net)
        relearn_trials, relearn_eval, relearn_history = train_regime(
            relearn_net,
            regime,
            max_repetitions=relearn_repetitions,
            eval_repetitions=eval_repetitions,
            stop_at_criterion=True,
        )
        initial_trials = int(seed_state["initial_trials_for_savings"])
        relearning_row = {
            "relearn_measured": True,
            "relearn_skipped_reason": "",
            "relearn_trials": int(relearn_trials),
            "relearn_score": float(relearn_eval.score),
            "relearn_reached_criterion": bool(relearn_eval.score >= regime.criterion_score),
            "relearn_savings": savings_score(initial_trials, int(relearn_trials)),
            "relearn_history": _history_row(relearn_history),
            "relearn_elapsed_s": float(perf_counter() - relearning_started),
        }

    return {
        "task_name": seed_state["task_name"],
        "seed": seed,
        **_protocol_row(protocol),
        "total_pulses": int(total_pulses),
        **_evaluation_controls_row(
            baseline_eval=baseline_eval,
            trained_eval=trained_eval,
            naive_weight_control_eval=naive_weight_control_eval,
            no_reset_eval=no_reset_eval,
        ),
        "training_repetitions": int(seed_state["training_repetitions"]),
        "initial_trials_to_criterion": int(seed_state["initial_trials_to_criterion"]),
        "initial_trials_for_savings": int(seed_state["initial_trials_for_savings"]),
        "training_reached_criterion": bool(seed_state["training_reached_criterion"]),
        "training_stop_at_criterion": bool(seed_state["training_stop_at_criterion"]),
        "training_history": seed_state["training_history"],
        "training_elapsed_s": float(seed_state["training_elapsed_s"]),
        "consolidation_rest_s": float(seed_state["consolidation_rest_s"]),
        "consolidation_neuron_spikes": int(seed_state["consolidation_neuron_spikes"]),
        "reset_score": float(reset_eval.score),
        "no_reset_score": float(no_reset_eval.score),
        "reset_minus_no_reset_score": float(reset_eval.score - no_reset_eval.score),
        "forgetting_score": float(no_reset_eval.score - reset_eval.score),
        "criterion_score": float(regime.criterion_score),
        **flags,
        "reset_positive_response_probability": float(reset_eval.positive_response_probability),
        "no_reset_positive_response_probability": float(no_reset_eval.positive_response_probability),
        "reset_negative_response_probability": float(reset_eval.negative_response_probability),
        "no_reset_negative_response_probability": float(no_reset_eval.negative_response_probability),
        **_weight_metrics(
            seed_state["baseline_weights"],
            seed_state["trained_weights"],
            reset_weights,
            no_reset_weights,
        ),
        "reset_window_neuron_spikes_reset": int(reset_activity.total_neuron_spikes),
        "reset_window_neuron_spikes_no_reset": int(no_reset_activity.total_neuron_spikes),
        "reset_window_neuron_spikes_delta": int(
            reset_activity.total_neuron_spikes - no_reset_activity.total_neuron_spikes
        ),
        **baseline_eval.to_row("baseline"),
        **trained_eval.to_row("trained"),
        **reset_eval.to_row("reset"),
        **no_reset_eval.to_row("no_reset"),
        **naive_weight_control_eval.to_row("naive_weight_control"),
        **relearning_row,
    }


def run_regime_reset_trial(
    culture: CultureConfig,
    regime: TaskRegime,
    protocol: ResetProtocol,
    *,
    seed: int,
    warmup_s: float = 0.5,
    consolidation_rest_s: float = 1.0,
    training_repetitions: int | None = None,
    eval_repetitions: int | None = None,
    stop_at_criterion: bool = False,
    measure_relearning: bool = False,
    relearn_only_if_forgot: bool = False,
    relearn_repetitions: int | None = None,
) -> dict[str, Any]:
    """Run one train, reset/no-reset, and post-task comparison for a task regime."""
    seed_state = prepare_regime_seed_state(
        culture,
        regime,
        seed=seed,
        warmup_s=warmup_s,
        consolidation_rest_s=consolidation_rest_s,
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=stop_at_criterion,
    )
    return evaluate_protocol_from_seed_state(
        seed_state,
        protocol,
        eval_repetitions=eval_repetitions,
        measure_relearning=measure_relearning,
        relearn_only_if_forgot=relearn_only_if_forgot,
        relearn_repetitions=relearn_repetitions,
    )


def run_regime_seed_protocols(
    culture: CultureConfig,
    regime: TaskRegime,
    protocols: list[ResetProtocol],
    *,
    seed: int,
    warmup_s: float = 0.5,
    consolidation_rest_s: float = 1.0,
    training_repetitions: int | None = None,
    eval_repetitions: int | None = None,
    stop_at_criterion: bool = False,
    measure_relearning: bool = False,
    relearn_only_if_forgot: bool = False,
    relearn_repetitions: int | None = None,
) -> list[dict[str, Any]]:
    """Train one seed once, then evaluate every requested protocol clone."""
    seed_state = prepare_regime_seed_state(
        culture,
        regime,
        seed=seed,
        warmup_s=warmup_s,
        consolidation_rest_s=consolidation_rest_s,
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=stop_at_criterion,
    )
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        started = perf_counter()
        row = evaluate_protocol_from_seed_state(
            seed_state,
            protocol,
            eval_repetitions=eval_repetitions,
            measure_relearning=measure_relearning,
            relearn_only_if_forgot=relearn_only_if_forgot,
            relearn_repetitions=relearn_repetitions,
        )
        row["job_elapsed_s"] = perf_counter() - started
        rows.append(row)
    return rows


def _run_seed_bundle(
    args: tuple[
        CultureConfig,
        TaskRegime,
        list[ResetProtocol],
        int,
        float,
        float,
        int | None,
        int | None,
        bool,
        bool,
        bool,
        int | None,
    ],
) -> list[dict[str, Any]]:
    (
        culture,
        regime,
        protocols,
        seed,
        warmup_s,
        consolidation_rest_s,
        training_repetitions,
        eval_repetitions,
        stop_at_criterion,
        measure_relearning,
        relearn_only_if_forgot,
        relearn_repetitions,
    ) = args
    return run_regime_seed_protocols(
        culture,
        regime,
        protocols,
        seed=seed,
        warmup_s=warmup_s,
        consolidation_rest_s=consolidation_rest_s,
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=stop_at_criterion,
        measure_relearning=measure_relearning,
        relearn_only_if_forgot=relearn_only_if_forgot,
        relearn_repetitions=relearn_repetitions,
    )


def run_regime_grid(
    culture: CultureConfig,
    regime: TaskRegime,
    protocols: list[ResetProtocol],
    *,
    seeds: list[int] | tuple[int, ...],
    warmup_s: float = 0.5,
    consolidation_rest_s: float = 1.0,
    training_repetitions: int | None = None,
    eval_repetitions: int | None = None,
    stop_at_criterion: bool = False,
    measure_relearning: bool = False,
    relearn_only_if_forgot: bool = False,
    relearn_repetitions: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Run a protocol x seed grid while training each seed only once."""
    jobs = [
        (
            culture,
            regime,
            list(protocols),
            int(seed),
            warmup_s,
            consolidation_rest_s,
            training_repetitions,
            eval_repetitions,
            stop_at_criterion,
            measure_relearning,
            relearn_only_if_forgot,
            relearn_repetitions,
        )
        for seed in seeds
    ]
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            rows.extend(_run_seed_bundle(job))
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(jobs))) as executor:
            futures = [executor.submit(_run_seed_bundle, job) for job in jobs]
            for future in as_completed(futures):
                rows.extend(future.result())
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = perf_counter() - started
    df.attrs["workers"] = int(workers)
    df.attrs["jobs"] = len(protocols) * len(seeds)
    return df


def summarize_regime_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one task-regime grid by protocol."""
    if df.empty:
        return df.copy()
    first_fields = ["task_name", "beta", "schedule", "spatial_mode", "duration_s", "current_uA", "pulse_width_us"]
    mean_fields = [
        "total_pulses",
        "baseline_score",
        "trained_score",
        "training_score_delta",
        "training_repetitions",
        "initial_trials_to_criterion",
        "initial_trials_for_savings",
        "training_reached_criterion",
        "reset_score",
        "no_reset_score",
        "reset_minus_no_reset_score",
        "forgetting_score",
        "criterion_score",
        "score_drop",
        "criterion_forget",
        "made_forget",
        "naive_weight_control_score",
        "naive_weight_control_minus_trained_score",
        "naive_weight_control_minus_no_reset_score",
        "naive_weight_control_displacement_norm",
        "trained_delta_norm",
        "reset_post_minus_trained_norm",
        "no_reset_post_minus_trained_norm",
        "reset_minus_no_reset_weight_norm",
        "weight_erasure_reset",
        "weight_erasure_no_reset",
        "residual_trace_correlation_reset",
        "residual_trace_correlation_no_reset",
        "erasure_projection_reset_vs_no_reset",
        "reset_window_neuron_spikes_delta",
        "consolidation_neuron_spikes",
        "relearn_trials",
        "relearn_measured",
        "relearn_score",
        "relearn_reached_criterion",
        "relearn_savings",
        "relearn_elapsed_s",
    ]
    aggregations: dict[str, Any] = {
        field: "first"
        for field in first_fields
        if field in df.columns
    }
    aggregations.update({field: "mean" for field in mean_fields if field in df.columns})
    aggregations["seed"] = "count"
    return (
        df.groupby("protocol_id", as_index=False)
        .agg(aggregations)
        .rename(columns={"seed": "replicates"})
        .sort_values(
            ["task_name", "reset_minus_no_reset_score", "reset_minus_no_reset_weight_norm"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )
