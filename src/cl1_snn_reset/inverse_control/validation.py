from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cl1_snn_reset.config import ExperimentConfig
from cl1_snn_reset.experiment import record_spontaneous_activity
from cl1_snn_reset.metrics import (
    criticality,
    health_metrics,
    path_erasure_score,
    savings_score,
    weight_erasure_score,
)
from cl1_snn_reset.network import build_network
from cl1_snn_reset.task import evaluate_task, train_to_criterion

from .inverse_optimizer import CandidateProtocol
from .pulse_compiler import compile_program_to_stim_events, estimate_energy_cost
from .state_projectors import StateProjector, StateVectorSpec, build_target_state


def validate_candidate_against_no_reset(
    *,
    candidate: CandidateProtocol,
    experiment_config: ExperimentConfig,
    projector: StateProjector,
    state_spec: StateVectorSpec,
    seed: int,
    target_mode: str = "trace_removed",
    relearn: bool = True,
) -> dict[str, Any]:
    cfg = replace(experiment_config, seed=int(seed))
    trained_net, baseline_state, trained_state, baseline_activity, initial = _train_state(
        cfg,
        projector,
        seed,
    )
    duration_s = candidate.stim_program.total_duration_s
    wait_net = copy.deepcopy(trained_net)
    wait_activity = wait_net.advance(duration_s * 1000.0, [], plasticity=True, record=True)
    no_reset_state = projector.project(
        wait_net,
        activity=wait_activity,
        baseline_activity=baseline_activity,
    )
    events = compile_program_to_stim_events(candidate.stim_program)
    stim_net = copy.deepcopy(trained_net)
    stim_activity = stim_net.advance(duration_s * 1000.0, events, plasticity=True, record=True)
    stimmed_state = projector.project(
        stim_net,
        activity=stim_activity,
        baseline_activity=baseline_activity,
    )
    target = build_target_state(
        state_spec,
        baseline_state,
        trained_state,
        no_reset_state,
        mode=target_mode,
    )
    task_mask = state_spec.group_mask(("task_path", "readout"))
    off_task_mask = ~task_mask
    task_no = _masked_norm(no_reset_state - target, task_mask)
    task_stim = _masked_norm(stimmed_state - target, task_mask)
    stimulus_effect = stimmed_state - no_reset_state
    post_behavior = evaluate_task(copy.deepcopy(stim_net), cfg.task, trials=cfg.task.eval_trials)
    if relearn:
        relearn_result = train_to_criterion(copy.deepcopy(stim_net), cfg.task)
        savings = savings_score(initial.trials_to_criterion, relearn_result.trials_to_criterion)
        relearn_trials = relearn_result.trials_to_criterion
    else:
        savings = float("nan")
        relearn_trials = -1
    health = health_metrics(stim_activity, duration_s=duration_s)
    crit = criticality(stim_activity, naive=baseline_activity)
    path0 = _path(cfg, build_network(cfg.culture, seed=int(seed)))
    path_trained = trained_net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    path_no = wait_net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    path_stim = stim_net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    row = {
        "protocol_id": candidate.protocol_id,
        "seed": int(seed),
        "program_family": candidate.stim_program.family,
        "duration_s": duration_s,
        "energy_cost": estimate_energy_cost(candidate.stim_program, events),
        "validated_causal_task_erasure": float(task_no - task_stim),
        "validated_weight_erasure": weight_erasure_score(
            baseline_state_weights(trained_net, cfg, seed),
            trained_net.weights_vector(),
            stim_net.weights_vector(),
        ),
        "validated_path_erasure": path_erasure_score(path0, path_trained, path_stim),
        "validated_no_reset_path_erasure": path_erasure_score(path0, path_trained, path_no),
        "validated_residual_performance": float(post_behavior),
        "validated_savings": float(savings),
        "validated_relearn_trials": int(relearn_trials),
        "validated_trace_auc": _trace_feature(state_spec, stimmed_state),
        "validated_health": health.score,
        "validated_criticality_distance": crit.distance_from_naive,
        "validated_orthogonal_damage": _masked_norm(stimmed_state - target, off_task_mask),
        "stimulus_effect_norm": float(np.linalg.norm(stimulus_effect)),
        "post_reset_firing_rate_hz": health.firing_rate_hz,
        "silent_fraction": _silent_fraction(stim_activity),
        "hyperactive_fraction": _hyperactive_fraction(stim_activity),
    }
    row["beats_no_reset"] = bool(
        row["validated_causal_task_erasure"] > 0.0
        and row["stimulus_effect_norm"] > 1e-9
    )
    row["passes_health_criterion"] = bool(row["validated_health"] >= 0.10)
    row["passes_generalization_criterion"] = False
    return row


def validate_candidates_against_no_reset(
    *,
    candidates: list[CandidateProtocol],
    experiment_config: ExperimentConfig,
    projector: StateProjector,
    state_spec: StateVectorSpec,
    seeds: tuple[int, ...] | list[int],
    target_mode: str = "trace_removed",
    output_dir: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = candidates[:limit] if limit is not None else candidates
    for candidate in selected:
        for seed in seeds:
            rows.append(
                validate_candidate_against_no_reset(
                    candidate=candidate,
                    experiment_config=experiment_config,
                    projector=projector,
                    state_spec=state_spec,
                    seed=int(seed),
                    target_mode=target_mode,
                )
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        grouped = df.groupby("protocol_id")["beats_no_reset"].transform("mean")
        df["passes_generalization_criterion"] = grouped >= 0.5
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "candidate_validation.csv", index=False)
        df.to_csv(output_dir / "paired_no_reset_validation.csv", index=False)
    return df


def bootstrap_candidate_effects(
    validation: pd.DataFrame,
    *,
    samples: int = 1000,
    random_seed: int = 123,
) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    rows = []
    for protocol_id, group in validation.groupby("protocol_id"):
        values = group["validated_causal_task_erasure"].to_numpy(dtype=np.float64)
        boot = []
        for _ in range(int(samples)):
            draw = rng.choice(values, size=values.size, replace=True)
            boot.append(float(np.mean(draw)))
        rows.append(
            {
                "protocol_id": protocol_id,
                "mean_causal_task_erasure": float(np.mean(values)),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
            }
        )
    return pd.DataFrame(rows)


def _train_state(cfg: ExperimentConfig, projector: StateProjector, seed: int):
    net = build_network(cfg.culture, seed=int(seed))
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)
    baseline_probe = copy.deepcopy(net)
    baseline_activity = record_spontaneous_activity(baseline_probe, cfg.readout_window_s)
    baseline_state = projector.project(net, activity=baseline_activity)
    initial = train_to_criterion(net, cfg.task)
    trained_probe = copy.deepcopy(net)
    trained_activity = record_spontaneous_activity(trained_probe, cfg.readout_window_s)
    trained_state = projector.project(
        net,
        activity=trained_activity,
        baseline_activity=baseline_activity,
    )
    return net, baseline_state, trained_state, baseline_activity, initial


def baseline_state_weights(trained_net, cfg: ExperimentConfig, seed: int) -> np.ndarray:
    baseline_net = build_network(cfg.culture, seed=int(seed))
    if cfg.warmup_s > 0:
        baseline_net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)
    return baseline_net.weights_vector()


def _path(cfg: ExperimentConfig, net) -> float:
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)
    return net.path_strength(cfg.task.input_channels, cfg.task.target_channels)


def _masked_norm(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(values[mask]))


def _trace_feature(spec: StateVectorSpec, state: np.ndarray) -> float:
    names = list(spec.feature_names)
    if "readout:trace_auc_proxy" not in names:
        return float("nan")
    return float(state[names.index("readout:trace_auc_proxy")])


def _silent_fraction(activity) -> float:
    if activity.duration_ms <= 0.0:
        return 1.0
    counts = np.bincount(activity.channels, weights=activity.counts, minlength=64)
    return float(np.mean(counts <= 0))


def _hyperactive_fraction(activity) -> float:
    duration_s = max(activity.duration_ms / 1000.0, 1e-9)
    counts = np.bincount(activity.channels, weights=activity.counts, minlength=64)
    rates = counts / duration_s
    return float(np.mean(rates > 80.0))
