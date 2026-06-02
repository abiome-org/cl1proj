from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import TrialArtifacts
from .config import ExperimentConfig
from .electrodes import ChannelActivity
from .metrics import TrialMetrics, compute_trial_metrics
from .network import NetworkSnapshot, build_network
from .protocols import ResetProtocol, protocol_events
from .task import TaskConfig, TrainingResult, evaluate_task, train_to_criterion
from .trace_probe import trace_auc_proxy


@dataclass(frozen=True)
class TrialResult:
    metrics: TrialMetrics
    initial: TrainingResult
    relearn: TrainingResult
    snapshots: dict[str, NetworkSnapshot] | None = None
    activities: dict[str, ChannelActivity] | None = None

    def to_row(self) -> dict[str, Any]:
        return self.metrics.to_row()


@dataclass(frozen=True)
class _PhaseCapture:
    weights: np.ndarray
    path_strength: float
    activity: ChannelActivity | None = None
    snapshot: NetworkSnapshot | None = None


def record_spontaneous_activity(net, duration_s: float) -> ChannelActivity:
    return net.advance(duration_s * 1000.0, [], plasticity=False, record=True)


def apply_reset_protocol(net, protocol: ResetProtocol, seed: int) -> tuple[ChannelActivity, int]:
    rng = np.random.default_rng(seed)
    events = protocol_events(protocol, n_channels=net.cfg.n_electrodes, rng=rng)
    activity = net.advance(protocol.duration_s * 1000.0, events, plasticity=True, record=True)
    total_pulses = int(sum(len(event.channels) for event in events))
    return activity, total_pulses


def _capture_phase(
    net,
    task: TaskConfig,
    *,
    readout_window_s: float,
    keep_snapshots: bool,
    record_activity: bool = True,
) -> _PhaseCapture:
    weights = net.weights_vector()
    path_strength = net.path_strength(task.input_channels, task.target_channels)
    activity = record_spontaneous_activity(net, readout_window_s) if record_activity else None
    snapshot = net.snapshot() if keep_snapshots else None
    return _PhaseCapture(
        weights=weights,
        path_strength=path_strength,
        activity=activity,
        snapshot=snapshot,
    )


def run_trial(cfg: ExperimentConfig, protocol: ResetProtocol, seed: int | None = None) -> TrialResult:
    trial_seed = int(cfg.seed if seed is None else seed)
    net = build_network(cfg.culture, seed=trial_seed)
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)

    baseline = _capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=cfg.keep_snapshots,
    )
    initial = train_to_criterion(net, cfg.task)
    trained = _capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=cfg.keep_snapshots,
    )

    A_post_reset, total_pulses = apply_reset_protocol(net, protocol, seed=trial_seed + 10_000)
    post_reset = _capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=cfg.keep_snapshots,
    )

    post_behavior = evaluate_task(net, cfg.task, trials=cfg.task.eval_trials)
    relearn = train_to_criterion(net, cfg.task)
    relearn_snap = net.snapshot() if cfg.keep_snapshots else None

    artifacts = TrialArtifacts(
        W0=baseline.weights,
        Wtrained=trained.weights,
        Wpost=post_reset.weights,
        A0=baseline.activity,
        Apost=post_reset.activity,
        initial=initial,
        relearn=relearn,
        post_behavior=post_behavior,
        protocol=protocol,
        seed=trial_seed,
        total_pulses=total_pulses,
        trace_auc=trace_auc_proxy(baseline.activity, post_reset.activity),
        path0=baseline.path_strength,
        path_trained=trained.path_strength,
        path_post=post_reset.path_strength,
    )
    metrics = compute_trial_metrics(artifacts)

    snapshots = None
    activities = None
    if cfg.keep_snapshots:
        snapshots = {
            "W0": baseline.snapshot,
            "Wtrained": trained.snapshot,
            "Wpost": post_reset.snapshot,
            "Wrelearn": relearn_snap,
        }
        activities = {
            "A0": baseline.activity,
            "Atrained": trained.activity,
            "Apost_reset_window": A_post_reset,
            "Apost": post_reset.activity,
        }
    return TrialResult(
        metrics    = metrics,
        initial    = initial,
        relearn    = relearn,
        snapshots  = snapshots,
        activities = activities,
    )
