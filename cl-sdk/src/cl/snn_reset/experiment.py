from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .electrodes import ChannelActivity
from .metrics import TrialMetrics, compute_trial_metrics
from .network import NetworkSnapshot, build_network
from .protocols import ResetProtocol, protocol_events
from .task import TrainingResult, evaluate_task, train_to_criterion
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


def record_spontaneous_activity(net, duration_s: float) -> ChannelActivity:
    return net.advance(duration_s * 1000.0, [], plasticity=False, record=True)


def apply_reset_protocol(net, protocol: ResetProtocol, seed: int) -> tuple[ChannelActivity, int]:
    rng = np.random.default_rng(seed)
    events = protocol_events(protocol, n_channels=net.cfg.n_electrodes, rng=rng)
    activity = net.advance(protocol.duration_s * 1000.0, events, plasticity=True, record=True)
    total_pulses = int(sum(len(event.channels) for event in events))
    return activity, total_pulses


def run_trial(cfg: ExperimentConfig, protocol: ResetProtocol, seed: int | None = None) -> TrialResult:
    trial_seed = int(cfg.seed if seed is None else seed)
    net = build_network(cfg.culture, seed=trial_seed)
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)

    W0 = net.weights_vector()
    path0 = net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    A0 = record_spontaneous_activity(net, cfg.readout_window_s)
    snap0 = net.snapshot() if cfg.keep_snapshots else None

    initial = train_to_criterion(net, cfg.task)
    W_trained = net.weights_vector()
    path_trained = net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    A_trained = record_spontaneous_activity(net, cfg.readout_window_s)
    snap_trained = net.snapshot() if cfg.keep_snapshots else None

    A_post_reset, total_pulses = apply_reset_protocol(net, protocol, seed=trial_seed + 10_000)
    W_post = net.weights_vector()
    path_post = net.path_strength(cfg.task.input_channels, cfg.task.target_channels)
    A_post = record_spontaneous_activity(net, cfg.readout_window_s)
    snap_post = net.snapshot() if cfg.keep_snapshots else None

    post_behavior = evaluate_task(net, cfg.task, trials=cfg.task.eval_trials)
    relearn = train_to_criterion(net, cfg.task)
    snap_relearn = net.snapshot() if cfg.keep_snapshots else None

    metrics = compute_trial_metrics(
        W0              = W0,
        Wtrained        = W_trained,
        Wpost           = W_post,
        A0              = A0,
        Apost           = A_post,
        initial         = initial,
        relearn         = relearn,
        post_behavior   = post_behavior,
        protocol        = protocol,
        seed            = trial_seed,
        total_pulses    = total_pulses,
        trace_auc       = trace_auc_proxy(A0, A_post),
        path0           = path0,
        path_trained    = path_trained,
        path_post       = path_post,
    )
    snapshots = None
    activities = None
    if cfg.keep_snapshots:
        snapshots = {
            "W0": snap0,
            "Wtrained": snap_trained,
            "Wpost": snap_post,
            "Wrelearn": snap_relearn,
        }
        activities = {
            "A0": A0,
            "Atrained": A_trained,
            "Apost_reset_window": A_post_reset,
            "Apost": A_post,
        }
    return TrialResult(
        metrics    = metrics,
        initial    = initial,
        relearn    = relearn,
        snapshots  = snapshots,
        activities = activities,
    )
