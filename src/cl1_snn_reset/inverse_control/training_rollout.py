from __future__ import annotations

import copy
from typing import Any

from cl1_snn_reset.config import ExperimentConfig
from cl1_snn_reset.experiment import record_spontaneous_activity
from cl1_snn_reset.network import build_network
from cl1_snn_reset.task import train_to_criterion

from .state_projectors import StateProjector


def train_baseline_and_task_states(
    cfg: ExperimentConfig,
    projector: StateProjector,
    seed: int,
) -> tuple[Any, Any, Any, Any, Any]:
    net = build_network(cfg.culture, seed=int(seed))
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)
    baseline_probe = copy.deepcopy(net)
    baseline_activity = record_spontaneous_activity(baseline_probe, cfg.readout_window_s)
    baseline_state = projector.project(net, activity=baseline_activity)
    training = train_to_criterion(net, cfg.task)
    trained_probe = copy.deepcopy(net)
    trained_activity = record_spontaneous_activity(trained_probe, cfg.readout_window_s)
    trained_state = projector.project(
        net,
        activity=trained_activity,
        baseline_activity=baseline_activity,
    )
    return net, baseline_state, trained_state, baseline_activity, training
