from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TaskConfig
from .electrodes import ChannelActivity, StimEvent


@dataclass(frozen=True)
class TrainingResult:
    trials_to_criterion: int
    response_probability: float
    reached_criterion: bool
    history: tuple[float, ...]


def paired_training_trial(net, cfg: TaskConfig) -> ChannelActivity:
    """Run one input-target pairing trial through the electrode interface."""
    events = [
        StimEvent(
            time_us        = 0,
            channels       = tuple(cfg.input_channels),
            current_uA     = cfg.input_current_uA,
            pulse_width_us = cfg.pulse_width_us,
        ),
        StimEvent(
            time_us        = int(round(cfg.pair_delay_ms * 1000.0)),
            channels       = tuple(cfg.target_channels),
            current_uA     = cfg.target_current_uA,
            pulse_width_us = cfg.pulse_width_us,
        ),
    ]
    return net.advance(cfg.inter_trial_ms, events, plasticity=True, record=True)


def evaluate_task(net, cfg: TaskConfig, *, trials: int | None = None) -> float:
    """Measure target response probability after input-channel stimulation."""
    n_trials = int(trials or cfg.eval_trials)
    successes = 0
    start_ms, stop_ms = cfg.response_window_ms
    targets = set(int(channel) for channel in cfg.target_channels)
    for _ in range(n_trials):
        activity = net.advance(
            cfg.inter_trial_ms,
            [
                StimEvent(
                    time_us        = 0,
                    channels       = tuple(cfg.input_channels),
                    current_uA     = cfg.input_current_uA,
                    pulse_width_us = cfg.pulse_width_us,
                )
            ],
            plasticity=False,
            record=True,
        )
        in_window = (
            (activity.spike_times_ms >= start_ms)
            & (activity.spike_times_ms <= stop_ms)
            & np.isin(activity.channels, list(targets))
        )
        successes += int(np.any(in_window))
    return successes / max(n_trials, 1)


def train_to_criterion(net, cfg: TaskConfig) -> TrainingResult:
    """Train by paired stimulation until response probability crosses criterion."""
    history: list[float] = []
    latest = evaluate_task(net, cfg, trials=cfg.eval_trials)
    history.append(latest)
    if latest >= cfg.criterion_response_probability:
        return TrainingResult(0, latest, True, tuple(history))

    for trial in range(1, cfg.max_trials + 1):
        paired_training_trial(net, cfg)
        if trial % cfg.eval_interval_trials != 0 and trial < cfg.max_trials:
            continue
        latest = evaluate_task(net, cfg, trials=cfg.eval_trials)
        history.append(latest)
        if latest >= cfg.criterion_response_probability:
            return TrainingResult(trial, latest, True, tuple(history))
    return TrainingResult(cfg.max_trials, latest, False, tuple(history))
