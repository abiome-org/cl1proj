from __future__ import annotations

import copy
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


@dataclass(frozen=True)
class TaskBranchMetrics:
    response_probability: float
    target_spikes_per_trial: float
    total_spikes_per_trial: float
    mean_first_latency_ms: float


@dataclass(frozen=True)
class EvokedTaskMetrics:
    input_metrics: TaskBranchMetrics
    sham_metrics: TaskBranchMetrics

    @property
    def evoked_response_probability(self) -> float:
        return self.input_metrics.response_probability - self.sham_metrics.response_probability

    @property
    def evoked_target_spikes_per_trial(self) -> float:
        return self.input_metrics.target_spikes_per_trial - self.sham_metrics.target_spikes_per_trial

    @property
    def evoked_total_spikes_per_trial(self) -> float:
        return self.input_metrics.total_spikes_per_trial - self.sham_metrics.total_spikes_per_trial


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


def evaluate_task_branch(
    net,
    cfg: TaskConfig,
    *,
    trials: int | None = None,
    with_input: bool = True,
) -> TaskBranchMetrics:
    """Measure target-window activity with or without the task input pulse."""
    n_trials = int(trials or cfg.eval_trials)
    successes = 0
    target_spikes = 0
    total_spikes = 0
    latencies: list[float] = []
    start_ms, stop_ms = cfg.response_window_ms
    targets = set(int(channel) for channel in cfg.target_channels)
    events = [
        StimEvent(
            time_us        = 0,
            channels       = tuple(cfg.input_channels),
            current_uA     = cfg.input_current_uA,
            pulse_width_us = cfg.pulse_width_us,
        )
    ] if with_input else []
    for _ in range(n_trials):
        activity = net.advance(
            cfg.inter_trial_ms,
            events,
            plasticity=False,
            record=True,
        )
        total_spikes += int(activity.total_neuron_spikes)
        in_window = (
            (activity.spike_times_ms >= start_ms)
            & (activity.spike_times_ms <= stop_ms)
            & np.isin(activity.channels, list(targets))
        )
        count = int(np.count_nonzero(in_window))
        target_spikes += count
        if count:
            successes += 1
            latencies.append(float(activity.spike_times_ms[in_window].min()))
    denom = max(n_trials, 1)
    return TaskBranchMetrics(
        response_probability=float(successes / denom),
        target_spikes_per_trial=float(target_spikes / denom),
        total_spikes_per_trial=float(total_spikes / denom),
        mean_first_latency_ms=float(np.mean(latencies)) if latencies else np.nan,
    )


def evaluate_evoked_task(net, cfg: TaskConfig, *, trials: int | None = None) -> EvokedTaskMetrics:
    """Measure input-minus-sham target response from cloned network states."""
    input_metrics = evaluate_task_branch(copy.deepcopy(net), cfg, trials=trials, with_input=True)
    sham_metrics = evaluate_task_branch(copy.deepcopy(net), cfg, trials=trials, with_input=False)
    return EvokedTaskMetrics(input_metrics=input_metrics, sham_metrics=sham_metrics)


def evaluate_task(net, cfg: TaskConfig, *, trials: int | None = None) -> float:
    """Measure uncorrected target response probability after input stimulation."""
    return evaluate_task_branch(net, cfg, trials=trials, with_input=True).response_probability


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
