from __future__ import annotations

import copy

import numpy as np

from .specs import ProbeMetrics, ProbeSpec, RegimeEvaluation, TaskRegime, TrainingTrialSpec, mean_or_zero


def evaluate_probe(net, probe: ProbeSpec, *, repetitions: int) -> ProbeMetrics:
    """Evaluate a single probe from one cloned network state."""
    successes = 0
    target_spikes = 0
    total_spikes = 0
    latencies: list[float] = []
    start_ms, stop_ms = probe.response_window_ms
    targets = set(int(channel) for channel in probe.target_channels)
    for _ in range(max(int(repetitions), 1)):
        activity = net.advance(
            probe.duration_ms,
            probe.events,
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

    denom = max(int(repetitions), 1)
    return ProbeMetrics(
        response_probability=float(successes / denom),
        target_spikes_per_trial=float(target_spikes / denom),
        total_spikes_per_trial=float(total_spikes / denom),
        mean_first_latency_ms=float(np.mean(latencies)) if latencies else np.nan,
    )


def evaluate_regime(net, regime: TaskRegime, *, repetitions: int | None = None) -> RegimeEvaluation:
    """Evaluate all probes from identical cloned network states."""
    regime.validate()
    repeats = int(repetitions or regime.eval_repetitions)
    probes = {
        probe.name: evaluate_probe(copy.deepcopy(net), probe, repetitions=repeats)
        for probe in regime.probes
    }
    positive = [probes[probe.name] for probe in regime.probes if probe.is_positive]
    negative = [probes[probe.name] for probe in regime.probes if probe.is_negative]
    positive_response = mean_or_zero([item.response_probability for item in positive])
    negative_response = mean_or_zero([item.response_probability for item in negative])
    positive_spikes = mean_or_zero([item.target_spikes_per_trial for item in positive])
    negative_spikes = mean_or_zero([item.target_spikes_per_trial for item in negative])
    return RegimeEvaluation(
        task_name=regime.name,
        score=float(positive_response - negative_response),
        positive_response_probability=positive_response,
        negative_response_probability=negative_response,
        positive_target_spikes_per_trial=positive_spikes,
        negative_target_spikes_per_trial=negative_spikes,
        probes=probes,
    )


def run_training_trial(net, trial: TrainingTrialSpec):
    """Run one plasticity-on training trial."""
    return net.advance(trial.duration_ms, trial.events, plasticity=True, record=True)


def train_regime(
    net,
    regime: TaskRegime,
    *,
    max_repetitions: int | None = None,
    eval_repetitions: int | None = None,
    stop_at_criterion: bool = True,
) -> tuple[int, RegimeEvaluation, tuple[float, ...]]:
    """Train a regime until its positive-vs-negative score reaches criterion."""
    regime.validate()
    limit = int(max_repetitions or regime.max_training_repetitions)
    history: list[float] = []
    latest = evaluate_regime(net, regime, repetitions=eval_repetitions)
    history.append(latest.score)
    if not regime.training_trials:
        return 0, latest, tuple(history)
    if stop_at_criterion and latest.score >= regime.criterion_score:
        return 0, latest, tuple(history)

    for repetition in range(1, limit + 1):
        for trial in regime.training_trials:
            run_training_trial(net, trial)
        latest = evaluate_regime(net, regime, repetitions=eval_repetitions)
        history.append(latest.score)
        if stop_at_criterion and latest.score >= regime.criterion_score:
            return repetition, latest, tuple(history)
    return limit, latest, tuple(history)
