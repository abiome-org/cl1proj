from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from .config import TaskConfig
from .electrodes import ChannelActivity
from .protocols import ResetProtocol
from .task import TrainingResult


@dataclass(frozen=True)
class HealthMetrics:
    firing_rate_hz: float
    active_channel_fraction: float
    ei_balance: float
    saturated: bool
    trainable: bool
    score: float


@dataclass(frozen=True)
class CriticalityMetrics:
    avalanche_alpha: float
    branching_ratio: float
    mean_avalanche_size: float
    distance_from_naive: float


@dataclass(frozen=True)
class TrialMetrics:
    protocol_id: str
    seed: int
    beta: float
    schedule: str
    spatial_mode: str
    duration_s: float
    current_uA: float
    pulse_width_us: int
    total_pulses: int
    initial_trials: int
    relearn_trials: int
    weight_erasure: float
    residual_performance: float
    savings: float
    criticality_distance: float
    trace_auc: float
    health: float
    energy_cost: float
    firing_rate_hz: float
    active_channel_fraction: float
    branching_ratio: float
    path_erasure: float

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def weight_erasure_score(W0: np.ndarray, Wtrained: np.ndarray, Wpost: np.ndarray) -> float:
    trained_delta = np.asarray(Wtrained, dtype=np.float64) - np.asarray(W0, dtype=np.float64)
    residual_delta = np.asarray(Wpost, dtype=np.float64) - np.asarray(W0, dtype=np.float64)
    denom = np.linalg.norm(trained_delta) + 1e-9
    return float(1.0 - np.linalg.norm(residual_delta) / denom)


def residual_trace_correlation(W0: np.ndarray, Wtrained: np.ndarray, Wpost: np.ndarray) -> float:
    trained_delta = np.asarray(Wtrained, dtype=np.float64) - np.asarray(W0, dtype=np.float64)
    residual_delta = np.asarray(Wpost, dtype=np.float64) - np.asarray(W0, dtype=np.float64)
    if np.std(trained_delta) < 1e-12 or np.std(residual_delta) < 1e-12:
        return 0.0
    return float(np.corrcoef(trained_delta, residual_delta)[0, 1])


def savings_score(initial_trials: int, relearn_trials: int) -> float:
    return float(1.0 - (float(relearn_trials) / (float(initial_trials) + 1e-9)))


def activity_features(activity: ChannelActivity, *, channel_count: int = 64) -> np.ndarray:
    counts = np.bincount(activity.channels, weights=activity.counts, minlength=channel_count).astype(np.float64)
    duration_s = max(activity.duration_ms / 1000.0, 1e-9)
    rates = counts / duration_s
    active_fraction = float(np.mean(counts > 0.0))
    total_rate = float(counts.sum() / duration_s)
    if activity.spike_times_ms.size > 1:
        isi = np.diff(np.sort(activity.spike_times_ms))
        isi_mean = float(np.mean(isi))
        isi_cv = float(np.std(isi) / max(isi_mean, 1e-9))
    else:
        isi_mean = 0.0
        isi_cv = 0.0
    return np.concatenate([
        rates,
        np.array([active_fraction, total_rate, isi_mean, isi_cv], dtype=np.float64),
    ])


def criticality(activity: ChannelActivity, naive: ChannelActivity | None = None) -> CriticalityMetrics:
    counts = activity.binned_counts(bin_ms=10.0)
    avalanche_sizes = counts.sum(axis=1)
    avalanche_sizes = avalanche_sizes[avalanche_sizes > 0]
    if avalanche_sizes.size >= 2:
        branching = float(np.mean(avalanche_sizes[1:] / np.maximum(avalanche_sizes[:-1], 1.0)))
        mean_size = float(np.mean(avalanche_sizes))
        # Crude log-log slope over the empirical CCDF; enough for screening.
        xs = np.sort(avalanche_sizes)
        ccdf = 1.0 - np.arange(xs.size) / max(xs.size, 1)
        valid = (xs > 0) & (ccdf > 0) & np.isfinite(xs) & np.isfinite(ccdf)
        if np.sum(valid) >= 2 and np.unique(xs[valid]).size >= 2:
            try:
                slope, _ = np.polyfit(np.log(xs[valid]), np.log(ccdf[valid]), 1)
                alpha = float(-slope) if np.isfinite(slope) else 0.0
            except np.linalg.LinAlgError:
                alpha = 0.0
        else:
            alpha = 0.0
    else:
        branching = 0.0
        mean_size = float(avalanche_sizes.mean()) if avalanche_sizes.size else 0.0
        alpha = 0.0
    distance = 0.0
    if naive is not None:
        a = activity_features(activity)
        b = activity_features(naive)
        distance = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
    return CriticalityMetrics(
        avalanche_alpha     = alpha,
        branching_ratio     = branching,
        mean_avalanche_size = mean_size,
        distance_from_naive = distance,
    )


def health_metrics(activity: ChannelActivity, *, duration_s: float, ei_balance: float = 1.0) -> HealthMetrics:
    total_spikes = float(np.sum(activity.counts))
    firing_rate = total_spikes / max(duration_s * 64.0, 1e-9)
    active_fraction = float(len(set(activity.channels.tolist())) / 64.0) if activity.channels.size else 0.0
    saturated = firing_rate > 80.0 or active_fraction > 0.98
    trainable = 0.01 <= firing_rate <= 80.0 and active_fraction >= 0.03 and not saturated
    rate_score = np.exp(-abs(np.log((firing_rate + 1e-6) / 3.0)))
    active_score = min(1.0, active_fraction / 0.35)
    score = float(np.clip(sqrt(max(rate_score, 0.0) * max(active_score, 0.0)), 0.0, 1.0))
    if saturated:
        score *= 0.25
    return HealthMetrics(
        firing_rate_hz          = float(firing_rate),
        active_channel_fraction = active_fraction,
        ei_balance              = float(ei_balance),
        saturated               = bool(saturated),
        trainable               = bool(trainable),
        score                   = score,
    )


def path_erasure_score(
    path0: float,
    path_trained: float,
    path_post: float,
) -> float:
    trained_delta = path_trained - path0
    residual_delta = path_post - path0
    return float(1.0 - abs(residual_delta) / (abs(trained_delta) + 1e-9))


def compute_trial_metrics(
    *,
    W0: np.ndarray,
    Wtrained: np.ndarray,
    Wpost: np.ndarray,
    A0: ChannelActivity,
    Apost: ChannelActivity,
    initial: TrainingResult,
    relearn: TrainingResult,
    post_behavior: float,
    protocol: ResetProtocol,
    seed: int,
    total_pulses: int,
    trace_auc: float,
    path0: float,
    path_trained: float,
    path_post: float,
) -> TrialMetrics:
    crit = criticality(Apost, naive=A0)
    health = health_metrics(Apost, duration_s=max(protocol.duration_s, Apost.duration_ms / 1000.0))
    return TrialMetrics(
        protocol_id             = protocol.id,
        seed                    = int(seed),
        beta                    = float(protocol.beta),
        schedule                = protocol.schedule,
        spatial_mode            = protocol.spatial_mode,
        duration_s              = float(protocol.duration_s),
        current_uA              = float(protocol.current_uA),
        pulse_width_us          = int(protocol.pulse_width_us),
        total_pulses            = int(total_pulses),
        initial_trials          = int(initial.trials_to_criterion),
        relearn_trials          = int(relearn.trials_to_criterion),
        weight_erasure          = weight_erasure_score(W0, Wtrained, Wpost),
        residual_performance    = float(post_behavior),
        savings                 = savings_score(initial.trials_to_criterion, relearn.trials_to_criterion),
        criticality_distance    = crit.distance_from_naive,
        trace_auc               = float(trace_auc),
        health                  = health.score,
        energy_cost             = protocol.total_charge_uC(total_pulses),
        firing_rate_hz          = health.firing_rate_hz,
        active_channel_fraction = health.active_channel_fraction,
        branching_ratio         = crit.branching_ratio,
        path_erasure            = path_erasure_score(path0, path_trained, path_post),
    )
