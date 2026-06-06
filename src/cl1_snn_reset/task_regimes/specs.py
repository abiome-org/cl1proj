from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..electrodes import StimEvent


def stim_event_ms(
    time_ms: float,
    channels: tuple[int, ...],
    current_uA: float,
    pulse_width_us: int = 160,
) -> StimEvent:
    """Build a stimulation event with millisecond timing."""
    return StimEvent(
        time_us=int(round(float(time_ms) * 1000.0)),
        channels=tuple(int(channel) for channel in channels),
        current_uA=float(current_uA),
        pulse_width_us=int(pulse_width_us),
    )


@dataclass(frozen=True)
class TrainingTrialSpec:
    """One plasticity-on trial used by a task regime."""

    name: str
    events: tuple[StimEvent, ...]
    duration_ms: float


@dataclass(frozen=True)
class ProbeSpec:
    """One plasticity-off probe used to score a task regime."""

    name: str
    events: tuple[StimEvent, ...]
    target_channels: tuple[int, ...]
    response_window_ms: tuple[float, float]
    duration_ms: float
    expected: str = "positive"

    @property
    def is_positive(self) -> bool:
        return self.expected == "positive"

    @property
    def is_negative(self) -> bool:
        return self.expected == "negative"


@dataclass(frozen=True)
class ProbeMetrics:
    """Repeated-trial score for one probe."""

    response_probability: float
    target_spikes_per_trial: float
    total_spikes_per_trial: float
    mean_first_latency_ms: float

    def to_prefixed_row(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_response_probability": self.response_probability,
            f"{prefix}_target_spikes_per_trial": self.target_spikes_per_trial,
            f"{prefix}_total_spikes_per_trial": self.total_spikes_per_trial,
            f"{prefix}_mean_first_latency_ms": self.mean_first_latency_ms,
        }


@dataclass(frozen=True)
class RegimeEvaluation:
    """Task-level evaluation with positive-vs-negative contrast."""

    task_name: str
    score: float
    positive_response_probability: float
    negative_response_probability: float
    positive_target_spikes_per_trial: float
    negative_target_spikes_per_trial: float
    probes: dict[str, ProbeMetrics]

    def to_row(self, prefix: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            f"{prefix}_score": self.score,
            f"{prefix}_positive_response_probability": self.positive_response_probability,
            f"{prefix}_negative_response_probability": self.negative_response_probability,
            f"{prefix}_positive_target_spikes_per_trial": self.positive_target_spikes_per_trial,
            f"{prefix}_negative_target_spikes_per_trial": self.negative_target_spikes_per_trial,
        }
        for name, metrics in self.probes.items():
            safe_name = name.replace("-", "_").replace(" ", "_")
            row.update(metrics.to_prefixed_row(f"{prefix}_{safe_name}"))
        return row


@dataclass(frozen=True)
class TaskRegime:
    """Reusable behavioral task definition for reset experiments."""

    name: str
    description: str
    training_trials: tuple[TrainingTrialSpec, ...]
    probes: tuple[ProbeSpec, ...]
    criterion_score: float
    max_training_repetitions: int
    eval_repetitions: int

    def validate(self) -> None:
        if not self.probes:
            raise ValueError("TaskRegime requires at least one probe.")
        if not any(probe.is_positive for probe in self.probes):
            raise ValueError("TaskRegime requires at least one positive probe.")
        if not any(probe.is_negative for probe in self.probes):
            raise ValueError("TaskRegime requires at least one negative or sham probe.")
        for trial in self.training_trials:
            if trial.duration_ms <= 0.0:
                raise ValueError(f"Training trial {trial.name!r} has non-positive duration.")
        for probe in self.probes:
            start_ms, stop_ms = probe.response_window_ms
            if stop_ms <= start_ms:
                raise ValueError(f"Probe {probe.name!r} has invalid response window.")
            if probe.duration_ms <= 0.0:
                raise ValueError(f"Probe {probe.name!r} has non-positive duration.")

    def to_metadata(self) -> dict[str, Any]:
        def event_payload(event: StimEvent) -> dict[str, Any]:
            return {
                "time_us": int(event.time_us),
                "channels": list(event.channels),
                "current_uA": float(event.current_uA),
                "pulse_width_us": int(event.pulse_width_us),
            }

        return {
            "name": self.name,
            "description": self.description,
            "criterion_score": float(self.criterion_score),
            "max_training_repetitions": int(self.max_training_repetitions),
            "eval_repetitions": int(self.eval_repetitions),
            "training_trials": [
                {
                    "name": trial.name,
                    "duration_ms": float(trial.duration_ms),
                    "events": [event_payload(event) for event in trial.events],
                }
                for trial in self.training_trials
            ],
            "probes": [
                {
                    "name": probe.name,
                    "duration_ms": float(probe.duration_ms),
                    "target_channels": list(probe.target_channels),
                    "response_window_ms": list(probe.response_window_ms),
                    "expected": probe.expected,
                    "events": [event_payload(event) for event in probe.events],
                }
                for probe in self.probes
            ],
        }


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
