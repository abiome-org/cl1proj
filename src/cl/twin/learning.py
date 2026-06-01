from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TaskTrial:
    """One closed-loop task outcome produced while training the twin."""

    timestamp: int
    correct: bool
    response_latency_frames: int | None = None


@dataclass(frozen=True)
class LearningCurveReport:
    """Summary of whether closed-loop behavior improves across trials."""

    trial_count: int
    early_accuracy: float
    late_accuracy: float
    accuracy_delta: float
    early_latency_median_frames: float
    late_latency_median_frames: float
    latency_delta_frames: float
    passed: bool
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for experiment logs."""
        return asdict(self)


class TwinLearningEvaluator:
    """
    Evaluate task-level learning curves for closed-loop twin experiments.

    The biological north star is not just matching firing statistics; the twin
    should improve under structured feedback and degrade or search under chaotic
    feedback.  This evaluator gives tests and notebooks a stable way to quantify
    that task-level behavior without coupling the SDK to one particular game.
    """

    @classmethod
    def evaluate_trials(
        cls,
        trials: Iterable[TaskTrial | dict[str, Any] | Any],
        *,
        window_size: int | None = None,
        min_accuracy_delta: float = 0.05,
        max_latency_delta_frames: float | None = None,
    ) -> LearningCurveReport:
        """Compare early and late windows of a closed-loop training run."""
        normalized = cls._normalize_trials(trials)
        if not normalized:
            return LearningCurveReport(
                trial_count                  = 0,
                early_accuracy               = 0.0,
                late_accuracy                = 0.0,
                accuracy_delta               = 0.0,
                early_latency_median_frames  = 0.0,
                late_latency_median_frames   = 0.0,
                latency_delta_frames         = 0.0,
                passed                       = False,
                metrics                      = {"window_size": 0, "correct_by_trial": []},
            )

        resolved_window = int(window_size or max(1, len(normalized) // 4))
        resolved_window = max(1, min(resolved_window, len(normalized)))
        ordered = sorted(normalized, key=lambda trial: trial.timestamp)
        early = ordered[:resolved_window]
        late = ordered[-resolved_window:]

        early_accuracy = cls._accuracy(early)
        late_accuracy = cls._accuracy(late)
        accuracy_delta = late_accuracy - early_accuracy
        early_latency = cls._median_latency(early)
        late_latency = cls._median_latency(late)
        latency_delta = late_latency - early_latency
        latency_ok = (
            True
            if max_latency_delta_frames is None
            else latency_delta <= float(max_latency_delta_frames)
        )
        passed = accuracy_delta >= float(min_accuracy_delta) and latency_ok

        return LearningCurveReport(
            trial_count                 = len(ordered),
            early_accuracy              = early_accuracy,
            late_accuracy               = late_accuracy,
            accuracy_delta              = accuracy_delta,
            early_latency_median_frames = early_latency,
            late_latency_median_frames  = late_latency,
            latency_delta_frames        = latency_delta,
            passed                      = passed,
            metrics                     = {
                "window_size": resolved_window,
                "correct_by_trial": [trial.correct for trial in ordered],
                "latency_by_trial": [trial.response_latency_frames for trial in ordered],
            },
        )

    @staticmethod
    def _normalize_trials(trials: Iterable[TaskTrial | dict[str, Any] | Any]) -> list[TaskTrial]:
        """Accept dataclasses, dicts, or simple objects as trial records."""
        normalized: list[TaskTrial] = []
        for index, trial in enumerate(trials):
            if isinstance(trial, TaskTrial):
                normalized.append(trial)
                continue
            if isinstance(trial, dict):
                normalized.append(TaskTrial(
                    timestamp               = int(trial.get("timestamp", index)),
                    correct                 = bool(trial["correct"]),
                    response_latency_frames = trial.get("response_latency_frames"),
                ))
                continue
            normalized.append(TaskTrial(
                timestamp               = int(getattr(trial, "timestamp", index)),
                correct                 = bool(getattr(trial, "correct")),
                response_latency_frames = getattr(trial, "response_latency_frames", None),
            ))
        return normalized

    @staticmethod
    def _accuracy(trials: list[TaskTrial]) -> float:
        """Return fraction of correct outcomes in a trial window."""
        if not trials:
            return 0.0
        return float(sum(1 for trial in trials if trial.correct) / len(trials))

    @staticmethod
    def _median_latency(trials: list[TaskTrial]) -> float:
        """Return median response latency, ignoring trials without latency."""
        values = [
            float(trial.response_latency_frames)
            for trial in trials
            if trial.response_latency_frames is not None
        ]
        if not values:
            return 0.0
        return float(np.median(values))
