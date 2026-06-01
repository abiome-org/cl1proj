from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .feedback import TwinFeedbackProtocol
from .learning import LearningCurveReport, TaskTrial, TwinLearningEvaluator
from .surrogate import SurrogateTwinModel
from .._data_buffer import StimRecord


@dataclass(frozen=True)
class TwinTrainingResult:
    """Summary returned by an accelerated closed-loop twin training run."""

    simulated_frames: int
    trial_count: int
    spike_count: int
    stim_count: int
    learning_report: LearningCurveReport

    def to_dict(self) -> dict:
        """Return a JSON-serializable training summary."""
        data = asdict(self)
        data["learning_report"] = self.learning_report.to_dict()
        return data


class TwinAcceleratedTrainer:
    """
    Run closed-loop twin training as fast as the model can advance.

    The producer's `CL_SDK_ACCELERATED_TIME` mode accelerates normal SDK reads.
    This helper is the experiment-facing counterpart: it applies structured or
    chaotic feedback directly to a `SurrogateTwinModel`, renders biological time
    in chunks, and returns a learning-curve report. It is intentionally small
    and deterministic so long-horizon notebooks can use the same twin dynamics
    without needing a subprocess or wall-clock sleeps.
    """

    def __init__(
        self,
        *,
        model: SurrogateTwinModel,
        protocol: TwinFeedbackProtocol,
        render_chunk_frames: int = 250,
    ):
        self.model = model
        self.protocol = protocol
        self.render_chunk_frames = max(1, int(render_chunk_frames))

    def run_trials(
        self,
        outcomes: Iterable[bool],
        *,
        start_timestamp: int = 0,
        trial_interval_frames: int = 2_500,
        sensory_channel: int = 0,
        motor_channel: int | None = None,
        current_uA: float = 1.0,
        response_latency_fn: Callable[[int, bool], int | None] | None = None,
    ) -> TwinTrainingResult:
        """
        Apply outcome-coded feedback and render between trials without sleeping.

        `outcomes` is deliberately generic: task/game code decides whether a
        trial was correct, while this trainer maps correct trials to structured
        feedback and incorrect trials to chaotic feedback using
        `TwinFeedbackProtocol`.
        """
        timestamp = int(start_timestamp)
        interval = max(1, int(trial_interval_frames))
        trials: list[TaskTrial] = []
        spike_count = 0
        stim_count = 0
        simulated_frames = 0

        for index, correct in enumerate(outcomes):
            feedback = self.protocol.from_outcome(
                timestamp       = timestamp,
                correct         = bool(correct),
                sensory_channel = sensory_channel,
                motor_channel   = motor_channel,
                current_uA      = current_uA,
            )
            for event in feedback:
                self.model.apply_stim(
                    StimRecord(timestamp=event.timestamp, channel=event.channel),
                    current_uA=event.current_uA,
                )
            stim_count += len(feedback)

            rendered, spikes = self._render_until(timestamp, timestamp + interval)
            simulated_frames += rendered
            spike_count += len(spikes)
            trials.append(TaskTrial(
                timestamp               = timestamp,
                correct                 = bool(correct),
                response_latency_frames = (
                    response_latency_fn(index, bool(correct))
                    if response_latency_fn is not None
                    else None
                ),
            ))
            timestamp += interval

        report = TwinLearningEvaluator.evaluate_trials(trials)
        return TwinTrainingResult(
            simulated_frames = simulated_frames,
            trial_count      = len(trials),
            spike_count      = spike_count,
            stim_count       = stim_count,
            learning_report  = report,
        )

    def _render_until(self, from_timestamp: int, to_timestamp: int) -> tuple[int, list]:
        """Render `[from_timestamp, to_timestamp)` in bounded chunks."""
        timestamp = int(from_timestamp)
        all_spikes = []
        while timestamp < to_timestamp:
            frame_count = min(self.render_chunk_frames, to_timestamp - timestamp)
            _, spikes = self.model.render(timestamp, frame_count)
            all_spikes.extend(spikes)
            timestamp += frame_count
        return to_timestamp - from_timestamp, all_spikes
