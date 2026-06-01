from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsSpike:
    """Spike scheduled by a recurrent population dynamics model."""

    timestamp: int
    channel: int
    strength: float


class PopulationDynamics:
    """
    Recurrent MEA-level neural dynamics engine.

    This is the first live dynamics layer toward the north-star biological twin.
    It is deliberately population-level rather than cell-level: each electrode
    channel represents a local neural population, and spikes propagate through a
    calibrated connectivity matrix with bounded refractory and gain dynamics.
    That gives closed-loop code a stateful, causal network while keeping the SDK
    simulator fast enough for real-time use.
    """

    def __init__(
        self,
        *,
        channel_count: int,
        connectivity: np.ndarray,
        mode: str = "off",
        coupling: float = 0.35,
        delay_frames: int = 12,
        refractory_frames: int = 20,
        rng: np.random.Generator,
    ):
        self.channel_count = channel_count
        self.mode = mode.lower()
        self.coupling = max(0.0, float(coupling))
        self.delay_frames = max(1, int(delay_frames))
        self.refractory_frames = max(0, int(refractory_frames))
        self.rng = rng
        self._last_spike_ts = np.full(channel_count, -1_000_000_000, dtype=np.int64)
        self._connectivity = self._prepare_connectivity(connectivity)

    @property
    def enabled(self) -> bool:
        """Whether recurrent propagation should be active."""
        return self.mode in {"population", "hawkes", "glm", "recurrent"}

    def on_spike(
        self,
        *,
        timestamp: int,
        channel: int,
        excitability: np.ndarray,
        response_gain: np.ndarray,
    ) -> list[DynamicsSpike]:
        """
        Convert one population spike into delayed downstream spike candidates.

        Positive connectivity creates excitatory propagation.  Negative
        connectivity suppresses downstream probability in this first
        population-level model; a later SNN engine can model inhibitory
        interneurons explicitly.
        """
        if not self.enabled or self.coupling <= 0.0:
            self._last_spike_ts[channel] = timestamp
            return []

        weights = self._connectivity[channel]
        candidates: list[DynamicsSpike] = []
        for target, weight in enumerate(weights):
            if target == channel or weight <= 0.0:
                continue
            if timestamp - self._last_spike_ts[target] < self.refractory_frames:
                continue
            probability = self.coupling * weight * excitability[target] * response_gain[target]
            if self.rng.random() <= min(1.0, max(0.0, probability)):
                delay_jitter = int(self.rng.integers(0, max(1, self.delay_frames // 2 + 1)))
                candidates.append(DynamicsSpike(
                    timestamp = timestamp + self.delay_frames + delay_jitter,
                    channel   = target,
                    strength  = float(weight),
                ))

        self._last_spike_ts[channel] = timestamp
        return candidates

    def decay(self) -> None:
        """
        Advance slow recurrent state.

        The current population model has no continuous synaptic traces yet, but
        this method is part of the stable dynamics contract for future Hawkes,
        GLM, or Izhikevich implementations.
        """

    def _prepare_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """Normalize a profile connectivity matrix into bounded directed weights."""
        matrix = np.asarray(connectivity, dtype=np.float64)
        if matrix.shape != (self.channel_count, self.channel_count):
            matrix = np.eye(self.channel_count, dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.clip(matrix, -1.0, 1.0)
        np.fill_diagonal(matrix, 0.0)
        return np.maximum(matrix, 0.0)
