from __future__ import annotations

import numpy as np


class PinkNoiseState:
    """
    Stateful MEA noise source with a lightweight 1/f-like spectrum.

    Thermal and biological MEA noise is not independent from sample to sample:
    electrode drift, amplifier baselines, and local field fluctuations add slow
    components on top of white noise.  A full spectral simulator would be heavy
    for the SDK subprocess, so this generator sums a few leaky components with
    different time constants.  The result has positive short-lag correlation and
    more low-frequency power while preserving the calibrated per-channel standard
    deviation supplied by `TwinProfile`.
    """

    def __init__(
        self,
        *,
        channel_count: int,
        rng: np.random.Generator,
        std_by_channel: np.ndarray,
        color: str = "pink",
    ):
        self.channel_count = channel_count
        self.rng = rng
        values = np.asarray(std_by_channel, dtype=np.float64)
        if values.shape != (channel_count,):
            values = np.ones(channel_count, dtype=np.float64)
        self.std_by_channel = np.maximum(values, 1e-6)
        self.color = color.lower()
        self._state = np.zeros((4, channel_count), dtype=np.float64)
        self._decay = np.array([0.55, 0.82, 0.94, 0.985], dtype=np.float64)
        self._mix = np.array([0.45, 0.30, 0.18, 0.07], dtype=np.float64)
        self._unit_scale = float(np.sqrt(0.55 * 0.55 + np.sum(self._mix * self._mix)))

    def sample(self, frame_count: int) -> np.ndarray:
        """Return noise frames with shape `(frame_count, channel_count)`."""
        if self.color in {"white", "gaussian"}:
            return self.rng.normal(
                loc=0.0,
                scale=self.std_by_channel,
                size=(frame_count, self.channel_count),
            )
        if frame_count <= 0:
            return np.zeros((0, self.channel_count), dtype=np.float64)

        output = np.zeros((frame_count, self.channel_count), dtype=np.float64)
        for frame_index in range(frame_count):
            innovations = self.rng.normal(0.0, 1.0, size=self._state.shape)
            drive_scale = np.sqrt(np.maximum(1.0 - self._decay * self._decay, 1e-9))
            self._state = self._decay[:, None] * self._state + drive_scale[:, None] * innovations
            white = self.rng.normal(0.0, 1.0, size=self.channel_count)
            output[frame_index] = 0.55 * white + (self._mix[:, None] * self._state).sum(axis=0)

        return output * (self.std_by_channel / max(self._unit_scale, 1e-6))
