from __future__ import annotations

import numpy as np


class PlasticityState:
    """
    Lightweight plasticity hook for the surrogate and future SNN engines.

    The north-star twin needs STP, STDP, and homeostatic regulation.  This class
    starts that contract with deterministic per-channel response gain updates so
    closed-loop training can modify future responses without destabilizing SDK
    tests that leave plasticity off.
    """

    def __init__(
        self,
        channel_count: int,
        mode: str = "off",
        dopamine: float = 0.0,
    ):
        self.channel_count = channel_count
        self.mode = mode.lower()
        self.dopamine = max(0.0, float(dopamine))
        self.response_gain = np.ones(channel_count, dtype=np.float64)
        self._last_stim_timestamp = np.full(channel_count, -1_000_000_000, dtype=np.int64)
        self._last_spike_timestamp = np.full(channel_count, -1_000_000_000, dtype=np.int64)

    @property
    def enabled(self) -> bool:
        """Whether this state should change model behavior."""
        return self.mode not in {"", "off", "none", "false", "0"}

    def on_stim(self, timestamp: int, coupling: np.ndarray) -> None:
        """Apply short-term depression/facilitation when an electrode is stimulated."""
        if not self.enabled:
            return
        affected = coupling > 0.05
        self._last_stim_timestamp[affected] = timestamp
        if self.mode in {"stp", "stdp", "stdp_homeostatic", "homeostatic"}:
            # Repeated stimulation temporarily depresses responsiveness near
            # the driven electrode, mimicking vesicle depletion and artifact
            # fatigue in a bounded way.
            self.response_gain[affected] *= 0.98
            self.response_gain[~affected] += (1.0 - self.response_gain[~affected]) * 0.01
            np.clip(self.response_gain, 0.25, 3.0, out=self.response_gain)

    def on_spike(self, timestamp: int, channel: int) -> None:
        """Update STDP-like gain after a spike event."""
        if not self.enabled:
            return
        self._last_spike_timestamp[channel] = timestamp
        if self.mode in {"stdp", "stdp_homeostatic"}:
            delta = timestamp - self._last_stim_timestamp[channel]
            learning_rate = 0.01 * (1.0 + self.dopamine)
            if 0 <= delta <= 250:
                self.response_gain[channel] += learning_rate * np.exp(-delta / 50.0)
            elif -250 <= delta < 0:
                self.response_gain[channel] -= learning_rate * np.exp(delta / 50.0)
            np.clip(self.response_gain, 0.25, 3.0, out=self.response_gain)

    def decay(self) -> None:
        """Slowly return gains toward neutral and apply homeostatic bounds."""
        if not self.enabled:
            return
        self.response_gain += (1.0 - self.response_gain) * 0.002
        if self.mode in {"stdp_homeostatic", "homeostatic"}:
            mean_gain = self.response_gain.mean()
            if mean_gain > 0:
                self.response_gain *= 1.0 / mean_gain
        np.clip(self.response_gain, 0.25, 3.0, out=self.response_gain)
