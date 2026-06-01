from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cl1_snn_reset import CultureConfig, StimEvent, build_network

if TYPE_CHECKING:
    from cl.twin.config import TwinConfig
    from cl.twin.izhikevich import SNNSpike


class ResetSNNAdapter:
    """
    CL producer adapter for the reset-platform culture.

    It accepts the same stimulation/render calls as the existing Izhikevich
    twin engines, but internally routes stimulation through the reset simulator's
    channel-level electrode interface.
    """

    def __init__(
        self,
        *,
        channel_count: int,
        neuron_count: int,
        frames_per_second: int,
        rng: np.random.Generator,
        config: TwinConfig,
    ):
        if channel_count != 64:
            raise ValueError("ResetSNNAdapter currently expects the CL1-style 64-channel MEA.")
        self.channel_count = int(channel_count)
        self.frames_per_second = int(frames_per_second)
        self.rng = rng
        culture = CultureConfig(
            n_neurons                  = max(channel_count, int(neuron_count)),
            excitatory_fraction        = config.snn_excitatory_fraction,
            field_size_mm              = 3.0,
            n_electrodes               = channel_count,
            connection_length_mm       = max(0.03, config.snn_length_constant_um / 1000.0),
            long_range_prob            = 0.02,
            mean_out_degree            = min(config.snn_max_targets_per_source, 64),
            max_out_degree             = max(8, config.snn_max_targets_per_source),
            background_noise_mv        = 1.0,
            spontaneous_rate_hz        = max(0.0, config.baseline_rate_hz),
            stim_gain_mv_per_uA        = 4.8 * config.snn_coupling,
            backend                    = config.snn_reset_backend,
        )
        self.network = build_network(culture, seed=int(config.seed))
        self._pending_stims: list[tuple[int, int, float]] = []

    @property
    def synapse_count(self) -> int:
        return self.network.synapse_count

    def apply_timed_stim(self, timestamp: int, channel: int, drive: np.ndarray) -> None:
        """Queue a timestamped stimulation pulse from the CL producer."""
        drive = np.asarray(drive, dtype=np.float64)
        current = float(np.clip(np.max(np.abs(drive)), 0.05, 8.0))
        self._pending_stims.append((int(timestamp), int(channel), current))

    def apply_stim(self, channel: int, drive: np.ndarray) -> None:
        """Compatibility method for older producer code paths."""
        self.apply_timed_stim(0, channel, drive)

    def render(
        self,
        from_timestamp: int,
        frame_count: int,
        *,
        excitability: np.ndarray,
        response_gain: np.ndarray,
    ) -> list[SNNSpike]:
        from cl.twin.izhikevich import SNNSpike

        duration_ms = frame_count * 1000.0 / self.frames_per_second
        to_timestamp = from_timestamp + frame_count
        due: list[StimEvent] = []
        remaining: list[tuple[int, int, float]] = []
        for timestamp, channel, current in self._pending_stims:
            if timestamp < to_timestamp:
                local_frames = max(0, timestamp - from_timestamp)
                due.append(StimEvent(
                    time_us        = int(round(local_frames * 1_000_000 / self.frames_per_second)),
                    channels       = (channel,),
                    current_uA     = current,
                    pulse_width_us = 160,
                ))
            else:
                remaining.append((timestamp, channel, current))
        self._pending_stims = remaining
        activity = self.network.advance(duration_ms, due, plasticity=True, record=True)
        spikes: list[SNNSpike] = []
        for time_ms, channel in zip(activity.spike_times_ms.tolist(), activity.channels.tolist()):
            frame_offset = int(round(time_ms * self.frames_per_second / 1000.0))
            if 0 <= frame_offset < frame_count:
                gain = float(response_gain[int(channel)] * excitability[int(channel)])
                spikes.append(SNNSpike(
                    timestamp = from_timestamp + frame_offset,
                    channel   = int(channel),
                    strength  = max(0.1, min(2.0, gain)),
                ))
        return spikes
