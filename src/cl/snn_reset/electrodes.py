from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import CultureConfig


@dataclass(frozen=True)
class StimEvent:
    """Charge-balanced channel-level stimulation event."""

    time_us: int
    channels: tuple[int, ...]
    current_uA: float
    pulse_width_us: int = 160
    phases: tuple[float, ...] = (-1.0, 1.0)


@dataclass(frozen=True)
class ChannelActivity:
    """Channel-level multi-unit readout."""

    spike_times_ms: np.ndarray
    channels: np.ndarray
    counts: np.ndarray
    duration_ms: float
    total_neuron_spikes: int

    def binned_counts(self, bin_ms: float = 10.0, channel_count: int = 64) -> np.ndarray:
        bins = max(1, int(np.ceil(self.duration_ms / bin_ms)))
        result = np.zeros((bins, channel_count), dtype=np.float64)
        if self.spike_times_ms.size == 0:
            return result
        bin_index = np.clip((self.spike_times_ms / bin_ms).astype(int), 0, bins - 1)
        np.add.at(result, (bin_index, self.channels), self.counts)
        return result


class ElectrodeArray:
    """
    MEA interface converting channel pulses into cell drive and hidden spikes
    into channel-level multi-unit events.
    """

    def __init__(
        self,
        electrode_xy: np.ndarray,
        neuron_xy: np.ndarray,
        *,
        electrode_radius_mm: float,
        stim_gamma: float = 1.35,
        record_gamma: float = 1.7,
    ):
        self.electrode_xy = np.asarray(electrode_xy, dtype=np.float64)
        self.neuron_xy = np.asarray(neuron_xy, dtype=np.float64)
        self.electrode_radius_mm = float(electrode_radius_mm)
        self.channel_count = int(self.electrode_xy.shape[0])

        deltas = self.electrode_xy[:, None, :] - self.neuron_xy[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        safe = np.maximum(distances, self.electrode_radius_mm)
        stim = 1.0 / np.power(safe, stim_gamma)
        stim /= np.maximum(stim.max(axis=1, keepdims=True), 1e-12)
        record = 1.0 / np.power(safe, record_gamma)
        record /= np.maximum(record.sum(axis=0, keepdims=True), 1e-12)
        self.stim_kernel = stim.astype(np.float64)
        self.record_kernel = record.astype(np.float64)
        self.nearest_channel = np.argmax(self.record_kernel, axis=0).astype(np.int64)

    @classmethod
    def from_config(cls, cfg: CultureConfig, neuron_xy: np.ndarray) -> "ElectrodeArray":
        side = int(round(np.sqrt(cfg.n_electrodes)))
        if side * side != cfg.n_electrodes:
            raise ValueError("Only square electrode grids are currently supported.")
        axis = np.linspace(0.0, cfg.field_size_mm, side)
        xx, yy = np.meshgrid(axis, axis)
        electrode_xy = np.column_stack([xx.ravel(), yy.ravel()])
        return cls(
            electrode_xy,
            neuron_xy,
            electrode_radius_mm = cfg.electrode_radius_mm,
            stim_gamma          = cfg.stim_kernel_gamma,
            record_gamma        = cfg.record_kernel_gamma,
        )

    def stimulate(self, event: StimEvent) -> np.ndarray:
        """Convert one channel pulse event into per-neuron drive."""
        drive = np.zeros(self.neuron_xy.shape[0], dtype=np.float64)
        phase_balance = float(np.sum(np.abs(event.phases))) / max(len(event.phases), 1)
        width_scale = max(0.1, float(event.pulse_width_us) / 160.0)
        for channel in event.channels:
            drive += self.stim_kernel[int(channel)]
        return drive * abs(float(event.current_uA)) * phase_balance * width_scale

    def record(
        self,
        spike_times_ms: np.ndarray,
        neuron_indices: np.ndarray,
        *,
        duration_ms: float,
    ) -> ChannelActivity:
        """Project hidden neuron spikes into channel-level multi-unit readout."""
        if len(neuron_indices) == 0:
            return ChannelActivity(
                spike_times_ms       = np.array([], dtype=np.float64),
                channels             = np.array([], dtype=np.int64),
                counts               = np.array([], dtype=np.int64),
                duration_ms          = float(duration_ms),
                total_neuron_spikes  = 0,
            )
        channels = self.nearest_channel[np.asarray(neuron_indices, dtype=np.int64)]
        return ChannelActivity(
            spike_times_ms      = np.asarray(spike_times_ms, dtype=np.float64),
            channels            = channels.astype(np.int64),
            counts              = np.ones(len(channels), dtype=np.int64),
            duration_ms         = float(duration_ms),
            total_neuron_spikes = int(len(channels)),
        )

    def channels_to_neuron_mask(self, channels: Iterable[int]) -> np.ndarray:
        selected = set(int(channel) for channel in channels)
        return np.fromiter(
            (channel in selected for channel in self.nearest_channel.tolist()),
            dtype=np.bool_,
            count=len(self.nearest_channel),
        )
