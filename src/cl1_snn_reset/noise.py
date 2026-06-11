"""Colored-noise generation and spatial channel-sampling strategies for reset stimulation."""
from __future__ import annotations

import numpy as np

from .electrodes import StimEvent


def colored_noise(beta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return zero-mean unit-variance noise with approximate 1/f**beta power.

    beta=-2 is violet-like, -1 blue, 0 white, 1 pink, and 2 red/brown.
    """
    n = max(2, int(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    phases = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    scale = np.power(freqs, -float(beta) / 2.0)
    spectrum = phases * scale
    signal = np.fft.irfft(spectrum, n=n)
    signal -= float(signal.mean())
    std = float(signal.std())
    return signal / (std if std > 1e-12 else 1.0)


def generate_colored_events(
    *,
    beta: float,
    duration_s: float,
    n_channels: int,
    current_uA: float,
    pulse_width_us: int,
    rate_hz: float,
    spatial_mode: str,
    rng: np.random.Generator,
    dt_ms: float = 1.0,
) -> list[StimEvent]:
    """
    Translate a spectral color into channel-level pulse-event statistics.

    Temporal color controls the event intensity envelope. Spatial color controls
    which channels co-activate within each event.
    """
    bins = max(1, int(round(duration_s * 1000.0 / dt_ms)))
    envelope = colored_noise(beta, bins, rng)
    envelope = np.exp(0.7 * envelope)
    envelope /= max(float(envelope.mean()), 1e-12)
    p = np.clip(rate_hz * (dt_ms / 1000.0) * envelope, 0.0, 0.95)
    event_bins = np.flatnonzero(rng.random(bins) < p)
    events: list[StimEvent] = []
    side = int(round(np.sqrt(n_channels)))
    for ordinal, bin_index in enumerate(event_bins.tolist()):
        channels = _sample_channels(
            spatial_mode = spatial_mode,
            n_channels   = n_channels,
            side         = side,
            ordinal      = ordinal,
            rng          = rng,
        )
        events.append(StimEvent(
            time_us        = int(round(bin_index * dt_ms * 1000.0)),
            channels       = channels,
            current_uA     = current_uA,
            pulse_width_us = pulse_width_us,
        ))
    return events


def _sample_channels(
    *,
    spatial_mode: str,
    n_channels: int,
    side: int,
    ordinal: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    mode = spatial_mode.lower()
    if mode == "shared":
        size = int(rng.integers(4, 9))
        return tuple(sorted(rng.choice(n_channels, size=size, replace=False).astype(int).tolist()))
    if mode == "correlated":
        center = int(rng.integers(0, n_channels))
        row, col = divmod(center, side)
        candidates = []
        for rr in range(max(0, row - 1), min(side, row + 2)):
            for cc in range(max(0, col - 1), min(side, col + 2)):
                candidates.append(rr * side + cc)
        size = min(len(candidates), int(rng.integers(2, 6)))
        return tuple(sorted(rng.choice(candidates, size=size, replace=False).astype(int).tolist()))
    if mode == "phase_shifted":
        width = max(1, side // 2)
        start = ordinal % n_channels
        return tuple(int((start + k) % n_channels) for k in range(width))
    # independent
    return (int(rng.integers(0, n_channels)),)
