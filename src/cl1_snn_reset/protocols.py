from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .electrodes import StimEvent
from .noise import generate_colored_events


@dataclass(frozen=True)
class ResetProtocol:
    beta: float
    duration_s: float
    current_uA: float
    pulse_width_us: int
    schedule: str
    spatial_mode: str
    burst_rate_hz: float | None = None
    epoch_s: float | None = None
    pause_s: float | None = None
    protocol_id: str | None = None

    @property
    def id(self) -> str:
        if self.protocol_id:
            return self.protocol_id
        return (
            f"b{self.beta:g}_{self.schedule}_{self.spatial_mode}_"
            f"{self.duration_s:g}s_{self.current_uA:g}uA"
        )

    def total_charge_uC(self, total_pulses: int) -> float:
        return pulse_energy_uC(
            self.current_uA,
            self.pulse_width_us,
            float(max(0, total_pulses)),
        )


def pulse_energy_uC(
    current_uA: float,
    pulse_width_us: float,
    pulse_count: float,
    *,
    channels_per_pulse: float = 1.0,
) -> float:
    return (
        abs(float(current_uA))
        * float(pulse_width_us)
        * 1e-6
        * max(float(pulse_count), 0.0)
        * max(float(channels_per_pulse), 1.0)
        * 2.0
    )


def shift_stim_events(events: list[StimEvent], offset_us: int) -> list[StimEvent]:
    if offset_us == 0:
        return list(events)
    return [
        StimEvent(
            time_us=event.time_us + offset_us,
            channels=event.channels,
            current_uA=event.current_uA,
            pulse_width_us=event.pulse_width_us,
            phases=event.phases,
        )
        for event in events
    ]


def stim_events_energy_uC(events: list[StimEvent]) -> float:
    cost = 0.0
    for event in events:
        cost += (
            abs(float(event.current_uA))
            * float(event.pulse_width_us)
            * 1e-6
            * max(len(event.channels), 1)
            * 2.0
        )
    return float(cost)


def _colored_events(
    protocol: ResetProtocol,
    *,
    n_channels: int,
    rng: np.random.Generator,
    duration_s: float | None = None,
    beta: float | None = None,
) -> list[StimEvent]:
    return generate_colored_events(
        beta           = protocol.beta if beta is None else beta,
        duration_s     = protocol.duration_s if duration_s is None else duration_s,
        n_channels     = n_channels,
        current_uA     = protocol.current_uA,
        pulse_width_us = protocol.pulse_width_us,
        rate_hz        = protocol.burst_rate_hz or _default_rate(protocol.beta if beta is None else beta),
        spatial_mode   = protocol.spatial_mode,
        rng            = rng,
    )


def protocol_events(
    protocol: ResetProtocol,
    *,
    n_channels: int,
    rng: np.random.Generator,
) -> list[StimEvent]:
    """Compile a reset protocol into CL1-style channel pulse events."""
    schedule = protocol.schedule.lower()
    if schedule == "static":
        return _colored_events(protocol, n_channels=n_channels, rng=rng)
    if schedule == "alternating_blue_red":
        half = protocol.duration_s / 2.0
        first = _colored_events(protocol, n_channels=n_channels, rng=rng, duration_s=half, beta=-1.0)
        second = _colored_events(protocol, n_channels=n_channels, rng=rng, duration_s=half, beta=2.0)
        return first + shift_stim_events(second, int(round(half * 1_000_000)))
    if schedule == "epoch_pause":
        epoch_s = protocol.epoch_s or 0.5
        pause_s = protocol.pause_s or 0.2
        events: list[StimEvent] = []
        t_s = 0.0
        while t_s < protocol.duration_s:
            active = min(epoch_s, protocol.duration_s - t_s)
            chunk = _colored_events(protocol, n_channels=n_channels, rng=rng, duration_s=active)
            events.extend(shift_stim_events(chunk, int(round(t_s * 1_000_000))))
            t_s += active + pause_s
        return events
    if schedule == "ramp":
        events = _colored_events(protocol, n_channels=n_channels, rng=rng)
        ramped = []
        for event in events:
            fraction = min(1.0, max(0.1, event.time_us / max(protocol.duration_s * 1_000_000, 1.0)))
            ramped.append(StimEvent(
                time_us        = event.time_us,
                channels       = event.channels,
                current_uA     = event.current_uA * fraction,
                pulse_width_us = event.pulse_width_us,
                phases         = event.phases,
            ))
        return ramped
    raise ValueError(f"Unknown reset schedule: {protocol.schedule}")


def coarse_protocol_grid() -> list[ResetProtocol]:
    """Initial coarse grid for reset protocol screening."""
    beta_values = [-2, -1, 0, 1, 2]
    spatial_modes = ["shared", "independent", "correlated", "phase_shifted"]
    schedules = ["static", "alternating_blue_red", "epoch_pause"]
    durations = [0.75, 1.5, 3.0]
    currents = [0.8, 1.6, 2.6]
    protocols: list[ResetProtocol] = []
    for beta, spatial_mode, schedule, duration, current in product(
        beta_values,
        spatial_modes,
        schedules,
        durations,
        currents,
    ):
        protocols.append(ResetProtocol(
            beta=beta,
            duration_s=duration,
            current_uA=current,
            pulse_width_us=160,
            schedule=schedule,
            spatial_mode=spatial_mode,
        ))
    return protocols


def _default_rate(beta: float) -> float:
    if beta >= 1.5:
        return 38.0
    if beta >= 0.5:
        return 48.0
    if beta <= -1.5:
        return 85.0
    if beta <= -0.5:
        return 70.0
    return 58.0
