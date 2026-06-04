from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cl1_snn_reset.electrodes import StimEvent
from cl1_snn_reset.protocols import protocol_events, shift_stim_events, stim_events_energy_uC

from .blocks import (
    AntiSTDPPairingBlock,
    ActuatorPositiveControlBlock,
    ColoredNoiseBlock,
    CoordinatedResetBlock,
    LowFrequencyDepotentiationBlock,
    ProbeTriggeredBlock,
    RestBlock,
    StimBlock,
    StimProgram,
    TaskInputDriveBlock,
)


@dataclass(frozen=True)
class InvalidStimProgramError(ValueError):
    reason: str
    violating_fields: tuple[str, ...] = ()

    def __str__(self) -> str:
        fields = ", ".join(self.violating_fields)
        suffix = f" ({fields})" if fields else ""
        return f"{self.reason}{suffix}"


def compile_program_to_stim_events(program: StimProgram) -> list[StimEvent]:
    _validate_blocks(program)
    events: list[StimEvent] = []
    offset_us = 0
    for index, block in enumerate(program.blocks):
        rng = np.random.default_rng(int(program.random_seed) + 1_003 * index)
        block_events = _compile_block(
            block,
            n_channels=len(program.constraints.allowed_electrodes),
            rng=rng,
        )
        events.extend(shift_stim_events(block_events, offset_us))
        offset_us += int(round(float(getattr(block, "duration_s", 0.0)) * 1_000_000.0))
    events = sorted(events, key=lambda event: event.time_us)
    validate_compiled_events(program, events)
    return events


def validate_stim_program(program: StimProgram) -> dict[str, Any]:
    events = compile_program_to_stim_events(program)
    counts = _events_per_electrode(events)
    return {
        "valid": True,
        "event_count": len(events),
        "pulses": int(sum(len(event.channels) for event in events)),
        "energy_cost": estimate_energy_cost(program, events),
        "max_events_per_electrode": int(max(counts.values(), default=0)),
        "total_duration_s": program.total_duration_s,
    }


def estimate_energy_cost(
    program: StimProgram,
    events: list[StimEvent] | None = None,
) -> float:
    if events is None:
        try:
            events = compile_program_to_stim_events(program)
        except InvalidStimProgramError:
            return float("inf")
    return stim_events_energy_uC(events)


def validate_compiled_events(program: StimProgram, events: list[StimEvent]) -> None:
    constraints = program.constraints
    if program.total_duration_s > constraints.max_total_duration_s + 1e-9:
        raise InvalidStimProgramError("Program exceeds maximum duration.", ("total_duration_s",))
    allowed = {int(channel) for channel in constraints.allowed_electrodes}
    last_by_channel: dict[int, float] = {}
    counts: dict[int, int] = {}
    for event in events:
        if event.current_uA < constraints.min_amplitude_uA - 1e-9:
            raise InvalidStimProgramError("Pulse below minimum amplitude.", ("current_uA",))
        if event.current_uA > constraints.max_amplitude_uA + 1e-9:
            raise InvalidStimProgramError("Pulse exceeds maximum amplitude.", ("current_uA",))
        if event.pulse_width_us > constraints.max_pulse_width_us + 1e-9:
            raise InvalidStimProgramError("Pulse width exceeds maximum.", ("pulse_width_us",))
        if constraints.require_charge_balanced and abs(sum(event.phases)) > 1e-9:
            raise InvalidStimProgramError("Pulse is not charge balanced.", ("phases",))
        for channel in event.channels:
            channel = int(channel)
            if channel not in allowed:
                raise InvalidStimProgramError("Pulse uses disallowed electrode.", ("channels",))
            now_ms = float(event.time_us) / 1000.0
            previous = last_by_channel.get(channel)
            if previous is not None:
                cooldown = now_ms - previous
                if cooldown + 1e-9 < constraints.min_cooldown_ms_per_electrode:
                    raise InvalidStimProgramError(
                        "Per-electrode cooldown violated.",
                        ("min_cooldown_ms_per_electrode",),
                    )
            last_by_channel[channel] = now_ms
            counts[channel] = counts.get(channel, 0) + 1
            if counts[channel] > constraints.max_events_per_electrode:
                raise InvalidStimProgramError(
                    "Per-electrode event limit exceeded.",
                    ("max_events_per_electrode",),
                )
    cost = estimate_energy_cost(program, events)
    if cost > constraints.max_energy_cost + 1e-12:
        raise InvalidStimProgramError("Energy budget exceeded.", ("max_energy_cost",))


def _validate_blocks(program: StimProgram) -> None:
    if not program.blocks:
        raise InvalidStimProgramError("Program has no stimulation blocks.", ("blocks",))
    for block in program.blocks:
        duration = float(getattr(block, "duration_s", 0.0))
        if duration < 0.0:
            raise InvalidStimProgramError("Block duration must be non-negative.", ("duration_s",))
        if isinstance(block, RestBlock):
            continue
        amplitude = float(getattr(block, "amplitude_uA", 0.0))
        if amplitude < program.constraints.min_amplitude_uA - 1e-9:
            raise InvalidStimProgramError("Block below minimum amplitude.", ("amplitude_uA",))
        if amplitude > program.constraints.max_amplitude_uA + 1e-9:
            raise InvalidStimProgramError("Block exceeds maximum amplitude.", ("amplitude_uA",))
        if float(getattr(block, "pulse_width_us", 0.0)) > program.constraints.max_pulse_width_us + 1e-9:
            raise InvalidStimProgramError("Block exceeds pulse width limit.", ("pulse_width_us",))


def _compile_block(
    block: StimBlock,
    *,
    n_channels: int,
    rng: np.random.Generator,
) -> list[StimEvent]:
    if isinstance(block, AntiSTDPPairingBlock):
        events: list[StimEvent] = []
        for index in range(block.repeat_count):
            base_ms = index * block.inter_pair_interval_ms
            jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
            first_ms = max(0.0, base_ms + jitter)
            second_ms = first_ms + block.delay_ms
            if second_ms >= block.duration_s * 1000.0:
                break
            events.append(_event(first_ms, (block.first_electrode,), block.amplitude_uA, block.pulse_width_us))
            events.append(_event(second_ms, (block.second_electrode,), block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, ProbeTriggeredBlock):
        events = []
        interval = max(block.cooldown_ms, 1.0)
        for index in range(block.max_triggers):
            probe_ms = index * interval
            jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
            probe_ms = max(0.0, probe_ms + jitter)
            response_mid = float(np.mean(block.response_window_ms))
            first_ms = probe_ms + response_mid
            second_ms = first_ms + block.anti_pair_delay_ms
            if second_ms >= block.duration_s * 1000.0:
                break
            events.append(_event(probe_ms, (block.probe_electrode,), block.amplitude_uA, block.pulse_width_us))
            events.append(_event(first_ms, (block.response_electrode,), block.amplitude_uA, block.pulse_width_us))
            events.append(_event(second_ms, (block.probe_electrode,), block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, CoordinatedResetBlock):
        events = []
        for cycle in range(block.cycles):
            cluster_order = list(range(len(block.clusters)))
            if block.sequence_mode == "shuffled":
                rng.shuffle(cluster_order)
            for ordinal in cluster_order:
                cluster = tuple(int(channel) for channel in block.clusters[ordinal])
                phase = block.phase_offsets_ms[min(ordinal, len(block.phase_offsets_ms) - 1)]
                jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
                time_ms = max(0.0, cycle * block.cycle_period_ms + phase + jitter)
                if time_ms >= block.duration_s * 1000.0:
                    continue
                events.append(_event(time_ms, cluster, block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, LowFrequencyDepotentiationBlock):
        events = []
        period_ms = 1000.0 / max(block.frequency_hz, 1e-9)
        steps = int(np.floor(block.duration_s * 1000.0 / period_ms)) + 1
        for step in range(steps):
            time_ms = step * period_ms
            jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
            time_ms = max(0.0, time_ms + jitter)
            if time_ms >= block.duration_s * 1000.0:
                continue
            for electrode in block.electrodes:
                events.append(_event(time_ms, (int(electrode),), block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, ColoredNoiseBlock):
        return protocol_events(block.to_reset_protocol(), n_channels=n_channels, rng=rng)
    if isinstance(block, ActuatorPositiveControlBlock):
        events = []
        period_ms = 1000.0 / max(block.frequency_hz, 1e-9)
        steps = int(np.floor(block.duration_s * 1000.0 / period_ms)) + 1
        for step in range(steps):
            time_ms = step * period_ms
            jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
            time_ms = max(0.0, time_ms + jitter)
            if time_ms >= block.duration_s * 1000.0:
                continue
            events.append(_event(time_ms, block.electrodes, block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, TaskInputDriveBlock):
        events = []
        period_ms = 1000.0 / max(block.frequency_hz, 1e-9)
        steps = int(np.floor(block.duration_s * 1000.0 / period_ms)) + 1
        for step in range(steps):
            time_ms = step * period_ms
            jitter = float(rng.normal(0.0, block.jitter_ms)) if block.jitter_ms > 0.0 else 0.0
            time_ms = max(0.0, time_ms + jitter)
            if time_ms >= block.duration_s * 1000.0:
                continue
            events.append(_event(time_ms, block.electrodes, block.amplitude_uA, block.pulse_width_us))
        return events
    if isinstance(block, RestBlock):
        return []
    raise InvalidStimProgramError("Unsupported stimulation block.", ("block_type",))


def _event(time_ms: float, channels: tuple[int, ...], current_uA: float, pulse_width_us: int) -> StimEvent:
    return StimEvent(
        time_us=int(round(float(time_ms) * 1000.0)),
        channels=tuple(int(channel) for channel in channels),
        current_uA=float(current_uA),
        pulse_width_us=int(pulse_width_us),
    )


def _events_per_electrode(events: list[StimEvent]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in events:
        for channel in event.channels:
            counts[int(channel)] = counts.get(int(channel), 0) + 1
    return counts
