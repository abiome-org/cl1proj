from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .blocks import (
    ActuatorPositiveControlBlock,
    AntiSTDPPairingBlock,
    ColoredNoiseBlock,
    CoordinatedResetBlock,
    LowFrequencyDepotentiationBlock,
    ProbeTriggeredBlock,
    RestBlock,
    StimBlock,
    StimConstraints,
    StimProgram,
    TaskInputDriveBlock,
)

_DEFAULT_FAMILIES = (
    "anti_stdp_pairing",
    "probe_triggered",
    "coordinated_reset",
    "low_frequency_depotentiation",
    "colored_noise",
    "actuator_positive_control",
    "task_input_drive",
    "rest_control",
)


@dataclass(frozen=True)
class StimSamplingConfig:
    include_blocks: tuple[str, ...] = _DEFAULT_FAMILIES
    amplitude_uA: tuple[float, ...] = (0.8, 1.2, 1.6, 2.0)
    duration_s: tuple[float, ...] = (0.75, 1.5, 3.0, 6.0)
    delays_ms: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    positive_control_frequency_hz: tuple[float, ...] = (80.0, 100.0, 120.0)
    input_drive_frequency_hz: tuple[float, ...] = (80.0, 90.0, 100.0, 110.0, 140.0, 160.0, 200.0)
    input_drive_modes: tuple[str, ...] = ("single_input", "input_neighborhood")

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> StimSamplingConfig:
        payload = dict(values or {})

        def floats(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
            if key not in payload:
                return default
            return tuple(float(v) for v in payload[key])

        def strings(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            if key not in payload:
                return default
            return tuple(str(v) for v in payload[key])

        include = payload.get("include_blocks")
        return cls(
            include_blocks=tuple(str(v) for v in include) if include else _DEFAULT_FAMILIES,
            amplitude_uA=floats("amplitude_uA", cls.amplitude_uA),
            duration_s=floats("duration_s", cls.duration_s),
            delays_ms=floats("delays_ms", cls.delays_ms),
            positive_control_frequency_hz=floats(
                "positive_control_frequency_hz",
                cls.positive_control_frequency_hz,
            ),
            input_drive_frequency_hz=floats("input_drive_frequency_hz", cls.input_drive_frequency_hz),
            input_drive_modes=strings("input_drive_modes", cls.input_drive_modes),
        )


def sample_stim_programs(
    *,
    count: int,
    constraints: StimConstraints,
    input_channel: int,
    target_channel: int,
    rng: np.random.Generator,
    sampling: StimSamplingConfig | None = None,
    include_blocks: tuple[str, ...] | list[str] | None = None,
    amplitude_uA: tuple[float, ...] | list[float] | None = None,
    duration_s: tuple[float, ...] | list[float] | None = None,
    delays_ms: tuple[float, ...] | list[float] | None = None,
    positive_control_frequency_hz: tuple[float, ...] | list[float] | None = None,
    input_drive_frequency_hz: tuple[float, ...] | list[float] | None = None,
    input_drive_modes: tuple[str, ...] | list[str] | None = None,
) -> list[StimProgram]:
    cfg = sampling or StimSamplingConfig()
    families = tuple(include_blocks or cfg.include_blocks)
    amplitude = tuple(amplitude_uA) if amplitude_uA is not None else cfg.amplitude_uA
    durations = tuple(duration_s) if duration_s is not None else cfg.duration_s
    delays = tuple(delays_ms) if delays_ms is not None else cfg.delays_ms
    positive_hz = (
        tuple(positive_control_frequency_hz)
        if positive_control_frequency_hz is not None
        else cfg.positive_control_frequency_hz
    )
    drive_hz = (
        tuple(input_drive_frequency_hz)
        if input_drive_frequency_hz is not None
        else cfg.input_drive_frequency_hz
    )
    drive_modes = tuple(input_drive_modes) if input_drive_modes is not None else cfg.input_drive_modes

    programs: list[StimProgram] = []
    for index in range(int(count)):
        family = str(families[index]) if index < len(families) else str(rng.choice(families))
        seed = int(rng.integers(0, 2**31 - 1))
        amp = _bounded_choice(rng, amplitude, constraints.min_amplitude_uA, constraints.max_amplitude_uA)
        dur = min(float(rng.choice(durations)), constraints.max_total_duration_s)
        pulse_width = int(min(160, constraints.max_pulse_width_us))
        metadata = {"sample_index": index, "program_family": family}
        block = _sample_block(
            family=family,
            constraints=constraints,
            input_channel=input_channel,
            target_channel=target_channel,
            rng=rng,
            amp=amp,
            dur=dur,
            pulse_width=pulse_width,
            delays_ms=delays,
            positive_control_frequency_hz=positive_hz,
            input_drive_frequency_hz=drive_hz,
            input_drive_modes=drive_modes,
        )
        programs.append(
            StimProgram(
                blocks=(block,),
                constraints=constraints,
                metadata=metadata,
                random_seed=seed,
            )
        )
    return programs


def mutate_stim_program(
    program: StimProgram,
    *,
    rng: np.random.Generator,
    input_channel: int,
    target_channel: int,
    scale: float = 0.35,
) -> StimProgram:
    if not program.blocks:
        return program
    block = program.blocks[0]
    constraints = program.constraints
    amp = _clip(
        float(getattr(block, "amplitude_uA", 0.0)) + rng.normal(0.0, scale),
        constraints.min_amplitude_uA,
        constraints.max_amplitude_uA,
    )
    duration = _clip(
        float(getattr(block, "duration_s", 0.1)) * float(np.exp(rng.normal(0.0, scale))),
        0.05,
        constraints.max_total_duration_s,
    )
    if isinstance(block, AntiSTDPPairingBlock):
        delay = _clip(block.delay_ms * float(np.exp(rng.normal(0.0, scale))), 1.0, 120.0)
        ipi = _clip(block.inter_pair_interval_ms * float(np.exp(rng.normal(0.0, scale))), 20.0, 400.0)
        repeat_count = max(1, min(constraints.max_events_per_electrode, int(duration * 1000.0 / ipi)))
        new_block: StimBlock = AntiSTDPPairingBlock(
            first_electrode=target_channel,
            second_electrode=input_channel,
            delay_ms=delay,
            inter_pair_interval_ms=ipi,
            repeat_count=repeat_count,
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=max(0.0, block.jitter_ms + rng.normal(0.0, 0.25)),
            duration_s=duration,
        )
    elif isinstance(block, LowFrequencyDepotentiationBlock):
        new_block = LowFrequencyDepotentiationBlock(
            electrodes=block.electrodes,
            frequency_hz=_clip(block.frequency_hz * float(np.exp(rng.normal(0.0, scale))), 0.2, 12.0),
            duration_s=duration,
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=max(0.0, block.jitter_ms + rng.normal(0.0, 0.25)),
        )
    elif isinstance(block, ColoredNoiseBlock):
        new_block = ColoredNoiseBlock(
            beta=_clip(block.beta + rng.normal(0.0, 0.75), -2.0, 2.0),
            event_rate_hz=_clip(block.event_rate_hz * float(np.exp(rng.normal(0.0, scale))), 2.0, 55.0),
            spatial_mode=block.spatial_mode,
            duration_s=duration,
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
        )
    elif isinstance(block, ActuatorPositiveControlBlock):
        new_block = ActuatorPositiveControlBlock(
            electrodes=block.electrodes,
            frequency_hz=_clip(block.frequency_hz * float(np.exp(rng.normal(0.0, scale))), 20.0, 160.0),
            duration_s=duration,
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=block.jitter_ms,
        )
    elif isinstance(block, TaskInputDriveBlock):
        new_block = TaskInputDriveBlock(
            electrodes=block.electrodes,
            frequency_hz=_clip(block.frequency_hz * float(np.exp(rng.normal(0.0, scale))), 40.0, 240.0),
            duration_s=duration,
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=block.jitter_ms,
            drive_mode=block.drive_mode,
        )
    elif isinstance(block, ProbeTriggeredBlock):
        new_block = ProbeTriggeredBlock(
            probe_electrode=input_channel,
            response_electrode=target_channel,
            response_window_ms=block.response_window_ms,
            anti_pair_delay_ms=_clip(block.anti_pair_delay_ms * float(np.exp(rng.normal(0.0, scale))), 1.0, 120.0),
            max_triggers=max(
                1,
                min(constraints.max_events_per_electrode, int(block.max_triggers + rng.integers(-3, 4))),
            ),
            cooldown_ms=_clip(block.cooldown_ms * float(np.exp(rng.normal(0.0, scale))), 20.0, 250.0),
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=max(0.0, block.jitter_ms + rng.normal(0.0, 0.25)),
            duration_s=duration,
        )
    elif isinstance(block, CoordinatedResetBlock):
        new_block = CoordinatedResetBlock(
            clusters=block.clusters,
            cycle_period_ms=_clip(block.cycle_period_ms * float(np.exp(rng.normal(0.0, scale))), 40.0, 400.0),
            phase_offsets_ms=block.phase_offsets_ms,
            sequence_mode=block.sequence_mode,
            cycles=max(1, min(constraints.max_events_per_electrode, int(block.cycles + rng.integers(-2, 3)))),
            amplitude_uA=amp,
            pulse_width_us=block.pulse_width_us,
            jitter_ms=max(0.0, block.jitter_ms + rng.normal(0.0, 0.5)),
            duration_s=duration,
        )
    else:
        new_block = RestBlock(duration_s=duration)
    metadata = dict(program.metadata)
    metadata["mutated_from"] = metadata.get("protocol_id", "unknown")
    return StimProgram(
        blocks=(new_block,),
        constraints=constraints,
        metadata=metadata,
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _sample_block(
    *,
    family: str,
    constraints: StimConstraints,
    input_channel: int,
    target_channel: int,
    rng: np.random.Generator,
    amp: float,
    dur: float,
    pulse_width: int,
    delays_ms: tuple[float, ...],
    positive_control_frequency_hz: tuple[float, ...],
    input_drive_frequency_hz: tuple[float, ...],
    input_drive_modes: tuple[str, ...],
) -> StimBlock:
    if family in {"anti_stdp_pairing", "anti_stdp"}:
        delay = float(rng.choice(delays_ms))
        ipi = float(rng.choice([40.0, 70.0, 100.0, 160.0, 250.0]))
        repeats = max(1, min(constraints.max_events_per_electrode, int(dur * 1000.0 / ipi)))
        return AntiSTDPPairingBlock(
            first_electrode=int(target_channel),
            second_electrode=int(input_channel),
            delay_ms=delay,
            inter_pair_interval_ms=ipi,
            repeat_count=repeats,
            amplitude_uA=amp,
            pulse_width_us=pulse_width,
            jitter_ms=float(rng.choice([0.0, 0.5, 1.0])),
            duration_s=dur,
        )
    if family == "probe_triggered":
        return ProbeTriggeredBlock(
            probe_electrode=int(input_channel),
            response_electrode=int(target_channel),
            response_window_ms=(4.0, 35.0),
            anti_pair_delay_ms=float(rng.choice(delays_ms)),
            max_triggers=max(1, int(rng.integers(4, 25))),
            cooldown_ms=float(rng.choice([35.0, 70.0, 120.0])),
            amplitude_uA=amp,
            pulse_width_us=pulse_width,
            jitter_ms=float(rng.choice([0.0, 0.5, 1.0])),
            duration_s=dur,
        )
    if family == "coordinated_reset":
        clusters = _sample_clusters(rng, constraints.allowed_electrodes)
        cycle_period = float(rng.choice([80.0, 120.0, 180.0, 250.0]))
        cycles = max(1, int(dur * 1000.0 / cycle_period))
        offsets = tuple(float(i * cycle_period / max(len(clusters), 1)) for i in range(len(clusters)))
        return CoordinatedResetBlock(
            clusters=clusters,
            cycle_period_ms=cycle_period,
            phase_offsets_ms=offsets,
            sequence_mode=str(rng.choice(["fixed", "shuffled"])),
            cycles=min(cycles, constraints.max_events_per_electrode),
            amplitude_uA=amp,
            pulse_width_us=pulse_width,
            jitter_ms=float(rng.choice([0.0, 1.0, 2.0])),
            duration_s=dur,
        )
    if family == "low_frequency_depotentiation":
        electrodes = _task_biased_electrodes(rng, constraints.allowed_electrodes, input_channel, target_channel)
        return LowFrequencyDepotentiationBlock(
            electrodes=electrodes,
            frequency_hz=float(rng.choice([0.5, 1.0, 2.0, 5.0, 8.0])),
            duration_s=dur,
            amplitude_uA=amp,
            pulse_width_us=pulse_width,
            jitter_ms=float(rng.choice([0.0, 0.5, 1.0])),
        )
    if family == "colored_noise":
        return ColoredNoiseBlock(
            beta=float(rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0])),
            event_rate_hz=float(rng.choice([5.0, 10.0, 18.0, 28.0, 40.0])),
            spatial_mode=str(rng.choice(["independent", "clustered", "global"])),
            duration_s=dur,
            amplitude_uA=amp,
            pulse_width_us=pulse_width,
            jitter_ms=0.0,
        )
    if family == "actuator_positive_control":
        return ActuatorPositiveControlBlock(
            electrodes=tuple(int(channel) for channel in constraints.allowed_electrodes),
            frequency_hz=float(rng.choice(positive_control_frequency_hz)),
            duration_s=dur,
            amplitude_uA=amp,
            pulse_width_us=int(min(200, constraints.max_pulse_width_us)),
            jitter_ms=0.0,
        )
    if family == "task_input_drive":
        mode = str(rng.choice(input_drive_modes))
        return TaskInputDriveBlock(
            electrodes=_input_drive_electrodes(
                mode,
                constraints.allowed_electrodes,
                input_channel,
                target_channel,
            ),
            frequency_hz=float(rng.choice(input_drive_frequency_hz)),
            duration_s=dur,
            amplitude_uA=amp,
            pulse_width_us=int(min(200, constraints.max_pulse_width_us)),
            jitter_ms=0.0,
            drive_mode=mode,
        )
    return RestBlock(duration_s=dur)


def _bounded_choice(
    rng: np.random.Generator,
    values: tuple[float, ...],
    low: float,
    high: float,
) -> float:
    valid = [float(value) for value in values if low <= float(value) <= high]
    return float(rng.choice(valid or [max(low, min(high, 0.8))]))


def _sample_clusters(
    rng: np.random.Generator,
    electrodes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(electrodes, dtype=int)
    rng.shuffle(values)
    cluster_count = int(rng.integers(3, 7))
    clusters = np.array_split(values[: min(values.size, cluster_count * 4)], cluster_count)
    return tuple(tuple(int(v) for v in cluster if cluster.size) for cluster in clusters)


def _task_biased_electrodes(
    rng: np.random.Generator,
    electrodes: tuple[int, ...],
    input_channel: int,
    target_channel: int,
) -> tuple[int, ...]:
    allowed = [int(v) for v in electrodes]
    chosen = {int(input_channel), int(target_channel)}
    extras = [value for value in allowed if value not in chosen]
    if extras:
        size = min(len(extras), int(rng.integers(2, 7)))
        chosen.update(int(v) for v in rng.choice(extras, size=size, replace=False))
    return tuple(sorted(chosen))


def _input_drive_electrodes(
    mode: str,
    allowed_electrodes: tuple[int, ...],
    input_channel: int,
    target_channel: int,
) -> tuple[int, ...]:
    allowed = {int(channel) for channel in allowed_electrodes}
    input_channel = int(input_channel)
    target_channel = int(target_channel)
    side = int(round(np.sqrt(max(len(allowed_electrodes), 1))))

    def keep(values: list[int]) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in values
            if int(value) in allowed and int(value) != target_channel
        )

    if mode == "single_input":
        return keep([input_channel])
    row, col = divmod(input_channel, max(side, 1))
    if mode == "input_neighborhood":
        values = []
        for rr in range(max(0, row - 1), min(side, row + 2)):
            for cc in range(max(0, col - 1), min(side, col + 2)):
                values.append(rr * side + cc)
        return keep(values)
    if mode == "input_column":
        return keep([rr * side + col for rr in range(side)])
    if mode == "input_row":
        return keep([row * side + cc for cc in range(side)])
    raise ValueError(f"Unknown task input drive mode: {mode}")


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))
