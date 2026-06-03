from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class StimConstraints:
    max_amplitude_uA: float = 2.0
    min_amplitude_uA: float = 0.0
    max_pulse_width_us: float = 200.0
    min_inter_pulse_interval_ms: float = 2.0
    max_total_duration_s: float = 6.0
    max_energy_cost: float = 0.1
    max_events_per_electrode: int = 500
    min_cooldown_ms_per_electrode: float = 2.0
    allowed_electrodes: tuple[int, ...] = tuple(range(64))
    require_charge_balanced: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "StimConstraints":
        payload = dict(values or {})
        if "allowed_electrodes" in payload:
            payload["allowed_electrodes"] = tuple(int(v) for v in payload["allowed_electrodes"])
        return cls(**payload)

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"allowed_electrodes": list(self.allowed_electrodes)}


@dataclass(frozen=True)
class AntiSTDPPairingBlock:
    first_electrode: int
    second_electrode: int
    delay_ms: float
    inter_pair_interval_ms: float
    repeat_count: int
    amplitude_uA: float
    pulse_width_us: int = 160
    jitter_ms: float = 0.0
    duration_s: float = 1.0
    block_type: str = field(default="anti_stdp_pairing", init=False)


@dataclass(frozen=True)
class ProbeTriggeredBlock:
    probe_electrode: int
    response_electrode: int
    response_window_ms: tuple[float, float]
    anti_pair_delay_ms: float
    max_triggers: int
    cooldown_ms: float
    amplitude_uA: float
    pulse_width_us: int = 160
    jitter_ms: float = 0.0
    duration_s: float = 1.0
    block_type: str = field(default="probe_triggered", init=False)


@dataclass(frozen=True)
class CoordinatedResetBlock:
    clusters: tuple[tuple[int, ...], ...]
    cycle_period_ms: float
    phase_offsets_ms: tuple[float, ...]
    sequence_mode: Literal["fixed", "shuffled", "adaptive"]
    cycles: int
    amplitude_uA: float
    pulse_width_us: int = 160
    jitter_ms: float = 0.0
    duration_s: float = 1.0
    block_type: str = field(default="coordinated_reset", init=False)


@dataclass(frozen=True)
class LowFrequencyDepotentiationBlock:
    electrodes: tuple[int, ...]
    frequency_hz: float
    duration_s: float
    amplitude_uA: float
    pulse_width_us: int = 160
    jitter_ms: float = 0.0
    block_type: str = field(default="low_frequency_depotentiation", init=False)


@dataclass(frozen=True)
class ColoredNoiseBlock:
    beta: float
    event_rate_hz: float
    spatial_mode: Literal["independent", "clustered", "global"]
    duration_s: float
    amplitude_uA: float
    pulse_width_us: int = 160
    jitter_ms: float = 0.0
    block_type: str = field(default="colored_noise", init=False)


@dataclass(frozen=True)
class ActuatorPositiveControlBlock:
    electrodes: tuple[int, ...]
    frequency_hz: float
    duration_s: float
    amplitude_uA: float
    pulse_width_us: int = 200
    jitter_ms: float = 0.0
    block_type: str = field(default="actuator_positive_control", init=False)


@dataclass(frozen=True)
class TaskInputDriveBlock:
    electrodes: tuple[int, ...]
    frequency_hz: float
    duration_s: float
    amplitude_uA: float
    pulse_width_us: int = 200
    jitter_ms: float = 0.0
    drive_mode: str = "single_input"
    block_type: str = field(default="task_input_drive", init=False)


@dataclass(frozen=True)
class RestBlock:
    duration_s: float
    block_type: str = field(default="rest", init=False)


StimBlock = (
    AntiSTDPPairingBlock
    | ProbeTriggeredBlock
    | CoordinatedResetBlock
    | LowFrequencyDepotentiationBlock
    | ColoredNoiseBlock
    | ActuatorPositiveControlBlock
    | TaskInputDriveBlock
    | RestBlock
)


@dataclass(frozen=True)
class StimProgram:
    blocks: tuple[StimBlock, ...]
    constraints: StimConstraints = field(default_factory=StimConstraints)
    metadata: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 0

    @property
    def total_duration_s(self) -> float:
        return float(sum(float(getattr(block, "duration_s", 0.0)) for block in self.blocks))

    @property
    def family(self) -> str:
        if not self.blocks:
            return "empty"
        return str(getattr(self.blocks[0], "block_type", "unknown"))

    def to_json(self) -> dict[str, Any]:
        return {
            "blocks": [block_to_json(block) for block in self.blocks],
            "constraints": self.constraints.to_json(),
            "metadata": dict(self.metadata),
            "random_seed": int(self.random_seed),
            "total_duration_s": self.total_duration_s,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "StimProgram":
        return cls(
            blocks=tuple(block_from_json(item) for item in payload.get("blocks", [])),
            constraints=StimConstraints.from_dict(payload.get("constraints", {})),
            metadata=dict(payload.get("metadata", {})),
            random_seed=int(payload.get("random_seed", 0)),
        )


STIM_FEATURE_NAMES = (
    "family:anti_stdp_pairing",
    "family:probe_triggered",
    "family:coordinated_reset",
    "family:low_frequency_depotentiation",
    "family:colored_noise",
    "family:actuator_positive_control",
    "family:task_input_drive",
    "family:rest",
    "total_duration_s",
    "block_count",
    "rough_event_count",
    "rough_energy_uC",
    "mean_amplitude_uA",
    "max_amplitude_uA",
    "pulse_width_us",
    "anti_pair_delay_ms",
    "low_frequency_hz",
    "colored_beta",
    "colored_event_rate_hz",
    "electrode_coverage",
    "electrode_entropy",
    "rest_fraction",
)


def block_to_json(block: StimBlock) -> dict[str, Any]:
    payload = asdict(block)
    payload["block_type"] = getattr(block, "block_type")
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = _tuple_to_list(value)
    return payload


def block_from_json(payload: dict[str, Any]) -> StimBlock:
    block_type = str(payload.get("block_type", "")).lower()
    values = {key: value for key, value in payload.items() if key != "block_type"}
    if block_type == "anti_stdp_pairing":
        return AntiSTDPPairingBlock(**values)
    if block_type == "probe_triggered":
        values["response_window_ms"] = tuple(values["response_window_ms"])
        return ProbeTriggeredBlock(**values)
    if block_type == "coordinated_reset":
        values["clusters"] = tuple(tuple(cluster) for cluster in values["clusters"])
        values["phase_offsets_ms"] = tuple(values["phase_offsets_ms"])
        return CoordinatedResetBlock(**values)
    if block_type == "low_frequency_depotentiation":
        values["electrodes"] = tuple(values["electrodes"])
        return LowFrequencyDepotentiationBlock(**values)
    if block_type == "colored_noise":
        return ColoredNoiseBlock(**values)
    if block_type == "actuator_positive_control":
        values["electrodes"] = tuple(values["electrodes"])
        return ActuatorPositiveControlBlock(**values)
    if block_type == "task_input_drive":
        values["electrodes"] = tuple(values["electrodes"])
        return TaskInputDriveBlock(**values)
    if block_type == "rest":
        return RestBlock(**values)
    raise ValueError(f"Unknown stimulation block type: {block_type}")


def sample_stim_programs(
    *,
    count: int,
    constraints: StimConstraints,
    input_channel: int,
    target_channel: int,
    rng: np.random.Generator,
    include_blocks: tuple[str, ...] | list[str] | None = None,
    amplitude_uA: tuple[float, ...] | list[float] = (0.8, 1.2, 1.6, 2.0),
    duration_s: tuple[float, ...] | list[float] = (0.75, 1.5, 3.0, 6.0),
    delays_ms: tuple[float, ...] | list[float] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0),
    positive_control_frequency_hz: tuple[float, ...] | list[float] = (80.0, 100.0, 120.0),
    input_drive_frequency_hz: tuple[float, ...] | list[float] = (80.0, 90.0, 100.0, 110.0, 140.0, 160.0, 200.0),
    input_drive_modes: tuple[str, ...] | list[str] = ("single_input", "input_neighborhood"),
) -> list[StimProgram]:
    families = tuple(include_blocks or (
        "anti_stdp_pairing",
        "probe_triggered",
        "coordinated_reset",
        "low_frequency_depotentiation",
        "colored_noise",
        "actuator_positive_control",
        "task_input_drive",
        "rest_control",
    ))
    programs: list[StimProgram] = []
    for index in range(int(count)):
        family = str(families[index]) if index < len(families) else str(rng.choice(families))
        seed = int(rng.integers(0, 2**31 - 1))
        amp = _bounded_choice(rng, amplitude_uA, constraints.min_amplitude_uA, constraints.max_amplitude_uA)
        dur = float(rng.choice(duration_s))
        dur = min(dur, constraints.max_total_duration_s)
        pulse_width = int(min(160, constraints.max_pulse_width_us))
        metadata = {"sample_index": index, "program_family": family}
        if family in {"anti_stdp_pairing", "anti_stdp"}:
            delay = float(rng.choice(delays_ms))
            ipi = float(rng.choice([40.0, 70.0, 100.0, 160.0, 250.0]))
            repeats = max(1, min(constraints.max_events_per_electrode, int(dur * 1000.0 / ipi)))
            block: StimBlock = AntiSTDPPairingBlock(
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
        elif family == "probe_triggered":
            block = ProbeTriggeredBlock(
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
        elif family == "coordinated_reset":
            clusters = _sample_clusters(rng, constraints.allowed_electrodes)
            cycle_period = float(rng.choice([80.0, 120.0, 180.0, 250.0]))
            cycles = max(1, int(dur * 1000.0 / cycle_period))
            offsets = tuple(float(i * cycle_period / max(len(clusters), 1)) for i in range(len(clusters)))
            block = CoordinatedResetBlock(
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
        elif family == "low_frequency_depotentiation":
            electrodes = _task_biased_electrodes(rng, constraints.allowed_electrodes, input_channel, target_channel)
            block = LowFrequencyDepotentiationBlock(
                electrodes=electrodes,
                frequency_hz=float(rng.choice([0.5, 1.0, 2.0, 5.0, 8.0])),
                duration_s=dur,
                amplitude_uA=amp,
                pulse_width_us=pulse_width,
                jitter_ms=float(rng.choice([0.0, 0.5, 1.0])),
            )
        elif family == "colored_noise":
            block = ColoredNoiseBlock(
                beta=float(rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0])),
                event_rate_hz=float(rng.choice([5.0, 10.0, 18.0, 28.0, 40.0])),
                spatial_mode=str(rng.choice(["independent", "clustered", "global"])),
                duration_s=dur,
                amplitude_uA=amp,
                pulse_width_us=pulse_width,
                jitter_ms=0.0,
            )
        elif family == "actuator_positive_control":
            block = ActuatorPositiveControlBlock(
                electrodes=tuple(int(channel) for channel in constraints.allowed_electrodes),
                frequency_hz=float(rng.choice(positive_control_frequency_hz)),
                duration_s=dur,
                amplitude_uA=amp,
                pulse_width_us=int(min(200, constraints.max_pulse_width_us)),
                jitter_ms=0.0,
            )
        elif family == "task_input_drive":
            mode = str(rng.choice(input_drive_modes))
            block = TaskInputDriveBlock(
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
        else:
            block = RestBlock(duration_s=dur)
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
            max_triggers=max(1, min(constraints.max_events_per_electrode, int(block.max_triggers + rng.integers(-3, 4)))),
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


def stim_program_features(program: StimProgram) -> np.ndarray:
    families = {
        "anti_stdp_pairing": 0,
        "probe_triggered": 1,
        "coordinated_reset": 2,
        "low_frequency_depotentiation": 3,
        "colored_noise": 4,
        "actuator_positive_control": 5,
        "task_input_drive": 6,
        "rest": 7,
    }
    values = np.zeros(len(STIM_FEATURE_NAMES), dtype=np.float64)
    if program.family in families:
        values[families[program.family]] = 1.0
    amps = [
        float(getattr(block, "amplitude_uA", 0.0))
        for block in program.blocks
        if hasattr(block, "amplitude_uA")
    ]
    widths = [
        float(getattr(block, "pulse_width_us", 0.0))
        for block in program.blocks
        if hasattr(block, "pulse_width_us")
    ]
    electrodes = _program_electrodes(program)
    event_count = _rough_event_count(program)
    rest_duration = sum(
        float(block.duration_s) for block in program.blocks if isinstance(block, RestBlock)
    )
    values[8] = program.total_duration_s
    values[9] = len(program.blocks)
    values[10] = event_count
    values[11] = _rough_energy_cost(program, event_count)
    values[12] = float(np.mean(amps)) if amps else 0.0
    values[13] = float(np.max(amps)) if amps else 0.0
    values[14] = float(np.mean(widths)) if widths else 0.0
    values[15] = _mean_attr(program, ("delay_ms", "anti_pair_delay_ms"))
    values[16] = _mean_attr(program, ("frequency_hz",))
    values[17] = _mean_attr(program, ("beta",))
    values[18] = _mean_attr(program, ("event_rate_hz",))
    values[19] = len(electrodes) / max(len(program.constraints.allowed_electrodes), 1)
    values[20] = _electrode_entropy(electrodes)
    values[21] = rest_duration / max(program.total_duration_s, 1e-9)
    return values


def _tuple_to_list(value: tuple[Any, ...]) -> list[Any]:
    return [_tuple_to_list(item) if isinstance(item, tuple) else item for item in value]


def _bounded_choice(
    rng: np.random.Generator,
    values: tuple[float, ...] | list[float],
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
    allowed = list(int(v) for v in electrodes)
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
    allowed = set(int(channel) for channel in allowed_electrodes)
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


def _rough_event_count(program: StimProgram) -> float:
    total = 0.0
    for block in program.blocks:
        if isinstance(block, AntiSTDPPairingBlock):
            total += 2 * block.repeat_count
        elif isinstance(block, ProbeTriggeredBlock):
            total += 3 * block.max_triggers
        elif isinstance(block, CoordinatedResetBlock):
            total += block.cycles * max(len(block.clusters), 1)
        elif isinstance(block, LowFrequencyDepotentiationBlock):
            total += block.frequency_hz * block.duration_s * max(len(block.electrodes), 1)
        elif isinstance(block, ColoredNoiseBlock):
            total += block.event_rate_hz * block.duration_s
        elif isinstance(block, ActuatorPositiveControlBlock):
            total += block.frequency_hz * block.duration_s * max(len(block.electrodes), 1)
        elif isinstance(block, TaskInputDriveBlock):
            total += block.frequency_hz * block.duration_s * max(len(block.electrodes), 1)
    return float(total)


def _rough_energy_cost(program: StimProgram, event_count: float) -> float:
    amps = [float(getattr(block, "amplitude_uA", 0.0)) for block in program.blocks]
    widths = [float(getattr(block, "pulse_width_us", 160.0)) for block in program.blocks]
    amp = float(np.mean(amps)) if amps else 0.0
    width = float(np.mean(widths)) if widths else 160.0
    return abs(amp) * width * 1e-6 * max(event_count, 0.0) * 2.0


def _program_electrodes(program: StimProgram) -> list[int]:
    electrodes: list[int] = []
    for block in program.blocks:
        if isinstance(block, AntiSTDPPairingBlock):
            electrodes.extend([block.first_electrode, block.second_electrode])
        elif isinstance(block, ProbeTriggeredBlock):
            electrodes.extend([block.probe_electrode, block.response_electrode])
        elif isinstance(block, CoordinatedResetBlock):
            for cluster in block.clusters:
                electrodes.extend(cluster)
        elif isinstance(block, LowFrequencyDepotentiationBlock):
            electrodes.extend(block.electrodes)
        elif isinstance(block, ActuatorPositiveControlBlock):
            electrodes.extend(block.electrodes)
        elif isinstance(block, TaskInputDriveBlock):
            electrodes.extend(block.electrodes)
    return [int(value) for value in electrodes]


def _electrode_entropy(electrodes: list[int]) -> float:
    if not electrodes:
        return 0.0
    counts = np.bincount(np.asarray(electrodes, dtype=int))
    p = counts[counts > 0] / float(np.sum(counts))
    entropy = -float(np.sum(p * np.log2(p)))
    return entropy / max(np.log2(max(len(counts), 2)), 1e-9)


def _mean_attr(program: StimProgram, attrs: tuple[str, ...]) -> float:
    values: list[float] = []
    for block in program.blocks:
        for attr in attrs:
            if hasattr(block, attr):
                values.append(float(getattr(block, attr)))
    return float(np.mean(values)) if values else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))
