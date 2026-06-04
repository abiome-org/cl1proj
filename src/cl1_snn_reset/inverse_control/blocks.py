from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from cl1_snn_reset.protocols import ResetProtocol


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
    def from_dict(cls, values: dict[str, Any] | None) -> StimConstraints:
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


_COLORED_SPATIAL_TO_PROTOCOL = {
    "clustered": "correlated",
    "global": "shared",
    "independent": "independent",
}


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

    def to_reset_protocol(self) -> ResetProtocol:
        return ResetProtocol(
            beta=float(self.beta),
            duration_s=float(self.duration_s),
            current_uA=float(self.amplitude_uA),
            pulse_width_us=int(self.pulse_width_us),
            schedule="static",
            spatial_mode=_COLORED_SPATIAL_TO_PROTOCOL[self.spatial_mode],
            burst_rate_hz=float(self.event_rate_hz),
        )


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
    def from_json(cls, payload: dict[str, Any]) -> StimProgram:
        return cls(
            blocks=tuple(block_from_json(item) for item in payload.get("blocks", [])),
            constraints=StimConstraints.from_dict(payload.get("constraints", {})),
            metadata=dict(payload.get("metadata", {})),
            random_seed=int(payload.get("random_seed", 0)),
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


def _tuple_to_list(value: tuple[Any, ...]) -> list[Any]:
    return [_tuple_to_list(item) if isinstance(item, tuple) else item for item in value]
