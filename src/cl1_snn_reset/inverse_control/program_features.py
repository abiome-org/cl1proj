from __future__ import annotations

import numpy as np

from cl1_snn_reset.protocols import pulse_energy_uC

from .blocks import (
    ActuatorPositiveControlBlock,
    AntiSTDPPairingBlock,
    ColoredNoiseBlock,
    CoordinatedResetBlock,
    LowFrequencyDepotentiationBlock,
    ProbeTriggeredBlock,
    RestBlock,
    StimProgram,
    TaskInputDriveBlock,
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

_FEATURE_INDEX = {name: index for index, name in enumerate(STIM_FEATURE_NAMES)}

_FAMILY_ONE_HOT = {
    "anti_stdp_pairing": "family:anti_stdp_pairing",
    "probe_triggered": "family:probe_triggered",
    "coordinated_reset": "family:coordinated_reset",
    "low_frequency_depotentiation": "family:low_frequency_depotentiation",
    "colored_noise": "family:colored_noise",
    "actuator_positive_control": "family:actuator_positive_control",
    "task_input_drive": "family:task_input_drive",
    "rest": "family:rest",
}


def stim_program_features(program: StimProgram) -> np.ndarray:
    values = np.zeros(len(STIM_FEATURE_NAMES), dtype=np.float64)
    family_key = _FAMILY_ONE_HOT.get(program.family)
    if family_key is not None:
        values[_FEATURE_INDEX[family_key]] = 1.0
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
    values[_FEATURE_INDEX["total_duration_s"]] = program.total_duration_s
    values[_FEATURE_INDEX["block_count"]] = len(program.blocks)
    values[_FEATURE_INDEX["rough_event_count"]] = event_count
    values[_FEATURE_INDEX["rough_energy_uC"]] = _rough_energy_cost(program, event_count)
    values[_FEATURE_INDEX["mean_amplitude_uA"]] = float(np.mean(amps)) if amps else 0.0
    values[_FEATURE_INDEX["max_amplitude_uA"]] = float(np.max(amps)) if amps else 0.0
    values[_FEATURE_INDEX["pulse_width_us"]] = float(np.mean(widths)) if widths else 0.0
    values[_FEATURE_INDEX["anti_pair_delay_ms"]] = _mean_attr(program, ("delay_ms", "anti_pair_delay_ms"))
    values[_FEATURE_INDEX["low_frequency_hz"]] = _mean_attr(program, ("frequency_hz",))
    values[_FEATURE_INDEX["colored_beta"]] = _mean_attr(program, ("beta",))
    values[_FEATURE_INDEX["colored_event_rate_hz"]] = _mean_attr(program, ("event_rate_hz",))
    values[_FEATURE_INDEX["electrode_coverage"]] = len(electrodes) / max(
        len(program.constraints.allowed_electrodes),
        1,
    )
    values[_FEATURE_INDEX["electrode_entropy"]] = _electrode_entropy(electrodes)
    values[_FEATURE_INDEX["rest_fraction"]] = rest_duration / max(program.total_duration_s, 1e-9)
    return values


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
    return pulse_energy_uC(amp, width, event_count)


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
