"""Frozen configuration dataclasses (culture, task, experiment, sweep) and the YAML experiment-config loader."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def _coerce_dataclass(cls: type, values: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    kwargs = {key: value for key, value in values.items() if key in allowed}
    return cls(**kwargs)


@dataclass(frozen=True)
class CultureConfig:
    """Spatial recurrent E/I culture used by the reset simulator."""

    n_neurons: int = 10_000
    excitatory_fraction: float = 0.8
    field_size_mm: float = 3.0
    n_electrodes: int = 64
    electrode_radius_mm: float = 0.12
    connection_length_mm: float = 0.25
    long_range_prob: float = 0.02
    mean_out_degree: int = 64
    max_out_degree: int = 96
    local_candidate_multiplier: int = 6
    build_workers: int = -1
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 6.0
    stim_tau_ms: float = 2.0
    v_rest_mv: float = -65.0
    v_reset_mv: float = -68.0
    v_threshold_mv: float = -50.0
    refractory_ms: float = 3.0
    background_noise_mv: float = 1.4
    spontaneous_rate_hz: float = 0.12
    excitatory_weight_range: tuple[float, float] = (0.08, 0.32)
    inhibitory_weight_range: tuple[float, float] = (-0.55, -0.12)
    w_max: float = 0.9
    w_min: float = -0.9
    stdp_tau_ms: float = 35.0
    stdp_a_plus: float = 0.012
    stdp_a_minus: float = 0.014
    homeostasis_interval_ms: float = 100.0
    target_rate_hz: float = 2.5
    homeostasis_rate: float = 0.012
    stim_gain_mv_per_uA: float = 5.5
    synaptic_gain_mv: float = 7.0
    record_kernel_gamma: float = 1.7
    stim_kernel_gamma: float = 1.35
    backend: str = "numpy"
    brian2_codegen_target: str = "numpy"


@dataclass(frozen=True)
class TaskConfig:
    """Electrode-to-electrode conditioned response task."""

    input_channels: tuple[int, ...] = (8,)
    target_channels: tuple[int, ...] = (55,)
    pair_delay_ms: float = 12.0
    inter_trial_ms: float = 70.0
    input_current_uA: float = 2.2
    target_current_uA: float = 2.0
    pulse_width_us: int = 160
    response_window_ms: tuple[float, float] = (4.0, 35.0)
    criterion_response_probability: float = 0.65
    max_trials: int = 120
    eval_trials: int = 8
    eval_interval_trials: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    """One train-reset-relearn replicate."""

    culture: CultureConfig = field(default_factory=CultureConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    readout_window_s: float = 1.5
    warmup_s: float = 0.5
    seed: int = 1
    keep_snapshots: bool = False


@dataclass(frozen=True)
class SweepConfig:
    """Parallel protocol sweep settings."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seeds: tuple[int, ...] = (1, 2, 3)
    workers: int = 1


def to_dict(value: Any) -> Any:
    """Convert nested dataclasses to plain Python dictionaries."""
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML with dataclass defaults."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for YAML config loading. Install this project with reset dependencies."
        ) from exc

    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    culture = _coerce_dataclass(CultureConfig, payload.get("culture", {}))
    task_values = payload.get("task", {})
    if "input_channels" in task_values:
        task_values["input_channels"] = tuple(task_values["input_channels"])
    if "target_channels" in task_values:
        task_values["target_channels"] = tuple(task_values["target_channels"])
    if "response_window_ms" in task_values:
        task_values["response_window_ms"] = tuple(task_values["response_window_ms"])
    task = _coerce_dataclass(TaskConfig, task_values)
    values = dict(payload)
    values.pop("culture", None)
    values.pop("task", None)
    return _coerce_dataclass(ExperimentConfig, values | {"culture": culture, "task": task})
