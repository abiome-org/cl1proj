from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cl1_snn_reset.config import ExperimentConfig

from .program_features import STIM_FEATURE_NAMES, stim_program_features
from .pulse_compiler import (
    InvalidStimProgramError,
    compile_program_to_stim_events,
    estimate_energy_cost,
)
from .state_projectors import StateProjector, StateVectorSpec
from .blocks import StimConstraints, StimProgram
from .stim_sampling import StimSamplingConfig, sample_stim_programs
from .training_rollout import train_baseline_and_task_states


@dataclass(frozen=True)
class RolloutExample:
    seed: int
    task_input_channel: int
    task_target_channel: int
    baseline_state: np.ndarray
    trained_state: np.ndarray
    no_reset_state: np.ndarray
    stimmed_state: np.ndarray
    stim_program_json: dict[str, Any]
    stim_features: np.ndarray
    regime_features: np.ndarray
    causal_delta: np.ndarray
    duration_s: float
    energy_cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = np.asarray(self.stimmed_state) - np.asarray(self.no_reset_state)
        if not np.allclose(np.asarray(self.causal_delta), expected):
            raise ValueError("RolloutExample causal_delta must equal stimmed_state - no_reset_state.")


@dataclass(frozen=True)
class CausalDeltaDataset:
    examples: tuple[RolloutExample, ...]
    state_spec: StateVectorSpec
    stim_feature_names: tuple[str, ...]
    regime_feature_names: tuple[str, ...]

    @property
    def baseline_states(self) -> np.ndarray:
        return np.vstack([example.baseline_state for example in self.examples])

    @property
    def trained_states(self) -> np.ndarray:
        return np.vstack([example.trained_state for example in self.examples])

    @property
    def no_reset_states(self) -> np.ndarray:
        return np.vstack([example.no_reset_state for example in self.examples])

    @property
    def stimmed_states(self) -> np.ndarray:
        return np.vstack([example.stimmed_state for example in self.examples])

    @property
    def stim_features(self) -> np.ndarray:
        return np.vstack([example.stim_features for example in self.examples])

    @property
    def regime_features(self) -> np.ndarray:
        return np.vstack([example.regime_features for example in self.examples])

    @property
    def causal_deltas(self) -> np.ndarray:
        return np.vstack([example.causal_delta for example in self.examples])

    def save(self, output_dir: Path) -> None:
        dataset_dir = Path(output_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_dir / "states.npz",
            baseline_states=self.baseline_states,
            trained_states=self.trained_states,
            no_reset_states=self.no_reset_states,
            stimmed_states=self.stimmed_states,
            stim_features=self.stim_features,
            regime_features=self.regime_features,
            causal_deltas=self.causal_deltas,
            durations=np.asarray([example.duration_s for example in self.examples], dtype=np.float64),
            energy_costs=np.asarray([example.energy_cost for example in self.examples], dtype=np.float64),
            seeds=np.asarray([example.seed for example in self.examples], dtype=np.int64),
            task_inputs=np.asarray([example.task_input_channel for example in self.examples], dtype=np.int64),
            task_targets=np.asarray([example.task_target_channel for example in self.examples], dtype=np.int64),
        )
        (dataset_dir / "stim_programs.jsonl").write_text(
            "".join(json.dumps(example.stim_program_json, sort_keys=True) + "\n" for example in self.examples),
            encoding="utf-8",
        )
        (dataset_dir / "state_vector_spec.json").write_text(
            json.dumps(self.state_spec.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        table = pd.DataFrame(
            [
                {
                    "example_id": example.metadata.get("example_id"),
                    "seed": example.seed,
                    "task_input_channel": example.task_input_channel,
                    "task_target_channel": example.task_target_channel,
                    "protocol_id": example.metadata.get("protocol_id"),
                    "program_family": example.metadata.get("program_family"),
                    "duration_s": example.duration_s,
                    "energy_cost": example.energy_cost,
                    "training_reached_criterion": example.metadata.get("training_reached_criterion"),
                    "training_trials": example.metadata.get("training_trials"),
                    "compiled_event_count": example.metadata.get("compiled_event_count"),
                    "stimulus_effect_norm": float(np.linalg.norm(example.causal_delta)),
                    "no_reset_spikes": example.metadata.get("no_reset_spikes"),
                    "stimmed_spikes": example.metadata.get("stimmed_spikes"),
                    "spike_delta": example.metadata.get("spike_delta"),
                    "weight_delta_norm": example.metadata.get("weight_delta_norm"),
                }
                for example in self.examples
            ]
        )
        _write_table(table, dataset_dir / "examples.parquet")

    @classmethod
    def load(cls, dataset_dir: Path) -> "CausalDeltaDataset":
        dataset_dir = Path(dataset_dir)
        with (dataset_dir / "state_vector_spec.json").open(encoding="utf-8") as handle:
            spec = StateVectorSpec.from_json_dict(json.load(handle))
        arrays = np.load(dataset_dir / "states.npz")
        programs = [
            json.loads(line)
            for line in (dataset_dir / "stim_programs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata_table = _read_examples_table(dataset_dir)
        examples: list[RolloutExample] = []
        for index, program in enumerate(programs):
            row_metadata = (
                metadata_table.iloc[index].dropna().to_dict()
                if metadata_table is not None and index < len(metadata_table)
                else {}
            )
            examples.append(
                RolloutExample(
                    seed=int(arrays["seeds"][index]),
                    task_input_channel=int(arrays["task_inputs"][index]),
                    task_target_channel=int(arrays["task_targets"][index]),
                    baseline_state=arrays["baseline_states"][index],
                    trained_state=arrays["trained_states"][index],
                    no_reset_state=arrays["no_reset_states"][index],
                    stimmed_state=arrays["stimmed_states"][index],
                    stim_program_json=program,
                    stim_features=arrays["stim_features"][index],
                    regime_features=arrays["regime_features"][index],
                    causal_delta=arrays["causal_deltas"][index],
                    duration_s=float(arrays["durations"][index]),
                    energy_cost=float(arrays["energy_costs"][index]),
                    metadata=row_metadata | dict(program.get("metadata", {})) | {
                        "example_id": index,
                        "protocol_id": program.get("metadata", {}).get("protocol_id"),
                        "program_family": program.get("metadata", {}).get("program_family"),
                    },
                )
            )
        return cls(
            examples=tuple(examples),
            state_spec=spec,
            stim_feature_names=tuple(STIM_FEATURE_NAMES),
            regime_feature_names=("duration_s", "energy_cost", "training_response_probability"),
        )


class CausalDeltaDatasetBuilder:
    def __init__(
        self,
        *,
        projector: StateProjector,
        experiment_config: ExperimentConfig,
        constraints: StimConstraints,
        seeds: Iterable[int],
        programs_per_trained_state: int,
        stim_sampling: StimSamplingConfig | dict[str, Any],
        random_seed: int = 123,
    ):
        self.projector = projector
        self.experiment_config = experiment_config
        self.constraints = constraints
        self.seeds = tuple(int(seed) for seed in seeds)
        self.programs_per_trained_state = int(programs_per_trained_state)
        self.stim_sampling = (
            stim_sampling
            if isinstance(stim_sampling, StimSamplingConfig)
            else StimSamplingConfig.from_dict(stim_sampling)
        )
        self.random_seed = int(random_seed)

    def build(self) -> CausalDeltaDataset:
        examples: list[RolloutExample] = []
        rng = np.random.default_rng(self.random_seed)
        task = self.experiment_config.task
        input_channel = int(task.input_channels[0])
        target_channel = int(task.target_channels[0])
        for seed in self.seeds:
            trained_net, baseline_state, trained_state, baseline_activity, training = train_baseline_and_task_states(
                self.experiment_config,
                self.projector,
                seed,
            )
            programs = sample_stim_programs(
                count=self.programs_per_trained_state,
                constraints=self.constraints,
                input_channel=input_channel,
                target_channel=target_channel,
                rng=rng,
                sampling=self.stim_sampling,
            )
            wait_cache: dict[float, tuple[np.ndarray, Any, Any]] = {}
            for local_index, program in enumerate(programs):
                protocol_id = f"inv_seed{seed}_{local_index:05d}_{program.family}"
                program = StimProgram(
                    blocks=program.blocks,
                    constraints=program.constraints,
                    metadata=dict(program.metadata) | {"protocol_id": protocol_id},
                    random_seed=program.random_seed,
                )
                try:
                    events = compile_program_to_stim_events(program)
                except InvalidStimProgramError:
                    continue
                duration_s = program.total_duration_s
                if duration_s not in wait_cache:
                    wait_net = copy.deepcopy(trained_net)
                    wait_activity = wait_net.advance(
                        duration_s * 1000.0,
                        [],
                        plasticity=True,
                        record=True,
                    )
                    wait_state = self.projector.project(
                        wait_net,
                        activity=wait_activity,
                        baseline_activity=baseline_activity,
                    )
                    wait_cache[duration_s] = (wait_state, wait_net, wait_activity)
                no_reset_state, wait_net, wait_activity = wait_cache[duration_s]

                stim_net = copy.deepcopy(trained_net)
                stim_activity = stim_net.advance(
                    duration_s * 1000.0,
                    events,
                    plasticity=True,
                    record=True,
                )
                stim_state = self.projector.project(
                    stim_net,
                    activity=stim_activity,
                    baseline_activity=baseline_activity,
                )
                causal_delta = stim_state - no_reset_state
                energy_cost = estimate_energy_cost(program, events)
                metadata = {
                    "example_id": len(examples),
                    "protocol_id": protocol_id,
                    "program_family": program.family,
                    "compiled_event_count": len(events),
                    "training_reached_criterion": training.reached_criterion,
                    "training_trials": training.trials_to_criterion,
                    "training_response_probability": training.response_probability,
                    "no_reset_spikes": wait_activity.total_neuron_spikes,
                    "stimmed_spikes": stim_activity.total_neuron_spikes,
                    "spike_delta": int(stim_activity.total_neuron_spikes - wait_activity.total_neuron_spikes),
                    "weight_delta_norm": float(np.linalg.norm(stim_net.weights_vector() - wait_net.weights_vector())),
                }
                examples.append(
                    RolloutExample(
                        seed=seed,
                        task_input_channel=input_channel,
                        task_target_channel=target_channel,
                        baseline_state=baseline_state,
                        trained_state=trained_state,
                        no_reset_state=no_reset_state,
                        stimmed_state=stim_state,
                        stim_program_json=program.to_json(),
                        stim_features=stim_program_features(program),
                        regime_features=np.asarray(
                            [duration_s, energy_cost, training.response_probability],
                            dtype=np.float64,
                        ),
                        causal_delta=causal_delta,
                        duration_s=duration_s,
                        energy_cost=energy_cost,
                        metadata=metadata,
                    )
                )
        if not examples:
            raise RuntimeError("No valid causal delta examples were generated.")
        return CausalDeltaDataset(
            examples=tuple(examples),
            state_spec=self.projector.spec,
            stim_feature_names=tuple(STIM_FEATURE_NAMES),
            regime_feature_names=("duration_s", "energy_cost", "training_response_probability"),
        )

def _write_table(table: pd.DataFrame, parquet_path: Path) -> None:
    try:
        table.to_parquet(parquet_path, index=False)
    except Exception:
        table.to_csv(parquet_path.with_suffix(".csv"), index=False)


def _read_examples_table(dataset_dir: Path) -> pd.DataFrame | None:
    parquet_path = dataset_dir / "examples.parquet"
    csv_path = dataset_dir / "examples.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None
