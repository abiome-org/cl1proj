from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .pulse_compiler import InvalidStimProgramError, compile_program_to_stim_events, estimate_energy_cost
from .state_projectors import StateVectorSpec
from .stim_grammar import StimConstraints, StimProgram, mutate_stim_program, sample_stim_programs, stim_program_features


@dataclass(frozen=True)
class CandidateProtocol:
    protocol_id: str
    stim_program: StimProgram
    predicted_delta: np.ndarray
    predicted_post_state: np.ndarray
    predicted_loss: float
    predicted_task_erasure: float
    predicted_health_penalty: float
    predicted_energy_cost: float
    model_uncertainty: float
    validation_metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "protocol_id": self.protocol_id,
            "program_family": self.stim_program.family,
            "duration_s": self.stim_program.total_duration_s,
            "energy_cost": self.predicted_energy_cost,
            "predicted_loss": self.predicted_loss,
            "predicted_task_erasure": self.predicted_task_erasure,
            "predicted_health_penalty": self.predicted_health_penalty,
            "model_uncertainty": self.model_uncertainty,
        }
        if self.validation_metrics:
            row.update(self.validation_metrics)
        return row


class InverseResetObjective:
    def __init__(
        self,
        state_spec: StateVectorSpec,
        *,
        loss_weights: dict[str, float] | None = None,
        max_energy_cost: float = 1.0,
    ):
        self.state_spec = state_spec
        self.loss_weights = {
            "task_trace": 3.0,
            "input_target_path": 3.0,
            "privileged_weight_projection": 2.0,
            "health": 2.0,
            "off_target_drift": 1.5,
            "savings_proxy": 2.0,
            "energy": 0.2,
            "uncertainty": 1.0,
        } | dict(loss_weights or {})
        self.max_energy_cost = float(max_energy_cost)
        self.task_mask = state_spec.group_mask(("task_path", "readout"))
        self.path_mask = state_spec.group_mask(("task_path",))
        self.privileged_mask = state_spec.group_mask(("privileged_weight_projection",))
        self.health_mask = state_spec.group_mask(("health", "criticality"))
        self.off_task_mask = ~(self.task_mask | self.privileged_mask | self.health_mask)

    def evaluate(
        self,
        *,
        predicted_post_state: np.ndarray,
        no_reset_state: np.ndarray,
        target_state: np.ndarray,
        energy_cost: float,
        uncertainty: float,
    ) -> dict[str, float]:
        pred = np.asarray(predicted_post_state, dtype=np.float64)
        no_reset = np.asarray(no_reset_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        task_remaining_no = _masked_norm(no_reset - target, self.task_mask)
        task_remaining_pred = _masked_norm(pred - target, self.task_mask)
        task_erasure = task_remaining_no - task_remaining_pred
        health_penalty = _masked_norm(pred - target, self.health_mask)
        losses = {
            "task_trace": _masked_mse(pred - target, self.task_mask),
            "input_target_path": _masked_mse(pred - target, self.path_mask),
            "privileged_weight_projection": _masked_mse(pred - target, self.privileged_mask),
            "health": _masked_mse(pred - target, self.health_mask),
            "off_target_drift": _masked_mse(pred - no_reset, self.off_task_mask),
            "savings_proxy": max(task_remaining_pred, 0.0) ** 2,
            "energy": (energy_cost / max(self.max_energy_cost, 1e-12)) ** 2,
            "uncertainty": float(uncertainty) ** 2,
        }
        total = float(sum(self.loss_weights[key] * value for key, value in losses.items()))
        return {
            "loss": total,
            "task_erasure": float(task_erasure),
            "health_penalty": float(health_penalty),
            **{f"loss_{key}": float(value) for key, value in losses.items()},
        }


class LinearInverseSolver:
    def propose_from_dataset(
        self,
        *,
        model,
        programs: list[StimProgram],
        x_current: np.ndarray,
        no_reset_state: np.ndarray,
        target_state: np.ndarray,
        objective: InverseResetObjective,
        candidate_count: int,
        protocol_prefix: str,
    ) -> list[CandidateProtocol]:
        return _score_programs(
            model=model,
            programs=programs,
            x_current=x_current,
            no_reset_state=no_reset_state,
            target_state=target_state,
            objective=objective,
            candidate_count=candidate_count,
            protocol_prefix=protocol_prefix,
        )


class RandomSearchOptimizer:
    def __init__(
        self,
        *,
        max_evaluations: int = 1000,
        random_seed: int = 123,
    ):
        self.max_evaluations = int(max_evaluations)
        self.random_seed = int(random_seed)

    def propose(
        self,
        *,
        model,
        x_current: np.ndarray,
        no_reset_state: np.ndarray,
        target_state: np.ndarray,
        objective: InverseResetObjective,
        constraints: StimConstraints,
        input_channel: int,
        target_channel: int,
        stim_sampling: dict[str, Any],
        candidate_count: int,
        protocol_prefix: str = "candidate",
    ) -> list[CandidateProtocol]:
        rng = np.random.default_rng(self.random_seed)
        programs = sample_stim_programs(
            count=self.max_evaluations,
            constraints=constraints,
            input_channel=input_channel,
            target_channel=target_channel,
            rng=rng,
            include_blocks=tuple(stim_sampling.get("include_blocks", ())),
            amplitude_uA=tuple(float(v) for v in stim_sampling.get("amplitude_uA", (0.8, 1.2, 1.6, 2.0))),
            duration_s=tuple(float(v) for v in stim_sampling.get("duration_s", (0.75, 1.5, 3.0))),
            delays_ms=tuple(float(v) for v in stim_sampling.get("delays_ms", (2, 5, 10, 20, 40, 80))),
            positive_control_frequency_hz=tuple(float(v) for v in stim_sampling.get("positive_control_frequency_hz", (80.0, 100.0, 120.0))),
            input_drive_frequency_hz=tuple(float(v) for v in stim_sampling.get("input_drive_frequency_hz", (80.0, 90.0, 100.0, 110.0, 140.0, 160.0, 200.0))),
            input_drive_modes=tuple(str(v) for v in stim_sampling.get("input_drive_modes", ("single_input", "input_neighborhood"))),
        )
        return _score_programs(
            model=model,
            programs=programs,
            x_current=x_current,
            no_reset_state=no_reset_state,
            target_state=target_state,
            objective=objective,
            candidate_count=candidate_count,
            protocol_prefix=protocol_prefix,
        )


class CMAESStimOptimizer(RandomSearchOptimizer):
    """
    Lightweight mixed-grammar evolution strategy.

    It uses random grammar sampling for exploration, then mutates elite valid
    pulse programs in a CEM/CMA-ES-like loop.  This avoids adding a dependency
    while still optimizing over continuous timing/amplitude parameters instead
    of ranking a fixed grid.
    """

    def __init__(
        self,
        *,
        max_evaluations: int = 1000,
        random_seed: int = 123,
        rounds: int = 4,
        elite_fraction: float = 0.20,
    ):
        super().__init__(max_evaluations=max_evaluations, random_seed=random_seed)
        self.rounds = int(rounds)
        self.elite_fraction = float(elite_fraction)

    def propose(self, **kwargs: Any) -> list[CandidateProtocol]:
        rng = np.random.default_rng(self.random_seed)
        budget = max(self.max_evaluations, kwargs["candidate_count"])
        per_round = max(kwargs["candidate_count"] * 4, budget // max(self.rounds, 1))
        pool = sample_stim_programs(
            count=per_round,
            constraints=kwargs["constraints"],
            input_channel=kwargs["input_channel"],
            target_channel=kwargs["target_channel"],
            rng=rng,
            include_blocks=tuple(kwargs["stim_sampling"].get("include_blocks", ())),
            amplitude_uA=tuple(float(v) for v in kwargs["stim_sampling"].get("amplitude_uA", (0.8, 1.2, 1.6, 2.0))),
            duration_s=tuple(float(v) for v in kwargs["stim_sampling"].get("duration_s", (0.75, 1.5, 3.0))),
            delays_ms=tuple(float(v) for v in kwargs["stim_sampling"].get("delays_ms", (2, 5, 10, 20, 40, 80))),
            positive_control_frequency_hz=tuple(float(v) for v in kwargs["stim_sampling"].get("positive_control_frequency_hz", (80.0, 100.0, 120.0))),
            input_drive_frequency_hz=tuple(float(v) for v in kwargs["stim_sampling"].get("input_drive_frequency_hz", (80.0, 90.0, 100.0, 110.0, 140.0, 160.0, 200.0))),
            input_drive_modes=tuple(str(v) for v in kwargs["stim_sampling"].get("input_drive_modes", ("single_input", "input_neighborhood"))),
        )
        evaluated: list[CandidateProtocol] = []
        for round_index in range(max(self.rounds, 1)):
            scored = _score_programs(
                model=kwargs["model"],
                programs=pool,
                x_current=kwargs["x_current"],
                no_reset_state=kwargs["no_reset_state"],
                target_state=kwargs["target_state"],
                objective=kwargs["objective"],
                candidate_count=max(1, int(len(pool) * self.elite_fraction)),
                protocol_prefix=f"{kwargs.get('protocol_prefix', 'candidate')}_r{round_index}",
            )
            evaluated.extend(scored)
            if len(evaluated) >= budget:
                break
            elites = [candidate.stim_program for candidate in scored]
            pool = []
            for elite in elites:
                mutations = max(1, per_round // max(len(elites), 1))
                for _ in range(mutations):
                    pool.append(
                        mutate_stim_program(
                            elite,
                            rng=rng,
                            input_channel=kwargs["input_channel"],
                            target_channel=kwargs["target_channel"],
                            scale=max(0.08, 0.45 * (0.7 ** round_index)),
                        )
                    )
        unique = _unique_candidates(evaluated)
        return sorted(unique, key=lambda candidate: candidate.predicted_loss)[: kwargs["candidate_count"]]


def write_candidates(
    candidates: list[CandidateProtocol],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame([candidate.to_row() for candidate in candidates]).to_csv(
        output_dir / "optimized_protocols.csv",
        index=False,
    )
    with (output_dir / "optimized_protocols.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = candidate.stim_program.to_json() | {
                "protocol_id": candidate.protocol_id,
                "predicted_loss": candidate.predicted_loss,
                "predicted_task_erasure": candidate.predicted_task_erasure,
                "model_uncertainty": candidate.model_uncertainty,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _score_programs(
    *,
    model,
    programs: list[StimProgram],
    x_current: np.ndarray,
    no_reset_state: np.ndarray,
    target_state: np.ndarray,
    objective: InverseResetObjective,
    candidate_count: int,
    protocol_prefix: str,
) -> list[CandidateProtocol]:
    candidates: list[CandidateProtocol] = []
    x = np.asarray(x_current, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    no_reset = np.asarray(no_reset_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    for index, program in enumerate(programs):
        try:
            events = compile_program_to_stim_events(program)
        except InvalidStimProgramError:
            continue
        stim = stim_program_features(program)[None, :]
        energy = estimate_energy_cost(program, events)
        regime = np.asarray([[program.total_duration_s, energy, 1.0]], dtype=np.float64)
        pred_delta = model.predict_delta(x, stim, regime)[0]
        uncertainty = float(model.predict_uncertainty(x, stim, regime)[0])
        pred_post = no_reset + pred_delta
        scored = objective.evaluate(
            predicted_post_state=pred_post,
            no_reset_state=no_reset,
            target_state=target,
            energy_cost=energy,
            uncertainty=uncertainty,
        )
        protocol_id = f"{protocol_prefix}_{index:05d}_{program.family}"
        program = StimProgram(
            blocks=program.blocks,
            constraints=program.constraints,
            metadata=dict(program.metadata) | {"protocol_id": protocol_id},
            random_seed=program.random_seed,
        )
        candidates.append(
            CandidateProtocol(
                protocol_id=protocol_id,
                stim_program=program,
                predicted_delta=pred_delta,
                predicted_post_state=pred_post,
                predicted_loss=float(scored["loss"]),
                predicted_task_erasure=float(scored["task_erasure"]),
                predicted_health_penalty=float(scored["health_penalty"]),
                predicted_energy_cost=energy,
                model_uncertainty=uncertainty,
                metadata={key: value for key, value in scored.items() if key.startswith("loss_")},
            )
        )
    return sorted(_unique_candidates(candidates), key=lambda candidate: candidate.predicted_loss)[:candidate_count]


def _unique_candidates(candidates: list[CandidateProtocol]) -> list[CandidateProtocol]:
    seen: set[str] = set()
    result: list[CandidateProtocol] = []
    for candidate in sorted(candidates, key=lambda item: item.predicted_loss):
        key = json.dumps(candidate.stim_program.to_json()["blocks"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _masked_norm(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(values[mask]))


def _masked_mse(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.square(values[mask])))
