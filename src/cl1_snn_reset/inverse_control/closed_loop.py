from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .inverse_optimizer import CandidateProtocol, InverseResetObjective, RandomSearchOptimizer
from .state_projectors import StateProjector, build_target_state
from .stim_grammar import StimConstraints


@dataclass
class ClosedLoopResetSession:
    candidate_history: list[CandidateProtocol] = field(default_factory=list)
    state_history: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelPredictiveResetController:
    """
    Short-block MPC hook for Phase 4.

    The current MVP uses the same constrained optimizer for each measured state;
    wetware execution is intentionally outside this subsystem.
    """

    def __init__(
        self,
        *,
        model,
        projector: StateProjector,
        objective: InverseResetObjective,
        constraints: StimConstraints,
        optimizer: RandomSearchOptimizer | None = None,
    ):
        self.model = model
        self.projector = projector
        self.objective = objective
        self.constraints = constraints
        self.optimizer = optimizer or RandomSearchOptimizer(max_evaluations=128)

    def choose_block(
        self,
        *,
        current_state: np.ndarray,
        baseline_state: np.ndarray,
        trained_state: np.ndarray,
        no_reset_state: np.ndarray,
        input_channel: int,
        target_channel: int,
        stim_sampling: dict[str, Any],
    ) -> CandidateProtocol:
        target = build_target_state(
            self.objective.state_spec,
            baseline_state,
            trained_state,
            no_reset_state,
        )
        candidates = self.optimizer.propose(
            model=self.model,
            x_current=current_state,
            no_reset_state=no_reset_state,
            target_state=target,
            objective=self.objective,
            constraints=self.constraints,
            input_channel=input_channel,
            target_channel=target_channel,
            stim_sampling=stim_sampling,
            candidate_count=1,
            protocol_prefix="mpc_block",
        )
        if not candidates:
            raise RuntimeError("No valid closed-loop stimulation block was found.")
        return candidates[0]
