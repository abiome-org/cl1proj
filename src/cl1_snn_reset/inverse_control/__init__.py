"""
Learned inverse reset control for the CL1-style SNN reset simulator.
"""

from .controllability import ControllabilityReport, analyze_controllability
from .forward_models import MeanZeroDeltaModel, RidgeDeltaModel, evaluate_forward_model
from .inverse_optimizer import (
    CandidateProtocol,
    EliteMutationStimOptimizer,
    InverseResetObjective,
    RandomSearchOptimizer,
)
from .pulse_compiler import (
    InvalidStimProgramError,
    compile_program_to_stim_events,
    estimate_energy_cost,
    validate_stim_program,
)
from .rollout_dataset import (
    CausalDeltaDataset,
    CausalDeltaDatasetBuilder,
    RolloutExample,
)
from .state_projectors import (
    HybridStateProjector,
    StateProjector,
    StateVectorSpec,
    build_target_state,
)
from .blocks import StimConstraints, StimProgram
from .program_features import STIM_FEATURE_NAMES, stim_program_features
from .stim_sampling import StimSamplingConfig, sample_stim_programs
from .validation import bootstrap_candidate_effects, validate_candidate_against_no_reset

__all__ = [
    "CandidateProtocol",
    "CausalDeltaDataset",
    "CausalDeltaDatasetBuilder",
    "ControllabilityReport",
    "EliteMutationStimOptimizer",
    "HybridStateProjector",
    "InvalidStimProgramError",
    "InverseResetObjective",
    "MeanZeroDeltaModel",
    "RandomSearchOptimizer",
    "RidgeDeltaModel",
    "RolloutExample",
    "StateProjector",
    "StateVectorSpec",
    "StimConstraints",
    "StimProgram",
    "StimSamplingConfig",
    "STIM_FEATURE_NAMES",
    "analyze_controllability",
    "bootstrap_candidate_effects",
    "build_target_state",
    "compile_program_to_stim_events",
    "estimate_energy_cost",
    "evaluate_forward_model",
    "sample_stim_programs",
    "stim_program_features",
    "validate_candidate_against_no_reset",
    "validate_stim_program",
]
