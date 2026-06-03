"""
Learned inverse reset control for the CL1-style SNN reset simulator.

The subsystem learns no-reset-relative stimulation effects and searches over
MEA-valid pulse programs.  Simulation-only weight features are allowed for
supervision and diagnostics; compiled candidates still act only through
channel-level stimulation events.
"""

from .controllability import ControllabilityReport, analyze_controllability
from .forward_models import MeanZeroDeltaModel, RidgeDeltaModel, evaluate_forward_model
from .inverse_optimizer import (
    CMAESStimOptimizer,
    CandidateProtocol,
    InverseResetObjective,
    LinearInverseSolver,
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
    ObservableStateProjector,
    PrivilegedStateProjector,
    StateProjector,
    StateVectorSpec,
    build_target_state,
)
from .stim_grammar import (
    AntiSTDPPairingBlock,
    ActuatorPositiveControlBlock,
    ColoredNoiseBlock,
    CoordinatedResetBlock,
    LowFrequencyDepotentiationBlock,
    ProbeTriggeredBlock,
    RestBlock,
    StimConstraints,
    StimProgram,
    TaskInputDriveBlock,
    sample_stim_programs,
    stim_program_features,
)
from .validation import bootstrap_candidate_effects, validate_candidate_against_no_reset

__all__ = [
    "AntiSTDPPairingBlock",
    "ActuatorPositiveControlBlock",
    "CMAESStimOptimizer",
    "CandidateProtocol",
    "CausalDeltaDataset",
    "CausalDeltaDatasetBuilder",
    "ColoredNoiseBlock",
    "ControllabilityReport",
    "CoordinatedResetBlock",
    "HybridStateProjector",
    "InvalidStimProgramError",
    "InverseResetObjective",
    "LinearInverseSolver",
    "LowFrequencyDepotentiationBlock",
    "MeanZeroDeltaModel",
    "ObservableStateProjector",
    "PrivilegedStateProjector",
    "ProbeTriggeredBlock",
    "RandomSearchOptimizer",
    "RestBlock",
    "RidgeDeltaModel",
    "RolloutExample",
    "StateProjector",
    "StateVectorSpec",
    "StimConstraints",
    "StimProgram",
    "TaskInputDriveBlock",
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
