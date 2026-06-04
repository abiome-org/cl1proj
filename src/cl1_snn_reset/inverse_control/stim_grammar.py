"""Backward-compatible re-exports for the inverse-control stimulation grammar."""

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
    block_from_json,
    block_to_json,
)
from .program_features import STIM_FEATURE_NAMES, stim_program_features
from .stim_sampling import StimSamplingConfig, mutate_stim_program, sample_stim_programs

__all__ = [
    "ActuatorPositiveControlBlock",
    "AntiSTDPPairingBlock",
    "ColoredNoiseBlock",
    "CoordinatedResetBlock",
    "LowFrequencyDepotentiationBlock",
    "ProbeTriggeredBlock",
    "RestBlock",
    "StimBlock",
    "StimConstraints",
    "StimProgram",
    "StimSamplingConfig",
    "TaskInputDriveBlock",
    "STIM_FEATURE_NAMES",
    "block_from_json",
    "block_to_json",
    "mutate_stim_program",
    "sample_stim_programs",
    "stim_program_features",
]
