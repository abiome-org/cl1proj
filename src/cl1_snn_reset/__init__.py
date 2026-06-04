"""
CL1-like SNN reset simulator library.

Import this package for culture simulation, train-reset-relearn trials, protocol
generation, metrics, and batch sweeps. Runnable study scripts live under
``experiments/`` and must not modify ``src/``.
"""
from .analysis import pareto_front, rank_protocols
from .config import CultureConfig, ExperimentConfig, TaskConfig, load_experiment_config
from .electrodes import ChannelActivity, ElectrodeArray, StimEvent
from .artifacts import TrialArtifacts
from .experiment import (
    PhaseSnapshot,
    TrialResult,
    apply_reset_protocol,
    build_trial_artifacts,
    capture_phase,
    record_spontaneous_activity,
    run_trial,
)
from .metrics import TrialMetrics, compute_trial_metrics, savings_score, weight_erasure_score
from .network import Brian2CultureNetwork, CorticalCultureNetwork, NetworkSnapshot, build_network
from .noise import colored_noise, generate_colored_events
from .protocols import ResetProtocol, coarse_protocol_grid, protocol_events
from .sweep import run_sweep, summarize_sweep
from .task import TrainingResult, evaluate_task, train_to_criterion
from .trace_probe import trace_probe_auc, trace_auc_proxy

__all__ = [
    "Brian2CultureNetwork",
    "ChannelActivity",
    "CorticalCultureNetwork",
    "CultureConfig",
    "ElectrodeArray",
    "ExperimentConfig",
    "NetworkSnapshot",
    "PhaseSnapshot",
    "ResetProtocol",
    "StimEvent",
    "TaskConfig",
    "TrainingResult",
    "TrialArtifacts",
    "TrialMetrics",
    "TrialResult",
    "apply_reset_protocol",
    "build_network",
    "build_trial_artifacts",
    "capture_phase",
    "coarse_protocol_grid",
    "colored_noise",
    "compute_trial_metrics",
    "evaluate_task",
    "generate_colored_events",
    "load_experiment_config",
    "pareto_front",
    "protocol_events",
    "rank_protocols",
    "record_spontaneous_activity",
    "run_sweep",
    "run_trial",
    "savings_score",
    "summarize_sweep",
    "trace_auc_proxy",
    "trace_probe_auc",
    "train_to_criterion",
    "weight_erasure_score",
]
