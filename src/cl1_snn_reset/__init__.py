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
from .task_regimes import (
    ProbeMetrics,
    ProbeSpec,
    RegimeEvaluation,
    TaskRegime,
    TrainingTrialSpec,
    conditioned_electrode_association,
    delayed_conditioned_response,
    evaluate_probe,
    evaluate_regime,
    evoked_channel_response,
    pattern_discrimination,
    run_regime_grid,
    run_regime_reset_trial,
    run_regime_seed_protocols,
    run_training_trial,
    stim_event_ms,
    summarize_regime_grid,
    temporal_order_discrimination,
    train_regime,
)
from .task import (
    EvokedTaskMetrics,
    TaskBranchMetrics,
    TrainingResult,
    evaluate_evoked_task,
    evaluate_task,
    evaluate_task_branch,
    train_to_criterion,
)
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
    "ProbeMetrics",
    "ProbeSpec",
    "RegimeEvaluation",
    "ResetProtocol",
    "StimEvent",
    "TaskConfig",
    "TaskRegime",
    "TaskBranchMetrics",
    "TrainingResult",
    "TrainingTrialSpec",
    "TrialArtifacts",
    "TrialMetrics",
    "TrialResult",
    "EvokedTaskMetrics",
    "apply_reset_protocol",
    "build_network",
    "build_trial_artifacts",
    "capture_phase",
    "coarse_protocol_grid",
    "colored_noise",
    "compute_trial_metrics",
    "conditioned_electrode_association",
    "delayed_conditioned_response",
    "evaluate_evoked_task",
    "evaluate_probe",
    "evaluate_regime",
    "evaluate_task",
    "evaluate_task_branch",
    "evoked_channel_response",
    "generate_colored_events",
    "load_experiment_config",
    "pareto_front",
    "pattern_discrimination",
    "protocol_events",
    "rank_protocols",
    "run_regime_grid",
    "run_regime_reset_trial",
    "run_regime_seed_protocols",
    "record_spontaneous_activity",
    "run_sweep",
    "run_trial",
    "run_training_trial",
    "savings_score",
    "stim_event_ms",
    "summarize_regime_grid",
    "summarize_sweep",
    "temporal_order_discrimination",
    "trace_auc_proxy",
    "trace_probe_auc",
    "train_to_criterion",
    "train_regime",
    "weight_erasure_score",
]
