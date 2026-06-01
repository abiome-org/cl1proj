"""
CL1-like SNN reset simulator.

The package exposes a spatial recurrent E/I culture, a first-class MEA
electrode layer, colored pulse reset protocols, train-reset-relearn experiments,
trace probes, and parallel sweeps.
"""
from .analysis import pareto_front, plot_pareto_summary, plot_protocol_scatter, rank_protocols
from .config import CultureConfig, ExperimentConfig, SweepConfig, TaskConfig, load_experiment_config
from .electrodes import ChannelActivity, ElectrodeArray, StimEvent
from .experiment import TrialResult, apply_reset_protocol, record_spontaneous_activity, run_trial
from .metrics import TrialMetrics, savings_score, weight_erasure_score
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
    "ResetProtocol",
    "StimEvent",
    "SweepConfig",
    "TaskConfig",
    "TrainingResult",
    "TrialMetrics",
    "TrialResult",
    "apply_reset_protocol",
    "build_network",
    "coarse_protocol_grid",
    "colored_noise",
    "evaluate_task",
    "generate_colored_events",
    "load_experiment_config",
    "pareto_front",
    "plot_pareto_summary",
    "plot_protocol_scatter",
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
