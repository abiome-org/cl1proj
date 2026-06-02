from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cl1_clsdk_bridge import RESET_DYNAMICS_MODES, is_reset_dynamics

from .config import TwinConfig

_SNN_DYNAMICS = {"izhikevich", "snn"} | set(RESET_DYNAMICS_MODES)

CapabilityStatus = Literal["implemented", "approximated", "roadmap"]


@dataclass(frozen=True)
class TwinCapability:
    """One explicit capability in the biological-twin integration ladder."""

    name: str
    status: CapabilityStatus
    detail: str


@dataclass(frozen=True)
class TwinCapabilityReport:
    """
    Machine-readable audit of how closely the backend matches the CL1 twin goal.

    The biological-twin feedback spans several levels of realism: bidirectional
    MEA coupling, culture calibration, cell-level dynamics, plasticity,
    pharmacology, and accelerated training. This report keeps those promises
    explicit so applications can surface caveats instead of inferring too much
    from the experimental backend name.
    """

    sim_mode: str
    dynamics_mode: str
    plasticity_mode: str
    capabilities: tuple[TwinCapability, ...]

    @property
    def implemented_count(self) -> int:
        return sum(1 for item in self.capabilities if item.status == "implemented")

    @property
    def approximated_count(self) -> int:
        return sum(1 for item in self.capabilities if item.status == "approximated")

    @property
    def roadmap_count(self) -> int:
        return sum(1 for item in self.capabilities if item.status == "roadmap")

    def by_status(self, status: CapabilityStatus) -> tuple[TwinCapability, ...]:
        return tuple(item for item in self.capabilities if item.status == status)


def describe_twin_capabilities(config: TwinConfig | None = None) -> TwinCapabilityReport:
    """Describe the active biological-twin capability envelope."""
    resolved = config or TwinConfig.from_env()
    dynamics = resolved.dynamics_mode.lower()
    plasticity = resolved.plasticity_mode.lower()

    capabilities = (
        TwinCapability(
            name="Drop-in producer integration",
            status="implemented",
            detail=(
                "BiologicalTwinProducer preserves the cl.open/shared-buffer API "
                "and can be selected with CL_SDK_SIM_MODE."
            ),
        ),
        TwinCapability(
            name="Bidirectional MEA stimulation",
            status="implemented",
            detail=(
                "Stim commands carry channel, timing, and current into the twin, "
                "where electrode geometry alters future simulated frames/spikes."
            ),
        ),
        TwinCapability(
            name="Stimulation artifact",
            status="implemented",
            detail=(
                "Stim events add a decaying raw-frame artifact so blanking and "
                "artifact-aware code can be tested."
            ),
        ),
        TwinCapability(
            name="Pink MEA noise",
            status="implemented",
            detail=(
                "Raw frames use a stateful multi-timescale noise source so "
                "calibrated channel noise has 1/f-like temporal correlation."
            ),
        ),
        TwinCapability(
            name="Culture calibration profile",
            status="implemented",
            detail=(
                "TwinProfile fits rates, ISIs, bursts, connectivity, stim response "
                "fields, uncertainty intervals, and topology priors from recordings."
            ),
        ),
        TwinCapability(
            name="Validation gates",
            status="implemented",
            detail=(
                "TwinValidator compares simulated output to calibrated profile "
                "targets, including optional stim-response and artifact gates."
            ),
        ),
        TwinCapability(
            name="Population recurrence",
            status="implemented" if dynamics in {"population", *_SNN_DYNAMICS} else "approximated",
            detail=(
                "PopulationDynamics provides MEA-level recurrence, and SNN modes "
                "provide cell-level recurrent propagation."
            ),
        ),
        TwinCapability(
            name="Izhikevich cell SNN",
            status="implemented" if dynamics == "izhikevich" else "approximated",
            detail=(
                "IzhikevichNetwork is available behind CL_SDK_TWIN_DYNAMICS=izhikevich "
                "with 2D cells, E/I synapses, field coupling, delays, and refractory gating."
            ),
        ),
        TwinCapability(
            name="Reset-platform SNN",
            status="implemented" if is_reset_dynamics(dynamics) else "approximated",
            detail=(
                "The train-reset-relearn SNN can be selected in the SDK twin via "
                "CL_SDK_TWIN_DYNAMICS=snn_reset, preserving electrode-only input "
                "and channel-level spike output."
            ),
        ),
        TwinCapability(
            name="STDP",
            status="implemented" if plasticity in {"stdp", "stdp_homeostatic"} else "approximated",
            detail=(
                "SNN mode adapts cell synapses in STDP modes; surrogate mode keeps "
                "bounded STDP-like response-gain plasticity."
            ),
        ),
        TwinCapability(
            name="Short-term plasticity",
            status="implemented" if plasticity in {"stp", "stdp", "stdp_homeostatic"} else "approximated",
            detail=(
                "SNN recurrent transmission supports depression and facilitation in "
                "plasticity modes; surrogate plasticity uses bounded gain fatigue."
            ),
        ),
        TwinCapability(
            name="Virtual pharmacology",
            status="implemented",
            detail=(
                "DIV, dopamine, GABA blockade, excitability, and culture state are "
                "configurable runtime modulators."
            ),
        ),
        TwinCapability(
            name="Closed-loop feedback",
            status="implemented",
            detail=(
                "TwinFeedbackProtocol emits structured and chaotic stimulation "
                "patterns routed through the same MEA coupling."
            ),
        ),
        TwinCapability(
            name="Learning-curve evaluation",
            status="implemented",
            detail=(
                "TwinLearningEvaluator compares early and late closed-loop task "
                "windows for performance and response-rate changes."
            ),
        ),
        TwinCapability(
            name="Large-scale sparse SNN",
            status=(
                "implemented"
                if dynamics in _SNN_DYNAMICS
                and resolved.snn_neuron_count >= resolved.snn_sparse_threshold
                else "approximated"
            ),
            detail=(
                "Izhikevich mode switches to sparse adjacency-list recurrent "
                "synapses above CL_SDK_TWIN_SNN_SPARSE_THRESHOLD so 5k-cell "
                "cultures are structurally supported without dense N squared storage."
            ),
        ),
        TwinCapability(
            name="Accelerated biological training",
            status="implemented",
            detail=(
                "TwinAcceleratedTrainer runs closed-loop feedback trials against "
                "the same model path without wall-clock sleeps for long-horizon training."
            ),
        ),
        TwinCapability(
            name="Extracellular waveform detector",
            status="implemented",
            detail=(
                "Biphasic EAP templates are written into raw frames and converted "
                "to SpikeRecord objects by a rolling negative-threshold detector."
            ),
        ),
    )

    return TwinCapabilityReport(
        sim_mode="surrogate",
        dynamics_mode=resolved.dynamics_mode,
        plasticity_mode=resolved.plasticity_mode,
        capabilities=capabilities,
    )
