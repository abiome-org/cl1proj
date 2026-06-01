"""
Biological digital-twin simulator components.

The public CL API remains rooted at ``cl.open()``.  This package is deliberately
kept behind the producer boundary so the biological twin can grow toward a live
SNN/plasticity engine without leaking experimental concepts into normal SDK
application code.
"""
from .capabilities import TwinCapability, TwinCapabilityReport, describe_twin_capabilities
from .config import TwinConfig
from .dynamics import DynamicsSpike, PopulationDynamics
from .feedback import FeedbackStim, TwinFeedbackProtocol
from .izhikevich import IzhikevichNetwork, SNNSpike
from .learning import LearningCurveReport, TaskTrial, TwinLearningEvaluator
from .maturation import CultureState, MaturationState
from .mea import MEAGeometry
from .noise import PinkNoiseState
from .plasticity import PlasticityState
from .profile import TwinProfile
from .reset_adapter import ResetSNNAdapter
from .spike_detector import DetectionBlankingWindow, RollingThresholdSpikeDetector
from .sparse_izhikevich import SparseIzhikevichNetwork, SparseSynapseGraph
from .surrogate import SurrogateTwinModel
from .tissue import TissueTopology
from .training import TwinAcceleratedTrainer, TwinTrainingResult
from .validation import TwinValidationReport, TwinValidator

__all__ = [
    "TwinConfig",
    "TwinCapability",
    "TwinCapabilityReport",
    "DynamicsSpike",
    "FeedbackStim",
    "IzhikevichNetwork",
    "LearningCurveReport",
    "CultureState",
    "MaturationState",
    "PopulationDynamics",
    "SNNSpike",
    "MEAGeometry",
    "PinkNoiseState",
    "PlasticityState",
    "TaskTrial",
    "TwinProfile",
    "DetectionBlankingWindow",
    "RollingThresholdSpikeDetector",
    "ResetSNNAdapter",
    "SparseIzhikevichNetwork",
    "SparseSynapseGraph",
    "TwinFeedbackProtocol",
    "TwinAcceleratedTrainer",
    "TwinTrainingResult",
    "TwinLearningEvaluator",
    "SurrogateTwinModel",
    "TissueTopology",
    "TwinValidationReport",
    "TwinValidator",
    "describe_twin_capabilities",
]
