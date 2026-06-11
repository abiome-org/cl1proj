"""TrialArtifacts: the intermediate weight/activity bundle handed from a trial to metric computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrodes import ChannelActivity
from .protocols import ResetProtocol
from .task import TrainingResult


@dataclass(frozen=True)
class TrialArtifacts:
    W0: np.ndarray
    Wtrained: np.ndarray
    Wpost: np.ndarray
    A0: ChannelActivity
    Apost: ChannelActivity
    initial: TrainingResult
    relearn: TrainingResult
    post_behavior: float
    protocol: ResetProtocol
    seed: int
    total_pulses: int
    trace_auc_proxy: float
    path0: float
    path_trained: float
    path_post: float
