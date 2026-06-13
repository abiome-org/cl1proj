"""Trace-detectability readout: a single-trial proxy on channel-level activity."""
from __future__ import annotations

import numpy as np

from .electrodes import ChannelActivity
from .metrics import activity_features


def trace_auc_proxy(naive: ChannelActivity, post: ChannelActivity) -> float:
    """
    Single-trial trace detectability proxy on channel-level readouts.

    Maps standardized feature distance into [0.5, 1.0] so individual rows still
    expose a trace detectability signal.
    """
    a = activity_features(naive)
    b = activity_features(post)
    distance = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-9))
    return float(0.5 + 0.5 * (1.0 - np.exp(-3.0 * distance)))
