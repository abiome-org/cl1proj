"""
Compatibility namespace for the reset simulator.

New code should import from ``cl1_snn_reset`` directly.  This module keeps
existing ``cl.snn_reset`` imports working for SDK-facing users and notebooks.
"""

from __future__ import annotations

import importlib
import sys

from cl1_snn_reset import *  # noqa: F401,F403
from cl1_snn_reset import __all__ as __all__

_SUBMODULES = (
    "analysis",
    "config",
    "electrodes",
    "experiment",
    "metrics",
    "network",
    "noise",
    "protocols",
    "sweep",
    "task",
    "trace_probe",
)

for _name in _SUBMODULES:
    _module = importlib.import_module(f"cl1_snn_reset.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

del importlib, sys, _SUBMODULES, _name, _module
