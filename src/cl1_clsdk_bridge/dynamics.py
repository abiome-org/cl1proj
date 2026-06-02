from __future__ import annotations

RESET_DYNAMICS_MODES = frozenset({"snn_reset", "reset_snn", "brian2_reset"})


def is_reset_dynamics(mode: str) -> bool:
    return mode.lower() in RESET_DYNAMICS_MODES


def reset_backend_for_mode(mode: str) -> str:
    if mode.lower() == "brian2_reset":
        return "brian2"
    return "numpy"
