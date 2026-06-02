from cl1_clsdk_bridge import (
    RESET_DYNAMICS_MODES,
    is_reset_dynamics,
    reset_backend_for_mode,
)


def test_reset_dynamics_modes():
    assert is_reset_dynamics("snn_reset")
    assert is_reset_dynamics("Brian2_Reset")
    assert not is_reset_dynamics("izhikevich")
    assert RESET_DYNAMICS_MODES == frozenset({"snn_reset", "reset_snn", "brian2_reset"})


def test_reset_backend_for_mode():
    assert reset_backend_for_mode("brian2_reset") == "brian2"
    assert reset_backend_for_mode("snn_reset") == "numpy"
