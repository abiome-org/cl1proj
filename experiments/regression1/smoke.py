from __future__ import annotations

import sys

from cl1_clsdk_bridge import ResetSNNAdapter
from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    TaskConfig,
    run_trial,
)
from cl.twin.config import TwinConfig


def assert_trial_smoke() -> None:
    cfg = ExperimentConfig(
        culture=CultureConfig(
            n_neurons=400,
            mean_out_degree=24,
            max_out_degree=48,
            local_candidate_multiplier=4,
            spontaneous_rate_hz=0.05,
            backend="numpy",
        ),
        task=TaskConfig(
            max_trials=12,
            eval_interval_trials=4,
            eval_trials=4,
            criterion_response_probability=0.5,
        ),
        readout_window_s=0.15,
        warmup_s=0.05,
    )
    protocol = ResetProtocol(
        beta=0,
        duration_s=0.2,
        current_uA=1.0,
        pulse_width_us=160,
        schedule="static",
        spatial_mode="independent",
        burst_rate_hz=40,
    )
    result = run_trial(cfg, protocol, seed=7)
    row = result.to_row()
    for key in ("weight_erasure", "health", "residual_performance", "protocol_id"):
        if key not in row:
            raise AssertionError(f"missing metric key: {key}")
    if not (0.0 <= float(row["health"]) <= 1.5):
        raise AssertionError(f"unexpected health: {row['health']}")


def assert_bridge_smoke() -> None:
    import numpy as np

    twin = TwinConfig(seed=3)
    adapter = ResetSNNAdapter(
        channel_count=64,
        neuron_count=400,
        frames_per_second=1000,
        rng=np.random.default_rng(3),
        config=twin,
    )
    adapter.apply_timed_stim(0, 8, np.array([1.2], dtype=np.float64))
    spikes = adapter.render(0, frame_count=50, excitability=np.ones(64), response_gain=np.ones(64))
    if not isinstance(spikes, list):
        raise AssertionError("ResetSNNAdapter.render did not return a spike list")


def main() -> None:
    assert_trial_smoke()
    assert_bridge_smoke()
    print("regression smoke: ok")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
