import numpy as np

from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    StimEvent,
    TaskConfig,
    build_network,
    protocol_events,
    run_trial,
)


def small_culture(**overrides):
    values = dict(
        n_neurons=192,
        mean_out_degree=16,
        max_out_degree=24,
        local_candidate_multiplier=3,
        background_noise_mv=0.8,
        spontaneous_rate_hz=0.0,
        stim_gain_mv_per_uA=8.0,
        v_threshold_mv=-58.0,
        backend="numpy",
    )
    values.update(overrides)
    return CultureConfig(**values)


def test_snn_reset_network_preserves_fixed_sign_weights():
    cfg = small_culture()
    net = build_network(cfg, seed=3)
    before_sign = np.sign(net.weights_vector())
    activity = net.advance(
        40.0,
        [StimEvent(time_us=0, channels=(8,), current_uA=3.0)],
        plasticity=True,
    )

    assert net.synapse_count > 0
    assert activity.duration_ms == 40.0
    assert np.array_equal(np.sign(net.weights_vector()), before_sign)


def test_reset_protocol_generates_channel_level_pulses():
    protocol = ResetProtocol(
        beta=1,
        duration_s=0.2,
        current_uA=1.0,
        pulse_width_us=160,
        schedule="epoch_pause",
        spatial_mode="correlated",
        burst_rate_hz=40,
        epoch_s=0.1,
        pause_s=0.05,
    )
    events = protocol_events(protocol, n_channels=64, rng=np.random.default_rng(4))

    assert events
    assert all(event.time_us >= 0 for event in events)
    assert all(0 <= channel < 64 for event in events for channel in event.channels)


def test_train_reset_relearn_trial_returns_required_metrics():
    cfg = ExperimentConfig(
        culture=small_culture(n_neurons=224),
        task=TaskConfig(
            input_channels=(8,),
            target_channels=(9,),
            max_trials=8,
            eval_interval_trials=4,
            eval_trials=3,
            criterion_response_probability=0.5,
            inter_trial_ms=40.0,
        ),
        readout_window_s=0.1,
        warmup_s=0.0,
    )
    protocol = ResetProtocol(
        beta=0,
        duration_s=0.12,
        current_uA=1.2,
        pulse_width_us=160,
        schedule="static",
        spatial_mode="independent",
        burst_rate_hz=30,
    )

    result = run_trial(cfg, protocol, seed=5)
    row = result.to_row()

    assert row["protocol_id"] == protocol.id
    assert row["total_pulses"] >= 0
    assert -10.0 < row["weight_erasure"] < 10.0
    assert 0.0 <= row["residual_performance"] <= 1.0
    assert 0.5 <= row["trace_auc"] <= 1.0
