import numpy as np

from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    StimEvent,
    TaskConfig,
    build_network,
    conditioned_electrode_association,
    delayed_conditioned_response,
    evaluate_evoked_task,
    evaluate_regime,
    evaluate_task_branch,
    evoked_channel_response,
    pattern_discrimination,
    protocol_events,
    run_regime_reset_trial,
    run_trial,
    savings_score,
    summarize_sweep,
    temporal_order_discrimination,
    weight_erasure_score,
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


def test_stronger_reset_schedules_generate_timed_events():
    rng = np.random.default_rng(7)
    chirp = ResetProtocol(
        beta=0,
        duration_s=1.0,
        current_uA=5.0,
        pulse_width_us=160,
        schedule="chirp",
        spatial_mode="shared",
        burst_rate_hz=30.0,
    )
    gated = ResetProtocol(
        beta=1,
        duration_s=1.0,
        current_uA=5.0,
        pulse_width_us=160,
        schedule="gated_burst",
        spatial_mode="correlated",
        burst_rate_hz=180.0,
        epoch_s=0.2,
        pause_s=0.2,
    )

    chirp_events = protocol_events(chirp, n_channels=64, rng=rng)
    gated_events = protocol_events(gated, n_channels=64, rng=rng)

    assert chirp_events
    assert gated_events
    assert max(event.time_us for event in chirp_events) < 1_000_000
    assert max(event.time_us for event in gated_events) < 1_000_000
    assert len({event.time_us for event in chirp_events}) > 1
    assert len({event.time_us for event in gated_events}) > 1


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
    assert 0.5 <= row["trace_auc_proxy"] <= 1.0


def test_evoked_task_metrics_compare_input_to_sham_branches():
    cfg = small_culture(
        background_noise_mv=0.0,
        spontaneous_rate_hz=0.0,
    )
    net = build_network(cfg, seed=5)
    task = TaskConfig(
        input_channels=(8,),
        target_channels=(8,),
        input_current_uA=20.0,
        inter_trial_ms=40.0,
        eval_trials=3,
        response_window_ms=(0.0, 35.0),
    )

    metrics = evaluate_evoked_task(net, task, trials=3)

    assert metrics.input_metrics.response_probability > metrics.sham_metrics.response_probability
    assert metrics.evoked_response_probability > 0.0
    assert metrics.evoked_target_spikes_per_trial > 0.0
    assert metrics.evoked_total_spikes_per_trial > 0.0


def test_zero_current_evoked_task_has_no_input_sham_contrast():
    cfg = small_culture(
        background_noise_mv=0.0,
        spontaneous_rate_hz=0.0,
    )
    net = build_network(cfg, seed=5)
    task = TaskConfig(
        input_channels=(8,),
        target_channels=(8,),
        input_current_uA=0.0,
        inter_trial_ms=40.0,
        eval_trials=3,
        response_window_ms=(0.0, 35.0),
    )

    input_metrics = evaluate_task_branch(net, task, trials=3, with_input=True)
    evoked = evaluate_evoked_task(net, task, trials=3)

    assert input_metrics.response_probability == 0.0
    assert evoked.evoked_response_probability == 0.0
    assert evoked.evoked_target_spikes_per_trial == 0.0


def test_task_regime_presets_validate():
    regimes = [
        evoked_channel_response(),
        conditioned_electrode_association(),
        delayed_conditioned_response(),
        pattern_discrimination(),
        temporal_order_discrimination(),
    ]

    for regime in regimes:
        regime.validate()
        assert regime.name
        assert any(probe.is_positive for probe in regime.probes)
        assert any(probe.is_negative for probe in regime.probes)


def test_evoked_channel_regime_scores_input_against_sham():
    cfg = small_culture(background_noise_mv=0.0, spontaneous_rate_hz=0.0)
    net = build_network(cfg, seed=5)
    regime = evoked_channel_response(
        input_channel=8,
        target_channel=8,
        input_current_uA=20.0,
        eval_repetitions=3,
    )

    evaluation = evaluate_regime(net, regime, repetitions=3)

    assert evaluation.score > 0.0
    assert evaluation.positive_response_probability > evaluation.negative_response_probability


def test_regime_reset_trial_returns_task_and_weight_metrics():
    cfg = small_culture(
        n_neurons=224,
        background_noise_mv=0.0,
        spontaneous_rate_hz=0.0,
    )
    regime = evoked_channel_response(
        input_channel=8,
        target_channel=8,
        input_current_uA=20.0,
        eval_repetitions=2,
    )
    protocol = ResetProtocol(
        beta=0,
        duration_s=0.04,
        current_uA=1.0,
        pulse_width_us=160,
        schedule="static",
        spatial_mode="independent",
        burst_rate_hz=10,
    )

    row = run_regime_reset_trial(
        cfg,
        regime,
        protocol,
        seed=5,
        warmup_s=0.0,
        eval_repetitions=2,
    )

    assert row["task_name"] == "evoked_channel_response"
    assert row["protocol_id"] == protocol.id
    assert row["training_repetitions"] == 0
    assert row["trained_score"] > 0.0
    assert row["reset_minus_no_reset_weight_norm"] >= 0.0


def test_savings_score_handles_initial_criterion_edge_case():
    assert savings_score(0, 0) == 0.0
    assert savings_score(0, 12) == -1.0


def test_weight_erasure_score_uses_trained_delta_norm():
    w0 = np.array([0.0, 0.0])
    wtrained = np.array([3.0, 4.0])

    assert np.isclose(weight_erasure_score(w0, wtrained, w0), 1.0)
    assert np.isclose(weight_erasure_score(w0, wtrained, wtrained), 0.0)
    assert np.isclose(weight_erasure_score(w0, wtrained, np.array([1.5, 2.0])), 0.5)
    assert np.isclose(weight_erasure_score(w0, wtrained, np.array([6.0, 8.0])), -1.0)


def test_reset_protocol_has_zero_charge_for_zero_pulses():
    protocol = ResetProtocol(
        beta=0,
        duration_s=1.0,
        current_uA=2.0,
        pulse_width_us=160,
        schedule="static",
        spatial_mode="independent",
    )

    assert protocol.total_charge_uC(0) == 0.0


def test_summarize_sweep_averages_stochastic_pulse_counts():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "protocol_id": "p",
                "seed": 1,
                "beta": 0,
                "schedule": "static",
                "spatial_mode": "shared",
                "duration_s": 1.0,
                "current_uA": 1.0,
                "pulse_width_us": 160,
                "total_pulses": 10,
                "weight_erasure": 0.1,
                "residual_performance": 0.2,
                "savings": 0.3,
                "trace_auc_proxy": 0.5,
                "criticality_distance": 0.1,
                "health": 0.9,
                "energy_cost": 1.0,
                "path_erasure": 0.2,
            },
            {
                "protocol_id": "p",
                "seed": 2,
                "beta": 0,
                "schedule": "static",
                "spatial_mode": "shared",
                "duration_s": 1.0,
                "current_uA": 1.0,
                "pulse_width_us": 160,
                "total_pulses": 30,
                "weight_erasure": 0.3,
                "residual_performance": 0.4,
                "savings": 0.5,
                "trace_auc_proxy": 0.7,
                "criticality_distance": 0.3,
                "health": 0.7,
                "energy_cost": 3.0,
                "path_erasure": 0.4,
            },
        ]
    )

    summary = summarize_sweep(df)

    assert summary.loc[0, "total_pulses"] == 20
    assert summary.loc[0, "replicates"] == 2
