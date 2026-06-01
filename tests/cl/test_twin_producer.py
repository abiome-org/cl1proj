import json

import numpy as np

import cl
from cl._data_buffer import StimRecord
from cl.twin import CultureState, DetectionBlankingWindow, IzhikevichNetwork, MEAGeometry, MaturationState, PinkNoiseState, PlasticityState, PopulationDynamics, RollingThresholdSpikeDetector, SparseIzhikevichNetwork, SurrogateTwinModel, TaskTrial, TissueTopology, TwinAcceleratedTrainer, TwinConfig, TwinFeedbackProtocol, TwinLearningEvaluator, TwinProfile, TwinValidator, describe_twin_capabilities


def test_twin_mode_stim_causes_artifact_and_evoked_spikes(monkeypatch):
    """
    The biological twin must be bidirectional: a stimulation event should alter
    future simulated measurements, not just appear as bookkeeping in stims.
    """
    monkeypatch.setenv("CL_SDK_SIM_MODE", "surrogate")
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")
    monkeypatch.setenv("CL_SDK_TWIN_SEED", "7")
    monkeypatch.setenv("CL_SDK_TWIN_BASELINE_RATE_HZ", "0")
    monkeypatch.setenv("CL_SDK_TWIN_EVOKED_PROBABILITY", "1")
    monkeypatch.setenv("CL_SDK_TWIN_ARTIFACT_AMPLITUDE", "1800")

    with cl.open() as neurons:
        start_ts = neurons.timestamp()
        neurons.stim(27, cl.StimDesign(160, -2.0, 160, 2.0), lead_time_us=80)

        frames = neurons.read(200, start_ts)
        spikes = neurons._read_spikes(200, start_ts)
        stims = neurons._read_stims(start_ts, start_ts + 200)

    assert stims, "twin mode should still record delivered stimulation"
    assert spikes, "stimulation should causally evoke future spikes"
    assert all(spike.timestamp >= stims[0].timestamp for spike in spikes)
    assert max(abs(spike.channel - 27) for spike in spikes) < 40
    assert np.max(np.abs(frames[:, 27])) > 500


def test_replay_mode_remains_default(monkeypatch):
    """The twin backend is opt-in so existing simulator behavior is preserved."""
    monkeypatch.delenv("CL_SDK_SIM_MODE", raising=False)
    monkeypatch.setenv("CL_SDK_ACCELERATED_TIME", "1")

    with cl.open() as neurons:
        assert getattr(neurons, "_sim_mode") == "replay"


def test_twin_capability_report_tracks_enabled_modes():
    default_report = describe_twin_capabilities(TwinConfig())
    assert default_report.roadmap_count == 0
    assert any(
        capability.name == "Extracellular waveform detector" and capability.status == "implemented"
        for capability in default_report.capabilities
    )
    assert any(
        capability.name == "Izhikevich cell SNN" and capability.status == "approximated"
        for capability in default_report.capabilities
    )

    enabled_report = describe_twin_capabilities(
        TwinConfig(dynamics_mode="izhikevich", plasticity_mode="stdp")
    )
    assert any(
        capability.name == "Izhikevich cell SNN" and capability.status == "implemented"
        for capability in enabled_report.capabilities
    )
    assert any(
        capability.name == "STDP" and capability.status == "implemented"
        for capability in enabled_report.capabilities
    )

    large_report = describe_twin_capabilities(
        TwinConfig(dynamics_mode="izhikevich", snn_neuron_count=5_000)
    )
    assert any(
        capability.name == "Large-scale sparse SNN" and capability.status == "implemented"
        for capability in large_report.capabilities
    )


def test_rolling_spike_detector_extracts_waveforms_and_blanks_artifacts():
    detector = RollingThresholdSpikeDetector(
        frames_per_second=25_000,
        channel_count=2,
        threshold_sigma=4.5,
        refractory_frames=10,
        baseline_noise_std=np.array([5.0, 5.0]),
    )
    frames = np.zeros((120, 2), dtype=np.float64)
    frames[50, 0] = -80.0
    frames[51, 0] = -120.0
    frames[52, 0] = -70.0
    frames[80, 1] = -130.0

    spikes = detector.detect(
        frames,
        from_timestamp=1_000,
        blanking_windows=[
            DetectionBlankingWindow(
                start_timestamp=1_078,
                end_timestamp=1_083,
                channel=1,
            )
        ],
    )

    assert len(spikes) == 1
    assert spikes[0].timestamp == 1_051
    assert spikes[0].channel == 0
    assert len(spikes[0].samples) == 75
    assert np.min(spikes[0].samples) == -120.0


def test_pink_noise_has_temporal_correlation_and_profile_scale():
    pink = PinkNoiseState(
        channel_count=2,
        rng=np.random.default_rng(123),
        std_by_channel=np.array([3.0, 9.0]),
        color="pink",
    ).sample(2_000)
    white = PinkNoiseState(
        channel_count=2,
        rng=np.random.default_rng(123),
        std_by_channel=np.array([3.0, 9.0]),
        color="white",
    ).sample(2_000)

    pink_lag1 = np.corrcoef(pink[:-1, 0], pink[1:, 0])[0, 1]
    white_lag1 = np.corrcoef(white[:-1, 0], white[1:, 0])[0, 1]

    assert pink_lag1 > white_lag1 + 0.05
    assert np.std(pink[:, 1]) > np.std(pink[:, 0]) * 2.5


def test_twin_profile_serializes_and_shapes_model(tmp_path):
    """A culture profile should be a stable artifact that changes twin dynamics."""
    profile = TwinProfile.default()
    baseline = [0.0] * 64
    baseline[5] = 12.0
    noise = [1.0] * 64
    noise[5] = 9.0
    latency = [[0 for _ in range(64)] for _ in range(64)]
    response = [[0.0 for _ in range(64)] for _ in range(64)]
    response[27][27] = 1.0
    latency[27][27] = 64
    profile = TwinProfile(
        baseline_rate_hz_by_channel       = baseline,
        noise_std_sample_units_by_channel = noise,
        connectivity                      = profile.connectivity,
        stim_response_probability         = response,
        stim_response_latency_frames      = latency,
    )

    profile_path = tmp_path / "culture-profile.json"
    profile.save(profile_path)
    loaded = TwinProfile.load(profile_path)

    config = TwinConfig(
        seed=11,
        baseline_rate_hz=0.0,
        noise_std_sample_units=1.0,
        evoked_probability=1.0,
        evoked_jitter_frames=0,
        plasticity_mode="off",
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=config,
        profile=loaded,
    )

    assert model._baseline_rate_hz[5] == 12.0
    assert model._noise_std[5] == 9.0

    model.apply_stim(StimRecord(timestamp=10, channel=27), current_uA=2.0)
    _, spikes = model.render(0, 100)

    assert any(spike.channel == 27 and abs(spike.timestamp - 74) <= 3 for spike in spikes)


def test_twin_profile_migrates_older_schema(tmp_path):
    """Older profile JSON should load with new fields filled safely."""
    old_profile = {
        "schema_version": 1,
        "channel_count": 64,
        "frames_per_second": 25_000,
        "baseline_rate_hz_by_channel": [0.0] * 64,
        "noise_std_sample_units_by_channel": [1.0] * 64,
        "connectivity": TwinProfile.default().connectivity,
        "stim_response_probability": [[0.0 for _ in range(64)] for _ in range(64)],
        "stim_response_latency_frames": [[0 for _ in range(64)] for _ in range(64)],
        "experimental_extra": "ignored",
    }
    profile_path = tmp_path / "old-profile.json"
    profile_path.write_text(json.dumps(old_profile), encoding="utf-8")

    loaded = TwinProfile.load(profile_path)

    assert loaded.schema_version == TwinProfile.CURRENT_SCHEMA_VERSION
    assert loaded.isi_median_frames_by_channel == [0] * 64
    assert loaded.burst_rate_hz_by_channel == [0.0] * 64
    assert loaded.channel_confidence_by_channel == [0.0] * 64
    assert loaded.stim_response_count == [[0 for _ in range(64)] for _ in range(64)]
    assert loaded.stim_response_confidence[0][0] == 1.0
    assert loaded.topology_neuron_count == 0
    assert loaded.topology_channel_density == [1.0 / 64] * 64
    assert loaded.baseline_rate_ci95_hz_by_channel == [[0.0, 0.0] for _ in range(64)]
    assert loaded.stim_response_probability_ci95[0][0] == [0.0, 0.0]
    assert loaded.stim_response_latency_ci95_frames[0][0] == [0, 0]
    assert loaded.field_confidence["baseline_rate_hz"] == 0.0
    assert not hasattr(loaded, "experimental_extra")


def test_twin_profile_confidence_scores_are_bounded():
    """Calibration confidence should summarize evidence without escaping [0, 1]."""
    channel_confidence, field_confidence = TwinProfile._estimate_confidence(
        duration_sec = 120.0,
        spike_counts = np.array([0.0, 10.0, 100.0]),
        sample_count = 25_000,
        stim_count   = 40,
        burst_count  = np.array([0.0, 3.0, 10.0]),
    )

    assert channel_confidence[0] == 0.0
    assert channel_confidence[2] > channel_confidence[1] > channel_confidence[0]
    assert all(0.0 <= value <= 1.0 for value in channel_confidence)
    assert all(0.0 <= value <= 1.0 for value in field_confidence.values())
    assert field_confidence["noise_std_sample_units"] == 1.0
    assert field_confidence["stim_response"] > 0.0


def test_twin_profile_estimates_isi_and_burst_statistics():
    """Culture profiles should capture temporal spike structure, not just rates."""
    spike_frames_by_channel = {ch: np.array([], dtype=np.int64) for ch in range(64)}
    spike_frames_by_channel[3] = np.array([0, 10, 20, 1000, 1010, 1020], dtype=np.int64)

    isi, burst_rate, burst_duration, burst_count = TwinProfile._estimate_temporal_structure(
        spike_frames_by_channel = spike_frames_by_channel,
        channel_count           = 64,
        frames_per_second       = 25_000,
        duration_sec            = 2.0,
        burst_isi_frames        = 50,
        burst_min_spikes        = 3,
    )

    assert isi[3] == 10
    assert burst_rate[3] == 1.0
    assert burst_duration[3] == 20
    assert burst_count[3] == 3.0
    assert burst_rate[4] == 0.0


def test_twin_profile_estimates_pairwise_stim_response_confidence():
    """Stim response calibration should include pairwise hit support."""
    class RecordingStub:
        stims = [
            {"timestamp": 10, "channel": 2},
            {"timestamp": 50, "channel": 2},
            {"timestamp": 90, "channel": 2},
        ]

    spike_frames_by_channel = {ch: np.array([], dtype=np.int64) for ch in range(64)}
    spike_frames_by_channel[7] = np.array([15, 55, 300], dtype=np.int64)

    probability, probability_ci, latency, latency_ci, count, confidence = TwinProfile._estimate_stim_response(
        recording=RecordingStub(),
        spike_frames_by_channel=spike_frames_by_channel,
        channel_count=64,
        response_window_frames=20,
    )

    assert probability[2, 7] == 2 / 3
    assert latency[2, 7] == 5
    assert count[2, 7] == 2
    assert 0.0 < confidence[2, 7] < 1.0
    assert confidence[2, 8] == 0.0
    assert 0.0 < probability_ci[2][7][0] < probability[2, 7] < probability_ci[2][7][1] <= 1.0
    assert latency_ci[2][7][0] <= latency[2, 7] <= latency_ci[2][7][1]


def test_twin_profile_uncertainty_intervals_are_bounded():
    """Profile intervals should expose uncertainty without invalid ranges."""
    rate_ci = TwinProfile._poisson_rate_ci95(
        spike_counts=np.array([0.0, 25.0]),
        duration_sec=5.0,
    )
    probability_ci = TwinProfile._binomial_ci95(hit_count=3, trial_count=5)
    latency_ci = TwinProfile._latency_ci95([4, 5, 9])

    assert rate_ci[0] == [0.0, 0.0]
    assert rate_ci[1][0] <= 5.0 <= rate_ci[1][1]
    assert 0.0 <= probability_ci[0] <= 0.6 <= probability_ci[1] <= 1.0
    assert latency_ci[0] <= 5 <= latency_ci[1]


def test_twin_profile_estimates_topology_density_from_spikes():
    """Activity support should produce a normalized topology density prior."""
    spike_counts = np.zeros(64)
    spike_counts[7] = 100.0
    spike_counts[8] = 25.0

    neuron_count, density = TwinProfile._estimate_topology(
        spike_counts=spike_counts,
        channel_count=64,
        min_neuron_count=64,
        max_neuron_count=512,
    )

    assert neuron_count >= 64
    assert np.isclose(sum(density), 1.0)
    assert density[7] > density[8] > density[0]


def test_twin_profile_topology_uses_stim_and_connectivity_support():
    """Recruitable connected channels should influence topology beyond spike counts."""
    spike_counts = np.zeros(64)
    spike_counts[7] = 100.0
    stim_response = np.zeros((64, 64), dtype=float)
    stim_response[0, 40] = 1.0
    connectivity = np.eye(64, dtype=float)
    connectivity[40, 41] = 0.9
    connectivity[41, 40] = 0.9

    _, density = TwinProfile._estimate_topology(
        spike_counts=spike_counts,
        channel_count=64,
        min_neuron_count=64,
        max_neuron_count=512,
        stim_response_probability=stim_response,
        connectivity=connectivity,
    )

    assert np.isclose(sum(density), 1.0)
    assert density[40] > density[1]
    assert density[41] > density[1]


def test_twin_config_loads_profile_path_from_env(monkeypatch, tmp_path):
    """The opt-in profile path should flow through runtime configuration."""
    response = [[0.0 for _ in range(64)] for _ in range(64)]
    latency = [[0 for _ in range(64)] for _ in range(64)]
    response[27][27] = 1.0
    latency[27][27] = 50
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=response,
        stim_response_latency_frames=latency,
    )
    profile_path = tmp_path / "profile.json"
    profile.save(profile_path)

    monkeypatch.setenv("CL_SDK_TWIN_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("CL_SDK_TWIN_DYNAMICS", "population")
    monkeypatch.setenv("CL_SDK_TWIN_CULTURE_STATE", "synchronized_burst")
    monkeypatch.setenv("CL_SDK_TWIN_RECURRENT_COUPLING", "0.75")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_NEURON_COUNT", "128")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_CONNECTION_PROBABILITY", "0.5")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_LENGTH_CONSTANT_UM", "180")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_FIELD_GAMMA", "1.4")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STDP_LEARNING_RATE", "0.01")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STDP_TAU_FRAMES", "250")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STP_RECOVERY_FRAMES", "1200")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STP_FACILITATION_FRAMES", "450")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STP_DEPRESSION", "0.25")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_STP_FACILITATION", "0.12")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_REFRACTORY_FRAMES", "40")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_MIN_PROPAGATION_DELAY_FRAMES", "3")
    monkeypatch.setenv("CL_SDK_TWIN_SNN_MAX_PROPAGATION_DELAY_FRAMES", "30")
    config = TwinConfig.from_env()
    loaded = TwinProfile.load(config.profile_path)

    assert config.profile_path == str(profile_path)
    assert config.dynamics_mode == "population"
    assert config.culture_state == "synchronized_burst"
    assert config.recurrent_coupling == 0.75
    assert config.snn_neuron_count == 128
    assert config.snn_connection_probability == 0.5
    assert config.snn_length_constant_um == 180
    assert config.snn_field_gamma == 1.4
    assert config.snn_stdp_learning_rate == 0.01
    assert config.snn_stdp_tau_frames == 250
    assert config.snn_stp_recovery_frames == 1200
    assert config.snn_stp_facilitation_frames == 450
    assert config.snn_stp_depression == 0.25
    assert config.snn_stp_facilitation == 0.12
    assert config.snn_refractory_frames == 40
    assert config.snn_min_propagation_delay_frames == 3
    assert config.snn_max_propagation_delay_frames == 30
    assert loaded.stim_response_latency_frames[27][27] == 50


def test_plasticity_state_modulates_response_gain():
    """Plasticity is opt-in, bounded, and changes future channel responsiveness."""
    plasticity = PlasticityState(channel_count=64, mode="stdp", dopamine=1.0)
    coupling = np.zeros(64)
    coupling[10] = 1.0

    initial_gain = plasticity.response_gain[10]
    plasticity.on_stim(timestamp=100, coupling=coupling)
    depressed_gain = plasticity.response_gain[10]
    plasticity.on_spike(timestamp=120, channel=10)
    potentiated_gain = plasticity.response_gain[10]

    assert depressed_gain < initial_gain
    assert potentiated_gain > depressed_gain
    assert 0.25 <= potentiated_gain <= 3.0


def test_feedback_protocol_separates_structured_and_chaotic_patterns():
    """Closed-loop outcomes should compile into biologically distinct stims."""
    protocol = TwinFeedbackProtocol(channel_count=64, frames_per_second=25_000)

    structured = protocol.from_outcome(
        timestamp=1_000,
        correct=True,
        sensory_channel=3,
        motor_channel=11,
        current_uA=2.0,
    )
    chaotic = protocol.from_outcome(
        timestamp=1_000,
        correct=False,
        sensory_channel=3,
        motor_channel=11,
        current_uA=2.0,
    )

    assert [event.channel for event in structured] == [3, 11]
    assert structured[1].timestamp > structured[0].timestamp
    assert len(chaotic) == 64 * 3
    assert {event.channel for event in chaotic} == set(range(64))
    assert max(event.timestamp for event in chaotic) > min(event.timestamp for event in chaotic)


def test_surrogate_feedback_uses_normal_stim_coupling():
    """Feedback patterns should perturb the twin through the MEA stim pathway."""
    protocol = TwinFeedbackProtocol(channel_count=64, frames_per_second=25_000)
    config = TwinConfig(
        seed=29,
        baseline_rate_hz=0.0,
        evoked_probability=1.0,
        evoked_jitter_frames=0,
        artifact_amplitude_sample_units=900.0,
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=config,
    )

    model.apply_feedback(protocol.structured(
        timestamp=0,
        sensory_channel=2,
        motor_channel=10,
        current_uA=2.0,
    ))
    frames, spikes = model.render(0, 300)

    assert spikes
    assert np.max(np.abs(frames[:, 2])) > 400
    assert model._excitability[2] > config.excitability


def test_learning_evaluator_passes_improving_closed_loop_trials():
    """Task-level validation should detect closed-loop learning improvement."""
    trials = [
        TaskTrial(timestamp=i, correct=i >= 8, response_latency_frames=30 - i)
        for i in range(12)
    ]

    report = TwinLearningEvaluator.evaluate_trials(
        trials,
        window_size=4,
        min_accuracy_delta=0.5,
        max_latency_delta_frames=0.0,
    )

    assert report.passed
    assert report.early_accuracy == 0.0
    assert report.late_accuracy == 1.0
    assert report.accuracy_delta == 1.0
    assert report.late_latency_median_frames < report.early_latency_median_frames
    assert report.to_dict()["trial_count"] == 12


def test_learning_evaluator_fails_flat_closed_loop_trials():
    """A twin with no task-performance improvement should fail learning gates."""
    trials = [
        {"timestamp": i, "correct": i % 2 == 0, "response_latency_frames": 20}
        for i in range(12)
    ]

    report = TwinLearningEvaluator.evaluate_trials(
        trials,
        window_size=4,
        min_accuracy_delta=0.25,
    )

    assert not report.passed
    assert report.accuracy_delta == 0.0


def test_accelerated_trainer_runs_feedback_without_wall_clock_producer():
    """Offline training should advance the same model path without subprocess sleeps."""
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(
            seed=31,
            baseline_rate_hz=0.0,
            evoked_probability=1.0,
            evoked_jitter_frames=0,
        ),
    )
    trainer = TwinAcceleratedTrainer(
        model=model,
        protocol=TwinFeedbackProtocol(channel_count=64, frames_per_second=25_000),
        render_chunk_frames=40,
    )

    result = trainer.run_trials(
        [False, False, True, True],
        trial_interval_frames=80,
        sensory_channel=3,
        motor_channel=12,
        current_uA=1.5,
        response_latency_fn=lambda index, correct: 30 - index if correct else 40,
    )

    assert result.simulated_frames == 320
    assert result.trial_count == 4
    assert result.stim_count > 4
    assert result.learning_report.late_accuracy > result.learning_report.early_accuracy
    assert result.to_dict()["learning_report"]["trial_count"] == 4


def test_maturation_state_scales_low_and_mature_div_differently():
    """DIV should create a bounded developmental prior for culture dynamics."""
    immature = MaturationState.from_div(5)
    mature = MaturationState.from_div(21)

    assert immature.maturity == 0.0
    assert mature.maturity == 1.0
    assert immature.baseline_rate_scale < mature.baseline_rate_scale
    assert immature.connection_probability_scale < mature.connection_probability_scale
    assert immature.burst_rate_scale < mature.burst_rate_scale
    assert immature.propagation_delay_scale > mature.propagation_delay_scale


def test_culture_state_resolves_explicit_and_auto_regimes():
    """Culture state should expose quiescent, normal, and synchronized regimes."""
    quiescent = CultureState.from_config(
        requested_state="quiescent",
        gaba_block=0.0,
        excitability=1.0,
    )
    normal = CultureState.from_config(
        requested_state="normal",
        gaba_block=0.0,
        excitability=1.0,
    )
    synchronized = CultureState.from_config(
        requested_state="auto",
        gaba_block=0.9,
        excitability=1.0,
    )

    assert quiescent.name == "quiescent"
    assert normal.name == "normal"
    assert synchronized.name == "synchronized_burst"
    assert quiescent.baseline_rate_scale < normal.baseline_rate_scale
    assert synchronized.burst_rate_scale > normal.burst_rate_scale
    assert synchronized.network_burst_rate_hz > 0.0


def test_surrogate_applies_div_maturation_to_rates_and_snn():
    """Low-DIV twins should be sparser and less recruitable than mature twins."""
    default_profile = TwinProfile.default()
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[10.0] * 64,
        noise_std_sample_units_by_channel=default_profile.noise_std_sample_units_by_channel,
        connectivity=default_profile.connectivity,
        stim_response_probability=default_profile.stim_response_probability,
        stim_response_latency_frames=default_profile.stim_response_latency_frames,
        burst_rate_hz_by_channel=[4.0] * 64,
        burst_spike_count_mean_by_channel=[5.0] * 64,
    )

    immature = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=31, div=5, dynamics_mode="izhikevich", snn_connection_probability=0.2),
        profile=profile,
    )
    mature = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=31, div=21, dynamics_mode="izhikevich", snn_connection_probability=0.2),
        profile=profile,
    )

    assert immature._baseline_rate_hz[0] < mature._baseline_rate_hz[0]
    assert immature._burst_rate_hz[0] < mature._burst_rate_hz[0]
    assert immature._burst_spike_count_mean[0] < mature._burst_spike_count_mean[0]
    assert immature._snn.coupling < mature._snn.coupling
    assert np.count_nonzero(immature._snn._cell_synapses) < np.count_nonzero(mature._snn._cell_synapses)
    assert immature._snn.max_propagation_delay_frames > mature._snn.max_propagation_delay_frames


def test_surrogate_applies_culture_state_and_network_bursts():
    """Synchronized cultures should be more burst-prone than quiescent cultures."""
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[4.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        burst_rate_hz_by_channel=[1.0] * 64,
        burst_spike_count_mean_by_channel=[3.0] * 64,
    )
    quiescent = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=41, culture_state="quiescent"),
        profile=profile,
    )
    synchronized = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=41, culture_state="synchronized_burst"),
        profile=profile,
    )

    assert quiescent._baseline_rate_hz[0] < synchronized._baseline_rate_hz[0]
    assert quiescent._burst_rate_hz[0] < synchronized._burst_rate_hz[0]
    assert quiescent._dynamics.coupling < synchronized._dynamics.coupling

    for offset in range(20):
        synchronized._maybe_add_network_burst(offset * 25_000, (offset + 1) * 25_000)
        if synchronized._pending_spikes:
            break
    channels = {event.channel for event in synchronized._pending_spikes}
    assert len(channels) >= 32


def test_population_dynamics_propagates_through_connectivity():
    """A source population spike should recruit a connected target population."""
    rng = np.random.default_rng(1)
    connectivity = np.zeros((64, 64), dtype=float)
    connectivity[0, 63] = 1.0
    dynamics = PopulationDynamics(
        channel_count=64,
        connectivity=connectivity,
        mode="population",
        coupling=1.0,
        delay_frames=20,
        refractory_frames=1,
        rng=rng,
    )

    spikes = dynamics.on_spike(
        timestamp=30,
        channel=0,
        excitability=np.ones(64),
        response_gain=np.ones(64),
    )

    assert len(spikes) == 1
    assert spikes[0].channel == 63
    assert spikes[0].timestamp >= 50


def test_surrogate_uses_population_dynamics_for_recurrent_spikes():
    """The surrogate should move beyond evoked spikes into recurrent activity."""
    response = [[0.0 for _ in range(64)] for _ in range(64)]
    latency = [[0 for _ in range(64)] for _ in range(64)]
    connectivity = [[0.0 for _ in range(64)] for _ in range(64)]
    connectivity[0][63] = 1.0
    response[0][0] = 1.0
    latency[0][0] = 30
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=connectivity,
        stim_response_probability=response,
        stim_response_latency_frames=latency,
    )
    config = TwinConfig(
        seed=5,
        baseline_rate_hz=0.0,
        evoked_probability=1.0,
        evoked_jitter_frames=0,
        dynamics_mode="population",
        recurrent_coupling=1.0,
        propagation_delay_frames=20,
        refractory_frames=1,
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=config,
        profile=profile,
    )

    model.apply_stim(StimRecord(timestamp=0, channel=0), current_uA=1.0)
    _, first_spikes = model.render(0, 40)
    _, second_spikes = model.render(40, 80)

    assert any(spike.channel == 0 for spike in first_spikes)
    assert any(spike.channel == 63 for spike in second_spikes)


def test_surrogate_uses_profile_burst_statistics_for_background_spikes():
    """Profiled bursts should create packeted spontaneous spike events."""
    burst_rate = [0.0] * 64
    burst_duration = [0] * 64
    burst_count = [0.0] * 64
    burst_rate[5] = 1000.0
    burst_duration[5] = 12
    burst_count[5] = 4.0
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=[[0.0 for _ in range(64)] for _ in range(64)],
        stim_response_latency_frames=[[0 for _ in range(64)] for _ in range(64)],
        burst_rate_hz_by_channel=burst_rate,
        burst_median_duration_frames_by_channel=burst_duration,
        burst_spike_count_mean_by_channel=burst_count,
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=2, baseline_rate_hz=0.0),
        profile=profile,
    )
    model.rng = np.random.default_rng(2)

    model._maybe_add_background_burst(
        channel=5,
        from_timestamp=0,
        to_timestamp=100,
        duration_sec=100 / 25_000,
    )

    assert len(model._pending_spikes) == 4
    assert {event.channel for event in model._pending_spikes} == {5}
    assert max(event.timestamp for event in model._pending_spikes) - min(
        event.timestamp for event in model._pending_spikes
    ) == 12


def test_surrogate_attenuates_low_confidence_stim_responses():
    """Pairwise stim confidence should temper calibrated response boosts."""
    response = [[0.0 for _ in range(64)] for _ in range(64)]
    response[0][63] = 1.0
    confidence_low = [[1.0 for _ in range(64)] for _ in range(64)]
    confidence_high = [[1.0 for _ in range(64)] for _ in range(64)]
    confidence_low[0][63] = 0.0
    profile_low = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=response,
        stim_response_latency_frames=[[0 for _ in range(64)] for _ in range(64)],
        stim_response_confidence=confidence_low,
    )
    profile_high = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=response,
        stim_response_latency_frames=[[0 for _ in range(64)] for _ in range(64)],
        stim_response_confidence=confidence_high,
    )
    config = TwinConfig(seed=2, baseline_rate_hz=0.0, evoked_probability=0.0)
    low = SurrogateTwinModel(64, 25_000, config, profile_low)
    high = SurrogateTwinModel(64, 25_000, config, profile_high)

    low.apply_stim(StimRecord(timestamp=0, channel=0), current_uA=1.0)
    high.apply_stim(StimRecord(timestamp=0, channel=0), current_uA=1.0)
    low_target_artifact = next(event for event in low._artifacts if event.channel == 63)
    high_target_artifact = next(event for event in high._artifacts if event.channel == 63)

    assert high_target_artifact.amplitude > low_target_artifact.amplitude


def test_surrogate_uses_profile_topology_for_snn():
    """A profile density prior should shape the constructed SNN topology."""
    density = [0.001] * 64
    density[7] = 1.0
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=[[0.0 for _ in range(64)] for _ in range(64)],
        stim_response_latency_frames=[[0 for _ in range(64)] for _ in range(64)],
        topology_neuron_count=96,
        topology_channel_density=density,
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(seed=3, snn_neuron_count=64),
        profile=profile,
    )

    assert model._snn.neuron_count == 96
    assert np.count_nonzero(model._snn.neuron_channel == 7) > 24


def test_twin_validator_passes_matching_profile_spikes():
    """Validation should compare simulated activity against profile targets."""
    profile = TwinProfile(
        channel_count=4,
        frames_per_second=100,
        duration_frames=100,
        baseline_rate_hz_by_channel=[6.0, 0.0, 0.0, 0.0],
        isi_median_frames_by_channel=[10, 0, 0, 0],
        burst_rate_hz_by_channel=[2.0, 0.0, 0.0, 0.0],
        burst_median_duration_frames_by_channel=[20, 0, 0, 0],
        burst_spike_count_mean_by_channel=[3.0, 0.0, 0.0, 0.0],
    )
    spikes = [
        {"timestamp": ts, "channel": 0}
        for ts in [0, 10, 20, 50, 60, 70]
    ]

    report = TwinValidator.validate_spikes(profile=profile, spikes=spikes)

    assert report.passed
    assert report.rate_mae_hz == 0.0
    assert report.isi_median_mae_frames == 0.0
    assert report.burst_rate_mae_hz == 0.0
    assert report.stim_response_probability_mae == 0.0
    assert report.stim_response_latency_mae_frames == 0.0
    assert report.to_dict()["spike_count"] == 6


def test_twin_validator_checks_stim_triggered_responses():
    """Intervention validation should compare evoked probability and latency."""
    response = [[0.0 for _ in range(4)] for _ in range(4)]
    latency = [[0 for _ in range(4)] for _ in range(4)]
    response[1][2] = 1.0
    latency[1][2] = 5
    profile = TwinProfile(
        channel_count=4,
        frames_per_second=100,
        duration_frames=100,
        stim_response_probability=response,
        stim_response_latency_frames=latency,
    )
    stims = [
        {"timestamp": 10, "channel": 1},
        {"timestamp": 50, "channel": 1},
    ]
    spikes = [
        {"timestamp": 15, "channel": 2},
        {"timestamp": 55, "channel": 2},
    ]

    report = TwinValidator.validate_spikes(
        profile=profile,
        spikes=spikes,
        stims=stims,
        response_window_frames=10,
    )

    assert report.passed
    assert report.stim_response_probability_mae == 0.0
    assert report.stim_response_latency_mae_frames == 0.0
    assert report.metrics["stim_response_pairs_evaluated"] == 1
    assert report.metrics["simulated_stim_response_probability"][1][2] == 1.0
    assert report.metrics["simulated_stim_response_latency_frames"][1][2] == 5


def test_twin_validator_fails_bad_stim_triggered_responses():
    """A twin that misses calibrated intervention effects should fail gates."""
    response = [[0.0 for _ in range(4)] for _ in range(4)]
    latency = [[0 for _ in range(4)] for _ in range(4)]
    response[1][2] = 1.0
    latency[1][2] = 5
    profile = TwinProfile(
        channel_count=4,
        frames_per_second=100,
        duration_frames=100,
        stim_response_probability=response,
        stim_response_latency_frames=latency,
    )

    report = TwinValidator.validate_spikes(
        profile=profile,
        spikes=[{"timestamp": 80, "channel": 2}],
        stims=[{"timestamp": 10, "channel": 1}],
        response_window_frames=10,
        tolerances={
            "stim_response_probability_mae": 0.1,
            "stim_response_latency_mae_frames": 1.0,
        },
    )

    assert not report.passed
    assert report.stim_response_probability_mae > 0.1


def test_twin_validator_checks_artifact_blanking_windows():
    """Stim-adjacent raw frames should show a detectable artifact window."""
    profile = TwinProfile(channel_count=4, frames_per_second=100, duration_frames=100)
    frames = np.zeros((100, 4), dtype=np.int16)
    frames[10:15, 1] = 500

    report = TwinValidator.validate_spikes(
        profile=profile,
        spikes=[],
        stims=[{"timestamp": 10, "channel": 1}],
        raw_frames=frames,
        raw_frame_start=0,
        artifact_blank_window_frames=5,
        artifact_threshold_sample_units=100.0,
    )

    assert report.passed
    assert report.artifact_blank_fraction == 1.0
    assert report.artifact_peak_abs_sample_units == 500.0
    assert report.metrics["artifact_windows_evaluated"] == 1


def test_twin_validator_fails_missing_artifact_blanking():
    """A twin that omits realistic stim artifacts should fail artifact gates."""
    profile = TwinProfile(channel_count=4, frames_per_second=100, duration_frames=100)
    frames = np.zeros((100, 4), dtype=np.int16)

    report = TwinValidator.validate_spikes(
        profile=profile,
        spikes=[],
        stims=[{"timestamp": 10, "channel": 1}],
        raw_frames=frames,
        raw_frame_start=0,
        artifact_blank_window_frames=5,
        artifact_threshold_sample_units=100.0,
        tolerances={
            "artifact_blank_fraction_min": 1.0,
            "artifact_peak_abs_sample_units_min": 100.0,
        },
    )

    assert not report.passed
    assert report.artifact_blank_fraction == 0.0
    assert report.artifact_peak_abs_sample_units == 0.0


def test_twin_validator_fails_out_of_tolerance_spikes():
    """Validation gates should fail when simulation drifts from the profile."""
    profile = TwinProfile(
        channel_count=4,
        frames_per_second=100,
        duration_frames=100,
        baseline_rate_hz_by_channel=[6.0, 0.0, 0.0, 0.0],
        isi_median_frames_by_channel=[10, 0, 0, 0],
        burst_rate_hz_by_channel=[2.0, 0.0, 0.0, 0.0],
    )
    spikes = [{"timestamp": 0, "channel": 0}]

    report = TwinValidator.validate_spikes(
        profile=profile,
        spikes=spikes,
        tolerances={
            "rate_mae_hz": 0.1,
            "isi_median_mae_frames": 0.1,
            "burst_rate_mae_hz": 0.1,
        },
    )

    assert not report.passed
    assert report.rate_mae_hz > 0.1


def test_izhikevich_network_emits_mea_spikes_after_stim():
    """The cell-level SNN should convert stimulation into MEA-channel spikes."""
    rng = np.random.default_rng(2)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=128,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        coupling=2.0,
    )
    drive = np.zeros(64)
    drive[0] = 3.0
    network.apply_stim(channel=0, drive=drive)

    spikes = network.render(
        0,
        300,
        excitability=np.ones(64),
        response_gain=np.ones(64),
    )

    assert spikes
    assert all(0 <= spike.channel < 64 for spike in spikes)


def test_tissue_topology_maps_cells_and_electrode_fields():
    """Virtual cells should live in MEA space and receive spatial field drive."""
    rng = np.random.default_rng(12)
    mea = MEAGeometry()
    topology = TissueTopology.random(
        neuron_count=128,
        mea=mea,
        rng=rng,
        field_gamma=1.25,
    )

    assert topology.neuron_xy_um.shape == (128, 2)
    assert topology.electrode_to_neuron_attenuation.shape == (64, 128)
    assert topology.neuron_distance_um.shape == (128, 128)
    assert np.all((topology.neuron_channel >= 0) & (topology.neuron_channel < 64))
    assert np.allclose(topology.electrode_to_neuron_attenuation.max(axis=1), 1.0)


def test_tissue_topology_can_sample_from_channel_density():
    """Calibrated channel density should bias virtual cell placement."""
    rng = np.random.default_rng(17)
    density = np.zeros(64)
    density[0] = 1.0
    topology = TissueTopology.random(
        neuron_count=128,
        mea=MEAGeometry(),
        rng=rng,
        field_gamma=1.25,
        channel_density=density,
    )

    assert np.count_nonzero(topology.neuron_channel == 0) > 64


def test_tissue_synapse_sign_follows_presynaptic_cell_type():
    """Inhibitory cells should make negative outgoing synapses."""
    rng = np.random.default_rng(14)
    topology = TissueTopology.random(
        neuron_count=8,
        mea=MEAGeometry(),
        rng=rng,
        field_gamma=1.25,
    )
    excitatory_mask = np.array([True, False, True, False, True, False, True, False])

    synapses = topology.distance_decayed_synapses(
        rng=rng,
        excitatory_mask=excitatory_mask,
        connection_probability=1.0,
        length_constant_um=1_000_000.0,
    )

    assert np.all(synapses[:, excitatory_mask] >= 0.0)
    assert np.all(synapses[:, ~excitatory_mask] <= 0.0)
    assert np.count_nonzero(synapses[:, ~excitatory_mask]) > 0


def test_izhikevich_network_uses_spatial_synapses_and_field_coupling():
    """The SNN should expose spatial topology, field coupling, and cell synapses."""
    rng = np.random.default_rng(9)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=96,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        coupling=2.0,
        connection_probability=1.0,
        length_constant_um=10_000.0,
    )
    drive = np.zeros(64)
    drive[0] = 2.0
    network.apply_stim(channel=0, drive=drive)

    assert network.topology.neuron_xy_um.shape == (96, 2)
    assert network._cell_synapses.shape == (96, 96)
    assert np.count_nonzero(network._cell_synapses) > 0
    assert np.count_nonzero(network.input_current) > 0


def test_sparse_izhikevich_supports_large_cultures_without_dense_distance_matrix():
    """Large SNN mode should avoid dense N x N synapse and distance storage."""
    network = SparseIzhikevichNetwork(
        channel_count=64,
        neuron_count=1_500,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(24),
        coupling=2.0,
        connection_probability=0.03,
        length_constant_um=180.0,
        max_targets_per_source=12,
    )
    drive = np.zeros(64)
    drive[0] = 3.0
    network.apply_stim(channel=0, drive=drive)

    assert network.topology.neuron_xy_um.shape == (1_500, 2)
    assert network.topology.neuron_distance_um.shape == (0, 0)
    assert network.synapse_count > 0
    assert max(len(targets) for targets in network._graph.targets_by_source) <= 12
    assert np.count_nonzero(network.input_current) > 0


def test_surrogate_selects_sparse_snn_above_threshold():
    """Producer-facing model should switch to sparse SNN at large cell counts."""
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(
            seed=30,
            dynamics_mode="izhikevich",
            snn_neuron_count=1_100,
            snn_sparse_threshold=1_000,
            snn_max_targets_per_source=8,
            snn_connection_probability=0.02,
            baseline_rate_hz=0.0,
        ),
    )

    assert isinstance(model._snn, SparseIzhikevichNetwork)
    assert model._snn.topology.neuron_distance_um.shape == (0, 0)


def test_izhikevich_gaba_block_suppresses_inhibitory_synapses():
    """Virtual GABA blockade should scale down negative inhibitory weights."""
    topology_rng = np.random.default_rng(15)
    topology = TissueTopology.random(
        neuron_count=64,
        mea=MEAGeometry(),
        rng=topology_rng,
        field_gamma=1.25,
    )
    base = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(16),
        connection_probability=1.0,
        length_constant_um=10_000.0,
        topology=topology,
        gaba_block=0.0,
    )
    blocked = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(16),
        connection_probability=1.0,
        length_constant_um=10_000.0,
        topology=topology,
        gaba_block=1.0,
    )

    inhibitory = base._cell_synapses < 0.0
    assert np.any(inhibitory)
    assert np.all(blocked._cell_synapses[inhibitory] == 0.0)
    assert np.all(blocked._cell_synapses[base._cell_synapses > 0.0] > 0.0)


def test_izhikevich_stdp_changes_cell_synapses_but_preserves_bounds():
    """Cell-level STDP should adapt actual SNN synapses in opt-in modes."""
    rng = np.random.default_rng(4)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        coupling=1.0,
        connection_probability=0.0,
        plasticity_mode="stdp",
        stdp_learning_rate=0.02,
        stdp_tau_frames=100.0,
    )
    network._cell_synapses[:] = 0.0
    network._baseline_cell_synapses[:] = 0.0
    network._cell_synapses[1, 0] = 0.05
    network._baseline_cell_synapses[1, 0] = 0.05
    network._last_cell_spike_ts[0] = 90

    network._apply_stdp(np.array([1]), timestamp=100)

    assert network._cell_synapses[1, 0] > 0.05
    assert network._cell_synapses[1, 0] <= 0.25
    assert np.sign(network._cell_synapses[1, 0]) == 1


def test_izhikevich_dopamine_gates_stdp_learning_rate():
    """Virtual dopamine should act as a third-factor gate for cell STDP."""
    topology = TissueTopology.random(
        neuron_count=64,
        mea=MEAGeometry(channel_count=64),
        rng=np.random.default_rng(21),
    )
    base = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(22),
        connection_probability=0.0,
        plasticity_mode="stdp",
        stdp_learning_rate=0.01,
        stdp_tau_frames=100.0,
        topology=topology,
        dopamine=0.0,
    )
    modulated = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(23),
        connection_probability=0.0,
        plasticity_mode="stdp",
        stdp_learning_rate=0.01,
        stdp_tau_frames=100.0,
        topology=topology,
        dopamine=2.0,
    )
    for network in (base, modulated):
        network._cell_synapses[:] = 0.0
        network._baseline_cell_synapses[:] = 0.0
        network._cell_synapses[1, 0] = 0.05
        network._baseline_cell_synapses[1, 0] = 0.05
        network._last_cell_spike_ts[0] = 90

    base._apply_stdp(np.array([1]), timestamp=100)
    modulated._apply_stdp(np.array([1]), timestamp=100)

    assert modulated._effective_stdp_learning_rate() > base._effective_stdp_learning_rate()
    assert modulated._cell_synapses[1, 0] > base._cell_synapses[1, 0]


def test_izhikevich_stdp_off_keeps_cell_synapses_fixed():
    """Default SNN mode should remain deterministic with fixed cell synapses."""
    rng = np.random.default_rng(4)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        coupling=1.0,
        connection_probability=0.0,
        plasticity_mode="off",
    )
    network._cell_synapses[:] = 0.0
    network._baseline_cell_synapses[:] = 0.0
    network._cell_synapses[1, 0] = 0.05
    network._baseline_cell_synapses[1, 0] = 0.05
    network._last_cell_spike_ts[0] = 90

    network._apply_stdp(np.array([1]), timestamp=100)

    assert network._cell_synapses[1, 0] == 0.05


def test_izhikevich_stdp_does_not_adapt_inhibitory_synapses():
    """Timing plasticity should leave inhibitory SNN synapses under GABA control."""
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=np.random.default_rng(32),
        connection_probability=0.0,
        plasticity_mode="stdp",
        stdp_learning_rate=0.05,
        stdp_tau_frames=100.0,
    )
    network._cell_synapses[:] = 0.0
    network._baseline_cell_synapses[:] = 0.0
    network._cell_synapses[1, 0] = -0.05
    network._baseline_cell_synapses[1, 0] = -0.05
    network._last_cell_spike_ts[0] = 90

    network._apply_stdp(np.array([1]), timestamp=100)

    assert network._cell_synapses[1, 0] == -0.05


def test_izhikevich_stp_depresses_and_recovers_transmission():
    """Short-term plasticity should fatigue active sources and then recover."""
    rng = np.random.default_rng(8)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        connection_probability=0.0,
        plasticity_mode="stp",
        stp_recovery_frames=10.0,
        stp_facilitation_frames=5.0,
        stp_depression=0.40,
        stp_facilitation=0.10,
    )
    network._cell_synapses[:] = 0.0
    network._cell_synapses[1, 0] = 0.10

    baseline = network._effective_cell_recurrent(np.array([0]))[1]
    network._apply_stp(np.array([0]))
    fatigued = network._effective_cell_recurrent(np.array([0]))[1]

    assert network._stp_resources[0] < 1.0
    assert network._stp_facilitation[0] > 1.0
    assert fatigued < baseline

    resource_after_spike = network._stp_resources[0]
    facilitation_after_spike = network._stp_facilitation[0]
    for _ in range(20):
        network._recover_stp()

    assert network._stp_resources[0] > resource_after_spike
    assert network._stp_facilitation[0] < facilitation_after_spike
    assert network._stp_resources[0] <= 1.0
    assert network._stp_facilitation[0] >= 1.0


def test_izhikevich_stp_off_leaves_fast_synaptic_state_unchanged():
    """Default mode should not mutate the hidden STP state."""
    rng = np.random.default_rng(8)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        connection_probability=0.0,
        plasticity_mode="off",
    )

    network._apply_stp(np.array([0]))
    network._recover_stp()

    assert network._stp_resources[0] == 1.0
    assert network._stp_facilitation[0] == 1.0


def test_izhikevich_schedules_recurrent_current_with_delays():
    """Cell recurrent current should arrive through propagation delay buckets."""
    rng = np.random.default_rng(10)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        connection_probability=0.0,
        min_propagation_delay_frames=2,
        max_propagation_delay_frames=2,
    )
    network._cell_synapses[:] = 0.0
    network._cell_synapses[1, 0] = 0.10
    network._schedule_cell_recurrent(np.array([0]))

    assert network._synaptic_delay_queue[network._delay_cursor, 1] == 0.0
    delayed_bucket = (network._delay_cursor + 2) % len(network._synaptic_delay_queue)
    assert network._synaptic_delay_queue[delayed_bucket, 1] == 3.0


def test_izhikevich_refractory_filter_blocks_recent_spikes():
    """Cells should not emit repeated spikes inside the refractory window."""
    rng = np.random.default_rng(11)
    network = IzhikevichNetwork(
        channel_count=64,
        neuron_count=64,
        frames_per_second=25_000,
        connectivity=np.zeros((64, 64)),
        rng=rng,
        connection_probability=0.0,
        refractory_frames=10,
    )
    network._last_cell_spike_ts[0] = 95
    network._last_cell_spike_ts[1] = 80
    network.v[0] = 35.0
    network.v[1] = 35.0

    fired = network._refractory_filter(np.array([0, 1]), timestamp=100)

    assert fired.tolist() == [1]
    assert network.v[0] <= network.c[0]


def test_surrogate_can_use_izhikevich_snn_mode():
    """The surrogate should expose a real cell-level SNN mode behind the same API."""
    profile = TwinProfile(
        baseline_rate_hz_by_channel=[0.0] * 64,
        noise_std_sample_units_by_channel=[1.0] * 64,
        connectivity=TwinProfile.default().connectivity,
        stim_response_probability=[[0.0 for _ in range(64)] for _ in range(64)],
        stim_response_latency_frames=[[0 for _ in range(64)] for _ in range(64)],
    )
    config = TwinConfig(
        seed=13,
        baseline_rate_hz=0.0,
        evoked_probability=0.0,
        dynamics_mode="izhikevich",
        snn_neuron_count=128,
        snn_coupling=2.0,
        dopamine=1.5,
    )
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=config,
        profile=profile,
    )

    model.apply_stim(StimRecord(timestamp=0, channel=0), current_uA=3.0)
    _, spikes = model.render(0, 300)

    assert model._snn.dopamine == 1.5
    assert spikes
