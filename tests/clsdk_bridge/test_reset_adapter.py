import numpy as np

from cl._data_buffer import StimRecord
from cl.twin import SurrogateTwinModel, TwinConfig, TwinProfile
from cl1_clsdk_bridge import ResetSNNAdapter


def test_reset_snn_adapter_uses_cl_channel_interface():
    config = TwinConfig(
        seed=7,
        baseline_rate_hz=0.0,
        snn_neuron_count=160,
        snn_max_targets_per_source=16,
        snn_coupling=2.0,
    )
    adapter = ResetSNNAdapter(
        channel_count=64,
        neuron_count=160,
        frames_per_second=25_000,
        rng=np.random.default_rng(7),
        config=config,
    )
    adapter.apply_timed_stim(10, 8, np.ones(64) * 4.0)
    spikes = adapter.render(
        0,
        500,
        excitability=np.ones(64),
        response_gain=np.ones(64),
    )

    assert adapter.synapse_count > 0
    assert all(0 <= spike.channel < 64 for spike in spikes)


def test_surrogate_selects_reset_snn_mode():
    model = SurrogateTwinModel(
        channel_count=64,
        frames_per_second=25_000,
        config=TwinConfig(
            seed=9,
            baseline_rate_hz=0.0,
            evoked_probability=0.0,
            dynamics_mode="snn_reset",
            snn_neuron_count=128,
            snn_max_targets_per_source=12,
            snn_coupling=2.0,
        ),
        profile=TwinProfile.default(),
    )

    model.apply_stim(StimRecord(timestamp=0, channel=8), current_uA=4.0)
    _, spikes = model.render(0, 600)

    assert isinstance(model._snn, ResetSNNAdapter)
    assert all(0 <= spike.channel < 64 for spike in spikes)
