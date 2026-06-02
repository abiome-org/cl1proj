from __future__ import annotations

from typing import TYPE_CHECKING

from cl1_snn_reset import CultureConfig

from .dynamics import is_reset_dynamics, reset_backend_for_mode

if TYPE_CHECKING:
    from cl.twin.config import TwinConfig


def culture_config_from_twin(
    config: TwinConfig,
    *,
    channel_count: int,
    neuron_count: int,
) -> CultureConfig:
    mode = config.dynamics_mode.lower()
    if is_reset_dynamics(mode) and mode == "brian2_reset":
        backend = reset_backend_for_mode(mode)
    else:
        backend = config.snn_reset_backend
    return CultureConfig(
        n_neurons                  = max(channel_count, int(neuron_count)),
        excitatory_fraction        = config.snn_excitatory_fraction,
        field_size_mm              = 3.0,
        n_electrodes               = channel_count,
        connection_length_mm       = max(0.03, config.snn_length_constant_um / 1000.0),
        long_range_prob            = 0.02,
        mean_out_degree            = min(config.snn_max_targets_per_source, 64),
        max_out_degree             = max(8, config.snn_max_targets_per_source),
        background_noise_mv        = 1.0,
        spontaneous_rate_hz        = max(0.0, config.baseline_rate_hz),
        stim_gain_mv_per_uA        = 4.8 * config.snn_coupling,
        backend                    = backend,
    )
