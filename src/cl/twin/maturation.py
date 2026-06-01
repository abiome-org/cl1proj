from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaturationState:
    """
    Bounded culture-maturation scalars derived from days in vitro.

    The values are deliberately coarse: DIV is a biological condition knob, not
    a substitute for fitting a real culture profile.  Low-DIV cultures are sparse
    and weakly coupled; mature cultures have stronger recurrent coupling and a
    higher chance of synchronized burst packets.
    """

    div: int
    maturity: float
    baseline_rate_scale: float
    excitability_scale: float
    stim_coupling_scale: float
    burst_rate_scale: float
    burst_count_scale: float
    connection_probability_scale: float
    recurrent_coupling_scale: float
    propagation_delay_scale: float

    @classmethod
    def from_div(cls, div: int) -> "MaturationState":
        """Convert days in vitro into smooth, bounded developmental scalars."""
        resolved_div = max(0, int(div))
        maturity = float(np.clip((resolved_div - 5.0) / 16.0, 0.0, 1.0))
        return cls(
            div                          = resolved_div,
            maturity                     = maturity,
            baseline_rate_scale          = 0.20 + 0.80 * maturity,
            excitability_scale           = 0.55 + 0.45 * maturity,
            stim_coupling_scale          = 0.60 + 0.40 * maturity,
            burst_rate_scale             = 0.05 + 0.95 * maturity * maturity,
            burst_count_scale            = 0.35 + 0.65 * maturity,
            connection_probability_scale = 0.25 + 0.75 * maturity,
            recurrent_coupling_scale     = 0.30 + 0.70 * maturity,
            # Lower-DIV cultures have less mature conduction/myelination, so
            # recurrent events propagate more slowly in the coarse simulator.
            propagation_delay_scale      = 1.50 - 0.50 * maturity,
        )


@dataclass(frozen=True)
class CultureState:
    """
    Coarse culture-level regime for the biological twin.

    Maturation describes slow development.  Culture state describes the current
    dynamical regime: quiescent tissue, normal activity, or synchronized
    network-wide bursting.  The scales are intentionally bounded so pharmacology
    and state changes alter the simulator without making it numerically wild.
    """

    name: str
    baseline_rate_scale: float
    excitability_scale: float
    stim_coupling_scale: float
    burst_rate_scale: float
    burst_count_scale: float
    recurrent_coupling_scale: float
    network_burst_rate_hz: float
    network_burst_channel_fraction: float

    @classmethod
    def from_config(
        cls,
        *,
        requested_state: str,
        gaba_block: float,
        excitability: float,
    ) -> "CultureState":
        """Resolve configured or inferred culture state into runtime scales."""
        state = (requested_state or "normal").strip().lower()
        if state == "auto":
            if float(gaba_block) >= 0.65:
                state = "synchronized_burst"
            elif float(excitability) <= 0.35:
                state = "quiescent"
            else:
                state = "normal"
        aliases = {
            "sync": "synchronized_burst",
            "synchronized": "synchronized_burst",
            "bursting": "synchronized_burst",
            "epileptiform": "synchronized_burst",
            "silent": "quiescent",
            "quiet": "quiescent",
        }
        state = aliases.get(state, state)
        if state == "quiescent":
            return cls(
                name                           = state,
                baseline_rate_scale            = 0.12,
                excitability_scale             = 0.45,
                stim_coupling_scale            = 0.65,
                burst_rate_scale               = 0.10,
                burst_count_scale              = 0.50,
                recurrent_coupling_scale       = 0.35,
                network_burst_rate_hz          = 0.0,
                network_burst_channel_fraction = 0.0,
            )
        if state == "synchronized_burst":
            return cls(
                name                           = state,
                baseline_rate_scale            = 1.35,
                excitability_scale             = 1.35,
                stim_coupling_scale            = 1.10,
                burst_rate_scale               = 3.00,
                burst_count_scale              = 1.80,
                recurrent_coupling_scale       = 1.60,
                network_burst_rate_hz          = 2.0,
                network_burst_channel_fraction = 0.75,
            )
        return cls(
            name                           = "normal",
            baseline_rate_scale            = 1.0,
            excitability_scale             = 1.0,
            stim_coupling_scale            = 1.0,
            burst_rate_scale               = 1.0,
            burst_count_scale              = 1.0,
            recurrent_coupling_scale       = 1.0,
            network_burst_rate_hz          = 0.0,
            network_burst_channel_fraction = 0.0,
        )
