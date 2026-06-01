from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mea import MEAGeometry
from .tissue import TissueTopology


@dataclass(frozen=True)
class SNNSpike:
    """MEA-level spike emitted by the cell-level Izhikevich engine."""

    timestamp: int
    channel: int
    strength: float


class IzhikevichNetwork:
    """
    Small cell-level spiking neural network for the biological twin.

    The network simulates virtual neurons with the Izhikevich equations and
    projects their spikes back onto the nearest MEA channel.  It is intentionally
    compact by default so it can run inside the SDK producer process, but the
    API is the same shape a larger GPU-backed implementation would use later.
    """

    def __init__(
        self,
        *,
        channel_count: int,
        neuron_count: int,
        frames_per_second: int,
        connectivity: np.ndarray,
        rng: np.random.Generator,
        excitatory_fraction: float = 0.8,
        coupling: float = 1.0,
        field_gamma: float = 1.25,
        connection_probability: float = 0.08,
        length_constant_um: float = 250.0,
        plasticity_mode: str = "off",
        stdp_learning_rate: float = 0.004,
        stdp_tau_frames: float = 500.0,
        stp_recovery_frames: float = 1500.0,
        stp_facilitation_frames: float = 500.0,
        stp_depression: float = 0.15,
        stp_facilitation: float = 0.08,
        refractory_frames: int = 25,
        min_propagation_delay_frames: int = 1,
        max_propagation_delay_frames: int = 25,
        gaba_block: float = 0.0,
        dopamine: float = 0.0,
        topology: TissueTopology | None = None,
    ):
        self.channel_count = channel_count
        self.neuron_count = max(channel_count, int(neuron_count))
        self.frames_per_second = frames_per_second
        self.rng = rng
        self.coupling = max(0.0, float(coupling))
        self.plasticity_mode = plasticity_mode.lower()
        self.stdp_learning_rate = max(0.0, float(stdp_learning_rate))
        self.stdp_tau_frames = max(1.0, float(stdp_tau_frames))
        self.stp_recovery_frames = max(1.0, float(stp_recovery_frames))
        self.stp_facilitation_frames = max(1.0, float(stp_facilitation_frames))
        self.stp_depression = float(np.clip(stp_depression, 0.0, 0.95))
        self.stp_facilitation = max(0.0, float(stp_facilitation))
        self.refractory_frames = max(0, int(refractory_frames))
        self.min_propagation_delay_frames = max(0, int(min_propagation_delay_frames))
        self.max_propagation_delay_frames = max(
            self.min_propagation_delay_frames,
            int(max_propagation_delay_frames),
        )
        self.gaba_block = float(np.clip(gaba_block, 0.0, 1.0))
        self.dopamine = float(np.clip(dopamine, 0.0, 5.0))

        self.is_excitatory = self.rng.random(self.neuron_count) < np.clip(excitatory_fraction, 0.0, 1.0)
        self.topology = topology or TissueTopology.random(
            neuron_count = self.neuron_count,
            mea          = MEAGeometry(channel_count=channel_count),
            rng          = self.rng,
            field_gamma  = field_gamma,
        )
        self.neuron_channel = self.topology.neuron_channel

        self.a = np.where(self.is_excitatory, 0.02, 0.10)
        self.b = np.where(self.is_excitatory, 0.20, 0.20)
        self.c = np.where(self.is_excitatory, -65.0, -65.0)
        self.d = np.where(self.is_excitatory, 8.0, 2.0)
        self.v = self.rng.normal(-65.0, 3.0, self.neuron_count)
        self.u = self.b * self.v
        self.input_current = np.zeros(self.neuron_count, dtype=np.float64)
        self.synaptic_current = np.zeros(self.neuron_count, dtype=np.float64)
        self._channel_connectivity = self._prepare_connectivity(connectivity)
        self._cell_synapses = self.topology.distance_decayed_synapses(
            rng                    = self.rng,
            excitatory_mask        = self.is_excitatory,
            connection_probability = connection_probability,
            length_constant_um     = length_constant_um,
        )
        self._cell_synapses[self._cell_synapses < 0.0] *= 1.0 - self.gaba_block
        self._baseline_cell_synapses = self._cell_synapses.copy()
        self._last_cell_spike_ts = np.full(self.neuron_count, -1_000_000_000, dtype=np.int64)
        self._stp_resources = np.ones(self.neuron_count, dtype=np.float64)
        self._stp_facilitation = np.ones(self.neuron_count, dtype=np.float64)
        self._cell_delay_frames = self._build_delay_matrix()
        self._synaptic_delay_queue = np.zeros(
            (self.max_propagation_delay_frames + 1, self.neuron_count),
            dtype=np.float64,
        )
        self._delay_cursor = 0

    def apply_stim(self, channel: int, drive: np.ndarray) -> None:
        """Inject electrode stimulation into nearby virtual cells."""
        channel_drive = np.asarray(drive, dtype=np.float64)
        if channel_drive.shape != (self.channel_count,):
            channel_drive = np.zeros(self.channel_count, dtype=np.float64)
            channel_drive[int(channel)] = 1.0
        # Electrode current is projected into virtual cells through a spatial
        # field attenuation matrix.  A scale of 18 places a strong SDK stim
        # above threshold without causing permanent saturation.
        cell_drive = channel_drive @ self.topology.electrode_to_neuron_attenuation
        self.input_current += 18.0 * cell_drive * self.coupling

    def render(
        self,
        from_timestamp: int,
        frame_count: int,
        *,
        excitability: np.ndarray,
        response_gain: np.ndarray,
    ) -> list[SNNSpike]:
        """
        Advance the SNN for ``frame_count`` 25 kHz frames and return MEA spikes.

        The numerical step is the CL frame duration in milliseconds.  This is
        smaller than the common 1 ms Izhikevich examples, which improves
        stability for the simulator's 25 kHz frame clock.
        """
        dt_ms = 1000.0 / self.frames_per_second
        spikes: list[SNNSpike] = []
        # Dopamine is modeled as a bounded third-factor modulator: it makes
        # cells slightly easier to recruit without replacing the explicit
        # electrode/profiling gains that still dominate stimulation behavior.
        dopamine_excitability = 1.0 + 0.15 * self.dopamine
        gain_by_cell = (
            response_gain[self.neuron_channel]
            * excitability[self.neuron_channel]
            * dopamine_excitability
        )

        for offset in range(frame_count):
            timestamp = from_timestamp + offset
            self.synaptic_current += self._synaptic_delay_queue[self._delay_cursor]
            self._synaptic_delay_queue[self._delay_cursor].fill(0.0)

            noise = self.rng.normal(0.0, 0.5, self.neuron_count)
            current = self.input_current * gain_by_cell + self.synaptic_current + noise

            dv = 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current
            du = self.a * (self.b * self.v - self.u)
            self.v += dt_ms * dv
            self.u += dt_ms * du

            threshold_crossings = np.flatnonzero(self.v >= 30.0)
            fired = self._refractory_filter(threshold_crossings, timestamp)
            if fired.size:
                for neuron in fired.tolist():
                    channel = int(self.neuron_channel[neuron])
                    spikes.append(SNNSpike(
                        timestamp = timestamp,
                        channel   = channel,
                        strength  = 1.0,
                    ))

                fired_channels = self.neuron_channel[fired]
                channel_counts = np.bincount(fired_channels, minlength=self.channel_count)
                channel_recurrent = self._channel_connectivity.T @ channel_counts.astype(np.float64)
                self._schedule_cell_recurrent(fired)
                self.synaptic_current += 3.0 * channel_recurrent[self.neuron_channel]
                self._apply_stp(fired)
                self._apply_stdp(fired, timestamp)
                self._last_cell_spike_ts[fired] = timestamp

                self.v[fired] = self.c[fired]
                self.u[fired] += self.d[fired]

            # Synaptic and stimulation drive decays over tens of frames, giving
            # the network memory without letting one stim dominate forever.
            self.input_current *= 0.92
            self.synaptic_current *= 0.90
            self._recover_stp()
            self._delay_cursor = (self._delay_cursor + 1) % len(self._synaptic_delay_queue)

        return spikes

    def _refractory_filter(self, threshold_crossings: np.ndarray, timestamp: int) -> np.ndarray:
        """Gate threshold crossings through a per-cell refractory interval."""
        if threshold_crossings.size == 0:
            return threshold_crossings
        can_fire = (
            timestamp - self._last_cell_spike_ts[threshold_crossings]
            >= self.refractory_frames
        )
        fired = threshold_crossings[can_fire]
        refractory = threshold_crossings[~can_fire]
        if refractory.size:
            self.v[refractory] = np.minimum(self.v[refractory], self.c[refractory])
        return fired

    def _stp_enabled(self) -> bool:
        """Return whether short-term synaptic plasticity should alter transmission."""
        return self.plasticity_mode in {"stp", "stdp", "stdp_homeostatic", "stp_stdp"}

    def _effective_cell_recurrent(self, fired: np.ndarray) -> np.ndarray:
        """Return recurrent cell current after presynaptic STP modulation."""
        if fired.size == 0:
            return np.zeros(self.neuron_count, dtype=np.float64)
        source_gain = self._stp_resources[fired] * self._stp_facilitation[fired]
        return (self._cell_synapses[:, fired] * source_gain[np.newaxis, :]).sum(axis=1)

    def _schedule_cell_recurrent(self, fired: np.ndarray) -> None:
        """Schedule cell-to-cell recurrent current through axonal delay buckets."""
        if fired.size == 0:
            return
        source_gain = self._stp_resources[fired] * self._stp_facilitation[fired]
        for source_index, source in enumerate(fired.tolist()):
            weights = self._cell_synapses[:, source] * source_gain[source_index]
            if not np.any(weights):
                continue
            delays = self._cell_delay_frames[:, source]
            for delay in np.unique(delays[weights != 0.0]):
                targets = (delays == delay) & (weights != 0.0)
                bucket = (self._delay_cursor + int(delay)) % len(self._synaptic_delay_queue)
                self._synaptic_delay_queue[bucket, targets] += 30.0 * weights[targets]

    def _apply_stp(self, fired: np.ndarray) -> None:
        """
        Apply vesicle-style depression and facilitation to fired presynaptic cells.

        Resources represent the readily releasable pool and recover toward one.
        Facilitation represents transient release probability and also decays
        toward one.  The product modulates outgoing synapses during the next
        recurrent transmission event.
        """
        if not self._stp_enabled() or fired.size == 0:
            return
        self._stp_resources[fired] *= 1.0 - self.stp_depression
        self._stp_resources[fired] = np.clip(self._stp_resources[fired], 0.05, 1.0)
        self._stp_facilitation[fired] += self.stp_facilitation * (3.0 - self._stp_facilitation[fired])
        self._stp_facilitation[fired] = np.clip(self._stp_facilitation[fired], 1.0, 3.0)

    def _recover_stp(self) -> None:
        """Relax short-term depression and facilitation back to baseline."""
        if not self._stp_enabled():
            return
        self._stp_resources += (1.0 - self._stp_resources) / self.stp_recovery_frames
        self._stp_facilitation += (1.0 - self._stp_facilitation) / self.stp_facilitation_frames
        self._stp_resources = np.clip(self._stp_resources, 0.05, 1.0)
        self._stp_facilitation = np.clip(self._stp_facilitation, 1.0, 3.0)

    def _apply_stdp(self, fired: np.ndarray, timestamp: int) -> None:
        """
        Apply bounded spike-timing-dependent plasticity to cell synapses.

        Synapses are stored as ``target, source`` weights.  If a source fired
        before a postsynaptic target, positive excitatory weights potentiate. If
        the postsynaptic target fired before the source, they depress.  Inhibitory
        signs are preserved by adapting magnitudes and clipping around the
        original sign.
        """
        if self.plasticity_mode not in {"stdp", "stdp_homeostatic"}:
            return
        learning_rate = self._effective_stdp_learning_rate()
        if learning_rate <= 0.0 or fired.size == 0:
            return

        for post in fired.tolist():
            incoming = self._cell_synapses[post]
            incoming_mask = incoming > 0.0
            if np.any(incoming_mask):
                delta_pre = timestamp - self._last_cell_spike_ts
                causal = incoming_mask & (delta_pre >= 0) & (delta_pre <= self.stdp_tau_frames)
                if np.any(causal):
                    incoming[causal] += (
                        np.sign(incoming[causal])
                        * learning_rate
                        * np.exp(-delta_pre[causal] / self.stdp_tau_frames)
                    )

            outgoing = self._cell_synapses[:, post]
            outgoing_mask = outgoing > 0.0
            if np.any(outgoing_mask):
                delta_post = timestamp - self._last_cell_spike_ts
                anti_causal = outgoing_mask & (delta_post >= 0) & (delta_post <= self.stdp_tau_frames)
                if np.any(anti_causal):
                    outgoing[anti_causal] -= (
                        np.sign(outgoing[anti_causal])
                        * learning_rate
                        * 1.05
                        * np.exp(-delta_post[anti_causal] / self.stdp_tau_frames)
                    )

        if self.plasticity_mode == "stdp_homeostatic":
            current_norm = np.maximum(np.abs(self._cell_synapses).sum(axis=0, keepdims=True), 1e-12)
            baseline_norm = np.maximum(np.abs(self._baseline_cell_synapses).sum(axis=0, keepdims=True), 1e-12)
            self._cell_synapses *= baseline_norm / current_norm

        # Preserve sign and keep weights in a biologically modest range.
        signs = np.sign(self._baseline_cell_synapses)
        magnitudes = np.clip(np.abs(self._cell_synapses), 0.0, 0.25)
        self._cell_synapses = magnitudes * signs

    def _effective_stdp_learning_rate(self) -> float:
        """
        Return the current long-term plasticity step size.

        Dopamine is the SNN's third-factor gate: identical pre/post spike timing
        produces larger updates under elevated virtual dopamine, matching the
        wetware-training idea that reward chemistry controls whether timing
        correlations are written into synapses.
        """
        return self.stdp_learning_rate * (1.0 + self.dopamine)

    def _prepare_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """Return bounded excitatory/inhibitory population connectivity."""
        matrix = np.asarray(connectivity, dtype=np.float64)
        if matrix.shape != (self.channel_count, self.channel_count):
            matrix = np.eye(self.channel_count, dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.clip(matrix, -1.0, 1.0)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _build_delay_matrix(self) -> np.ndarray:
        """Convert cell distances into integer frame delays for recurrent spikes."""
        if self.max_propagation_delay_frames <= self.min_propagation_delay_frames:
            delays = np.full(
                (self.neuron_count, self.neuron_count),
                self.min_propagation_delay_frames,
                dtype=np.int64,
            )
            np.fill_diagonal(delays, 0)
            return delays

        max_distance = max(float(np.max(self.topology.neuron_distance_um)), 1.0)
        distance_fraction = self.topology.neuron_distance_um / max_distance
        delay_span = self.max_propagation_delay_frames - self.min_propagation_delay_frames
        delays = self.min_propagation_delay_frames + np.rint(distance_fraction * delay_span)
        delays = np.clip(delays, self.min_propagation_delay_frames, self.max_propagation_delay_frames)
        np.fill_diagonal(delays, 0.0)
        return delays.astype(np.int64)
