from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .izhikevich import SNNSpike
from .mea import MEAGeometry
from .tissue import TissueTopology


@dataclass(frozen=True)
class SparseSynapseGraph:
    """Adjacency-list synapse graph for large virtual cultures."""

    targets_by_source: tuple[np.ndarray, ...]
    weights_by_source: tuple[np.ndarray, ...]
    signs_by_source: tuple[np.ndarray, ...]
    delays_by_source: tuple[np.ndarray, ...]
    sources_by_target: tuple[np.ndarray, ...]
    edge_indices_by_target: tuple[np.ndarray, ...]

    @property
    def edge_count(self) -> int:
        return int(sum(len(targets) for targets in self.targets_by_source))


class SparseIzhikevichNetwork:
    """
    Large-culture Izhikevich SNN using sparse recurrent synapses.

    This engine supports thousands of cells laid out over the MEA with
    distance-decayed E/I synapses, propagation delays, STP, and bounded STDP.
    It deliberately shares the compact SNN's public methods so the producer can
    choose a dense or sparse implementation without changing application-facing
    SDK calls.
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
        max_targets_per_source: int = 64,
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
        self.max_targets_per_source = max(1, int(max_targets_per_source))

        self.is_excitatory = self.rng.random(self.neuron_count) < np.clip(excitatory_fraction, 0.0, 1.0)
        self.topology = topology or TissueTopology.random(
            neuron_count                = self.neuron_count,
            mea                         = MEAGeometry(channel_count=channel_count),
            rng                         = self.rng,
            field_gamma                 = field_gamma,
            compute_pairwise_distances  = False,
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
        self._last_cell_spike_ts = np.full(self.neuron_count, -1_000_000_000, dtype=np.int64)
        self._stp_resources = np.ones(self.neuron_count, dtype=np.float64)
        self._stp_facilitation = np.ones(self.neuron_count, dtype=np.float64)
        self._synaptic_delay_queue = np.zeros(
            (self.max_propagation_delay_frames + 1, self.neuron_count),
            dtype=np.float64,
        )
        self._delay_cursor = 0
        self._graph = self._build_sparse_graph(
            connection_probability = connection_probability,
            length_constant_um     = length_constant_um,
        )

    @property
    def synapse_count(self) -> int:
        """Number of recurrent cell synapses currently stored."""
        return self._graph.edge_count

    def apply_stim(self, channel: int, drive: np.ndarray) -> None:
        """Inject electrode stimulation into nearby virtual cells."""
        channel_drive = np.asarray(drive, dtype=np.float64)
        if channel_drive.shape != (self.channel_count,):
            channel_drive = np.zeros(self.channel_count, dtype=np.float64)
            channel_drive[int(channel)] = 1.0
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
        """Advance the sparse SNN and return MEA-channel spikes."""
        dt_ms = 1000.0 / self.frames_per_second
        spikes: list[SNNSpike] = []
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

            fired = self._refractory_filter(np.flatnonzero(self.v >= 30.0), timestamp)
            if fired.size:
                for neuron in fired.tolist():
                    spikes.append(SNNSpike(
                        timestamp = timestamp,
                        channel   = int(self.neuron_channel[neuron]),
                        strength  = 1.0,
                    ))
                fired_channels = self.neuron_channel[fired]
                channel_counts = np.bincount(fired_channels, minlength=self.channel_count)
                channel_recurrent = self._channel_connectivity.T @ channel_counts.astype(np.float64)
                self._schedule_sparse_recurrent(fired)
                self.synaptic_current += 3.0 * channel_recurrent[self.neuron_channel]
                self._apply_stp(fired)
                self._apply_stdp(fired, timestamp)
                self._last_cell_spike_ts[fired] = timestamp
                self.v[fired] = self.c[fired]
                self.u[fired] += self.d[fired]

            self.input_current *= 0.92
            self.synaptic_current *= 0.90
            self._recover_stp()
            self._delay_cursor = (self._delay_cursor + 1) % len(self._synaptic_delay_queue)

        return spikes

    def _build_sparse_graph(
        self,
        *,
        connection_probability: float,
        length_constant_um: float,
    ) -> SparseSynapseGraph:
        """Sample distance-decayed outgoing synapses without an N x N matrix."""
        targets_by_source: list[np.ndarray] = []
        weights_by_source: list[np.ndarray] = []
        delays_by_source: list[np.ndarray] = []
        incoming_sources: list[list[int]] = [[] for _ in range(self.neuron_count)]
        incoming_edge_indices: list[list[int]] = [[] for _ in range(self.neuron_count)]
        xy = self.topology.neuron_xy_um
        max_distance = max(float(np.sqrt(2.0) * MEAGeometry().tissue_size_um), 1.0)
        delay_span = self.max_propagation_delay_frames - self.min_propagation_delay_frames
        for source in range(self.neuron_count):
            distances = np.linalg.norm(xy - xy[source], axis=1)
            probability = connection_probability * np.exp(-distances / max(length_constant_um, 1.0))
            probability[source] = 0.0
            connected = np.flatnonzero(self.rng.random(self.neuron_count) < probability)
            if connected.size > self.max_targets_per_source:
                # Prefer local targets when the probability field is dense. This
                # preserves spatial biology while bounding per-spike work.
                local_order = np.argsort(distances[connected])
                connected = connected[local_order[:self.max_targets_per_source]]
            weights = self.rng.uniform(0.02, 0.12, size=connected.size)
            if not self.is_excitatory[source]:
                weights *= -(1.0 - self.gaba_block)
            distance_fraction = distances[connected] / max_distance
            delays = self.min_propagation_delay_frames + np.rint(distance_fraction * delay_span)
            delays = np.clip(delays, self.min_propagation_delay_frames, self.max_propagation_delay_frames)
            targets_by_source.append(connected.astype(np.int64))
            weights_by_source.append(weights.astype(np.float64))
            delays_by_source.append(delays.astype(np.int64))
            for edge_index, target in enumerate(connected.tolist()):
                incoming_sources[target].append(source)
                incoming_edge_indices[target].append(edge_index)
        return SparseSynapseGraph(
            targets_by_source=tuple(targets_by_source),
            weights_by_source=tuple(weights_by_source),
            signs_by_source=tuple(np.sign(values).astype(np.float64) for values in weights_by_source),
            delays_by_source=tuple(delays_by_source),
            sources_by_target=tuple(np.asarray(values, dtype=np.int64) for values in incoming_sources),
            edge_indices_by_target=tuple(np.asarray(values, dtype=np.int64) for values in incoming_edge_indices),
        )

    def _refractory_filter(self, threshold_crossings: np.ndarray, timestamp: int) -> np.ndarray:
        """Gate threshold crossings through a per-cell refractory interval."""
        if threshold_crossings.size == 0:
            return threshold_crossings
        can_fire = timestamp - self._last_cell_spike_ts[threshold_crossings] >= self.refractory_frames
        fired = threshold_crossings[can_fire]
        refractory = threshold_crossings[~can_fire]
        if refractory.size:
            self.v[refractory] = np.minimum(self.v[refractory], self.c[refractory])
        return fired

    def _schedule_sparse_recurrent(self, fired: np.ndarray) -> None:
        """Schedule sparse recurrent current from fired source cells."""
        if fired.size == 0:
            return
        for source in fired.tolist():
            targets = self._graph.targets_by_source[source]
            if targets.size == 0:
                continue
            source_gain = self._stp_resources[source] * self._stp_facilitation[source]
            weights = self._graph.weights_by_source[source] * source_gain
            delays = self._graph.delays_by_source[source]
            for delay in np.unique(delays):
                selected = delays == delay
                bucket = (self._delay_cursor + int(delay)) % len(self._synaptic_delay_queue)
                self._synaptic_delay_queue[bucket, targets[selected]] += 30.0 * weights[selected]

    def _stp_enabled(self) -> bool:
        return self.plasticity_mode in {"stp", "stdp", "stdp_homeostatic", "stp_stdp"}

    def _apply_stp(self, fired: np.ndarray) -> None:
        """Apply short-term depression/facilitation to sparse source cells."""
        if not self._stp_enabled() or fired.size == 0:
            return
        self._stp_resources[fired] *= 1.0 - self.stp_depression
        self._stp_resources[fired] = np.clip(self._stp_resources[fired], 0.05, 1.0)
        self._stp_facilitation[fired] += self.stp_facilitation * (3.0 - self._stp_facilitation[fired])
        self._stp_facilitation[fired] = np.clip(self._stp_facilitation[fired], 1.0, 3.0)

    def _recover_stp(self) -> None:
        """Relax short-term plasticity state back toward baseline."""
        if not self._stp_enabled():
            return
        self._stp_resources += (1.0 - self._stp_resources) / self.stp_recovery_frames
        self._stp_facilitation += (1.0 - self._stp_facilitation) / self.stp_facilitation_frames
        self._stp_resources = np.clip(self._stp_resources, 0.05, 1.0)
        self._stp_facilitation = np.clip(self._stp_facilitation, 1.0, 3.0)

    def _apply_stdp(self, fired: np.ndarray, timestamp: int) -> None:
        """
        Apply sparse, source-local STDP updates.

        Dense all-pairs STDP is exactly the scalability trap this engine avoids.
        Instead, each fired source adapts only its stored outgoing targets based
        on their recent postsynaptic spike times. This preserves the biological
        causal/anti-causal timing signal while keeping updates proportional to
        active edges rather than N squared.
        """
        if self.plasticity_mode not in {"stdp", "stdp_homeostatic"} or fired.size == 0:
            return
        learning_rate = self.stdp_learning_rate * (1.0 + self.dopamine)
        if learning_rate <= 0.0:
            return
        touched_sources: set[int] = set()
        for post in fired.tolist():
            sources = self._graph.sources_by_target[post]
            edge_indices = self._graph.edge_indices_by_target[post]
            if sources.size:
                delta_pre = timestamp - self._last_cell_spike_ts[sources]
                causal = (delta_pre >= 0) & (delta_pre <= self.stdp_tau_frames)
                for source, edge_index, delta in zip(
                    sources[causal].tolist(),
                    edge_indices[causal].tolist(),
                    delta_pre[causal].tolist(),
                ):
                    weights = self._graph.weights_by_source[source]
                    original_sign = np.sign(weights[edge_index])
                    if original_sign <= 0.0:
                        continue
                    weights[edge_index] += (
                        original_sign
                        * learning_rate
                        * np.exp(-delta / self.stdp_tau_frames)
                    )
                    touched_sources.add(source)

        for source in fired.tolist():
            targets = self._graph.targets_by_source[source]
            if targets.size == 0:
                continue
            weights = self._graph.weights_by_source[source]
            original_signs = np.sign(weights)
            delta_post = timestamp - self._last_cell_spike_ts[targets]
            anti_causal = (
                (original_signs > 0.0)
                & (delta_post >= 0)
                & (delta_post <= self.stdp_tau_frames)
            )
            if np.any(anti_causal):
                weights[anti_causal] -= (
                    original_signs[anti_causal]
                    * learning_rate
                    * 1.05
                    * np.exp(-delta_post[anti_causal] / self.stdp_tau_frames)
                )
            touched_sources.add(source)

        for source in touched_sources:
            weights = self._graph.weights_by_source[source]
            signs = self._graph.signs_by_source[source]
            nonzero = signs != 0.0
            weights[nonzero] = np.clip(np.abs(weights[nonzero]), 0.0, 0.25) * signs[nonzero]

    def _prepare_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        """Return bounded population-level channel connectivity."""
        matrix = np.asarray(connectivity, dtype=np.float64)
        if matrix.shape != (self.channel_count, self.channel_count):
            matrix = np.eye(self.channel_count, dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.clip(matrix, -1.0, 1.0)
        np.fill_diagonal(matrix, 0.0)
        return matrix
