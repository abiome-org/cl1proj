"""Spiking culture simulator: an LIF/STDP recurrent E/I network with slow homeostasis, electrode I/O, and weight/activity snapshots (NumPy core, optional Brian2 backend)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .config import CultureConfig
from .electrodes import ChannelActivity, ElectrodeArray, StimEvent


@dataclass(frozen=True)
class NetworkSnapshot:
    weights: np.ndarray
    weight_sources: np.ndarray
    weight_targets: np.ndarray
    channel_path_strength: np.ndarray


class CorticalCultureNetwork:
    """
    Fast sparse LIF/STDP culture with a CL1-like electrode interface.

    This is the production reset/sweep engine.  It stores hidden weights and
    spikes for simulation metrics, but external protocols only stimulate and
    record through the electrode layer.
    """

    def __init__(self, cfg: CultureConfig, seed: int = 1):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.n_neurons = int(cfg.n_neurons)
        self.neuron_xy = self.rng.uniform(
            0.0,
            cfg.field_size_mm,
            size=(self.n_neurons, 2),
        )
        self.electrodes = ElectrodeArray.from_config(cfg, self.neuron_xy)
        self.is_excitatory = self.rng.random(self.n_neurons) < cfg.excitatory_fraction
        self.v = self.rng.normal(cfg.v_rest_mv, 2.0, self.n_neurons).astype(np.float64)
        self.syn_current = np.zeros(self.n_neurons, dtype=np.float64)
        self.stim_current = np.zeros(self.n_neurons, dtype=np.float64)
        self.last_spike_ms = np.full(self.n_neurons, -1e9, dtype=np.float64)
        self.refractory_until_ms = np.zeros(self.n_neurons, dtype=np.float64)
        self.rate_ema_hz = np.zeros(self.n_neurons, dtype=np.float64)
        self.elapsed_ms = 0.0
        self._homeostasis_elapsed = 0.0

        self.sources, self.targets, self.weights = self._build_sparse_connectome()
        self.baseline_weights = self.weights.copy()
        self.signs = np.sign(self.baseline_weights)
        self._out_start = self._csr_start(self.sources, self.n_neurons)
        self._incoming_edges = self._incoming_index()
        self._channel_matrix_cache: np.ndarray | None = None
        self._channel_matrix_weights_id: int | None = None

    @property
    def synapse_count(self) -> int:
        return int(self.weights.size)

    def snapshot(self) -> NetworkSnapshot:
        return NetworkSnapshot(
            weights               = self.weights.copy(),
            weight_sources        = self.sources.copy(),
            weight_targets        = self.targets.copy(),
            channel_path_strength = self.channel_connectivity_matrix(),
        )

    def weights_vector(self) -> np.ndarray:
        return self.weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != self.weights.shape:
            raise ValueError("Weight vector shape does not match this network.")
        magnitudes = np.clip(np.abs(values), 0.0, self.cfg.w_max)
        self.weights = magnitudes * self.signs
        self._invalidate_channel_matrix_cache()

    def advance(
        self,
        duration_ms: float,
        events: Iterable[StimEvent] | None = None,
        *,
        plasticity: bool = True,
        record: bool = True,
    ) -> ChannelActivity:
        events_by_step = self._events_by_step(duration_ms, events or [])
        dt = float(self.cfg.dt_ms)
        steps = max(1, int(np.ceil(duration_ms / dt)))
        spike_times: list[float] = []
        spike_neurons: list[np.ndarray] = []
        for step in range(steps):
            local_ms = step * dt
            now = self.elapsed_ms + local_ms
            for event in events_by_step.get(step, []):
                self.stim_current += (
                    self.cfg.stim_gain_mv_per_uA
                    * self.electrodes.stimulate(event)
                )

            self.syn_current *= np.exp(-dt / self.cfg.synapse_tau_ms)
            self.stim_current *= np.exp(-dt / self.cfg.stim_tau_ms)
            noise = self.rng.normal(0.0, self.cfg.background_noise_mv, self.n_neurons)
            dv = (
                self.cfg.v_rest_mv
                - self.v
                + self.syn_current
                + self.stim_current
                + noise
            ) * (dt / self.cfg.membrane_tau_ms)
            active = now >= self.refractory_until_ms
            self.v[active] += dv[active]

            spontaneous_p = self.cfg.spontaneous_rate_hz * dt / 1000.0
            spontaneous = self.rng.random(self.n_neurons) < spontaneous_p
            fired = np.flatnonzero((self.v >= self.cfg.v_threshold_mv) | spontaneous)
            if fired.size:
                spike_times.extend([local_ms] * int(fired.size))
                spike_neurons.append(fired.astype(np.int64))
                self._deliver_synapses(fired)
                if plasticity:
                    self._apply_stdp(fired, now)
                self.last_spike_ms[fired] = now
                self.refractory_until_ms[fired] = now + self.cfg.refractory_ms
                self.v[fired] = self.cfg.v_reset_mv

            self._update_rate_ema(fired.size, dt)
            self._homeostasis_elapsed += dt
            if plasticity and self._homeostasis_elapsed >= self.cfg.homeostasis_interval_ms:
                self._apply_homeostasis()
                self._homeostasis_elapsed = 0.0

        self.elapsed_ms += steps * dt
        if not record or not spike_neurons:
            return self.electrodes.record(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                duration_ms=steps * dt,
            )
        neuron_indices = np.concatenate(spike_neurons)
        return self.electrodes.record(
            np.asarray(spike_times, dtype=np.float64),
            neuron_indices,
            duration_ms=steps * dt,
        )

    def _invalidate_channel_matrix_cache(self) -> None:
        self._channel_matrix_cache = None
        self._channel_matrix_weights_id = None

    def channel_connectivity_matrix(self) -> np.ndarray:
        weights_id = id(self.weights)
        if self._channel_matrix_cache is not None and self._channel_matrix_weights_id == weights_id:
            return self._channel_matrix_cache
        channels_source = self.electrodes.nearest_channel[self.sources]
        channels_target = self.electrodes.nearest_channel[self.targets]
        matrix = np.zeros((self.cfg.n_electrodes, self.cfg.n_electrodes), dtype=np.float64)
        positive = self.weights > 0
        np.add.at(
            matrix,
            (channels_source[positive], channels_target[positive]),
            self.weights[positive],
        )
        counts = np.zeros_like(matrix)
        np.add.at(counts, (channels_source[positive], channels_target[positive]), 1.0)
        self._channel_matrix_cache = matrix / np.maximum(counts, 1.0)
        self._channel_matrix_weights_id = weights_id
        return self._channel_matrix_cache

    def path_strength(
        self,
        input_channels: Iterable[int],
        target_channels: Iterable[int],
    ) -> float:
        matrix = self.channel_connectivity_matrix()
        return float(matrix[np.ix_(list(input_channels), list(target_channels))].mean())

    def _build_sparse_connectome(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree = cKDTree(self.neuron_xy)
        k = min(
            self.n_neurons,
            max(8, self.cfg.max_out_degree * self.cfg.local_candidate_multiplier),
        )
        _, neighbor_ids = tree.query(self.neuron_xy, k=k, workers=int(self.cfg.build_workers))
        sources: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        long_range_edges = max(1, int(round(self.cfg.mean_out_degree * self.cfg.long_range_prob)))

        for source in range(self.n_neurons):
            candidates = np.asarray(neighbor_ids[source], dtype=np.int64)
            candidates = candidates[candidates != source]
            distances = np.linalg.norm(self.neuron_xy[candidates] - self.neuron_xy[source], axis=1)
            local_score = np.exp(-distances / max(self.cfg.connection_length_mm, 1e-6))
            if local_score.sum() <= 0.0:
                selected = np.array([], dtype=np.int64)
            else:
                p = local_score / local_score.sum()
                local_degree = max(1, self.cfg.mean_out_degree - long_range_edges)
                degree = min(len(candidates), local_degree)
                selected = self.rng.choice(candidates, size=degree, replace=False, p=p)
            if long_range_edges:
                random_targets = self.rng.integers(0, self.n_neurons, size=long_range_edges * 3)
                random_targets = random_targets[random_targets != source]
                if random_targets.size:
                    selected = np.unique(np.concatenate([
                        selected,
                        random_targets[:long_range_edges],
                    ])).astype(np.int64)
            if selected.size > self.cfg.max_out_degree:
                selected = self.rng.choice(selected, size=self.cfg.max_out_degree, replace=False)
            if selected.size == 0:
                continue
            if self.is_excitatory[source]:
                low, high = self.cfg.excitatory_weight_range
            else:
                low, high = self.cfg.inhibitory_weight_range
            source_weights = self.rng.uniform(low, high, size=selected.size)
            sources.append(np.full(selected.size, source, dtype=np.int64))
            targets.append(selected.astype(np.int64))
            weights.append(source_weights.astype(np.float64))

        source_array = np.concatenate(sources) if sources else np.array([], dtype=np.int64)
        target_array = np.concatenate(targets) if targets else np.array([], dtype=np.int64)
        weight_array = np.concatenate(weights) if weights else np.array([], dtype=np.float64)
        order = np.argsort(source_array, kind="stable")
        return source_array[order], target_array[order], weight_array[order]

    @staticmethod
    def _csr_start(sources: np.ndarray, n_sources: int) -> np.ndarray:
        counts = np.bincount(sources, minlength=n_sources)
        start = np.zeros(n_sources + 1, dtype=np.int64)
        start[1:] = np.cumsum(counts)
        return start

    def _incoming_index(self) -> list[np.ndarray]:
        incoming_edges: list[list[int]] = [[] for _ in range(self.n_neurons)]
        for edge, target in enumerate(self.targets.tolist()):
            incoming_edges[target].append(edge)
        return [np.asarray(values, dtype=np.int64) for values in incoming_edges]

    def _events_by_step(
        self,
        duration_ms: float,
        events: Iterable[StimEvent],
    ) -> dict[int, list[StimEvent]]:
        steps = max(1, int(np.ceil(duration_ms / self.cfg.dt_ms)))
        result: dict[int, list[StimEvent]] = {}
        for event in events:
            event_ms = max(0.0, float(event.time_us) / 1000.0)
            if event_ms >= duration_ms:
                continue
            step = min(steps - 1, int(event_ms / self.cfg.dt_ms))
            result.setdefault(step, []).append(event)
        return result

    def _deliver_synapses(self, fired: np.ndarray) -> None:
        for source in fired.tolist():
            start = self._out_start[source]
            end = self._out_start[source + 1]
            if start == end:
                continue
            targets = self.targets[start:end]
            weights = self.weights[start:end] * self.cfg.synaptic_gain_mv
            np.add.at(self.syn_current, targets, weights)

    def _apply_stdp(self, fired: np.ndarray, now_ms: float) -> None:
        fired_mask = np.zeros(self.n_neurons, dtype=bool)
        fired_mask[fired] = True
        for post in fired:
            edges = self._incoming_edges[post]
            if edges.size == 0:
                continue
            sources = self.sources[edges]
            excitatory = self.is_excitatory[sources]
            delta = now_ms - self.last_spike_ms[sources]
            causal = excitatory & (delta >= 0.0) & (delta <= self.cfg.stdp_tau_ms)
            if np.any(causal):
                self.weights[edges[causal]] += self.cfg.stdp_a_plus * np.exp(
                    -delta[causal] / self.cfg.stdp_tau_ms
                )

        for source in fired:
            if not self.is_excitatory[source]:
                continue
            start = self._out_start[source]
            end = self._out_start[source + 1]
            if start == end:
                continue
            targets = self.targets[start:end]
            delta = now_ms - self.last_spike_ms[targets]
            anti_causal = (delta >= 0.0) & (delta <= self.cfg.stdp_tau_ms)
            anti_causal &= ~fired_mask[targets]
            if np.any(anti_causal):
                edges = np.arange(start, end, dtype=np.int64)[anti_causal]
                self.weights[edges] -= self.cfg.stdp_a_minus * np.exp(
                    -delta[anti_causal] / self.cfg.stdp_tau_ms
                )
        self._clip_weights()

    def _clip_weights(self) -> None:
        excitatory = self.signs > 0.0
        inhibitory = self.signs < 0.0
        self.weights[excitatory] = np.clip(self.weights[excitatory], 0.0, self.cfg.w_max)
        self.weights[inhibitory] = np.clip(self.weights[inhibitory], self.cfg.w_min, 0.0)
        self._invalidate_channel_matrix_cache()

    def _update_rate_ema(self, total_fired: int, dt_ms: float) -> None:
        instantaneous_rate = (total_fired / max(self.n_neurons, 1)) * (1000.0 / dt_ms)
        alpha = min(1.0, dt_ms / 1000.0)
        self.rate_ema_hz += alpha * (instantaneous_rate - self.rate_ema_hz)

    def _apply_homeostasis(self) -> None:
        mean_rate = float(self.rate_ema_hz.mean())
        if mean_rate <= 0.0:
            factor = 1.0 + self.cfg.homeostasis_rate
        else:
            error = (self.cfg.target_rate_hz - mean_rate) / max(self.cfg.target_rate_hz, 1e-9)
            factor = 1.0 + self.cfg.homeostasis_rate * np.clip(error, -1.0, 1.0)
        excitatory = self.signs > 0.0
        self.weights[excitatory] *= factor
        self._clip_weights()


class Brian2CultureNetwork:
    """
    Brian2-backed LIF/STDP implementation for smaller exact runs.

    The public methods mirror `CorticalCultureNetwork`; sweeps default to the
    sparse NumPy engine because repeated Brian2 graph runs carry higher overhead.
    """

    def __init__(self, cfg: CultureConfig, seed: int = 1):
        try:
            import brian2 as b2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("brian2 is required for backend='brian2'.") from exc

        self.b2 = b2
        self.cfg = cfg
        self.seed = int(seed)
        b2.prefs.codegen.target = cfg.brian2_codegen_target
        self._fast = CorticalCultureNetwork(cfg, seed=seed)
        self.n_neurons = self._fast.n_neurons
        self.neuron_xy = self._fast.neuron_xy
        self.electrodes = self._fast.electrodes
        self.is_excitatory = self._fast.is_excitatory
        self.sources = self._fast.sources
        self.targets = self._fast.targets
        self.signs = self._fast.signs
        self.elapsed_ms = 0.0
        b2.start_scope()
        b2.seed(seed)
        b2.defaultclock.dt = cfg.dt_ms * b2.ms
        eqs = """
        dv/dt = (v_rest - v + I_syn + I_stim) / tau_m : volt (unless refractory)
        dI_syn/dt = -I_syn / tau_syn : volt
        dI_stim/dt = -I_stim / tau_stim : volt
        """
        self.group = b2.NeuronGroup(
            self.n_neurons,
            eqs,
            threshold="v > v_threshold",
            reset="v = v_reset",
            refractory=cfg.refractory_ms * b2.ms,
            method="exact",
            namespace={
                "v_rest": cfg.v_rest_mv * b2.mV,
                "v_threshold": cfg.v_threshold_mv * b2.mV,
                "v_reset": cfg.v_reset_mv * b2.mV,
                "tau_m": cfg.membrane_tau_ms * b2.ms,
                "tau_syn": cfg.synapse_tau_ms * b2.ms,
                "tau_stim": cfg.stim_tau_ms * b2.ms,
            },
        )
        self.group.v = cfg.v_rest_mv * b2.mV
        self.syn = b2.Synapses(
            self.group,
            self.group,
            """
            w : volt
            dapre/dt = -apre / tau_stdp : 1 (event-driven)
            dapost/dt = -apost / tau_stdp : 1 (event-driven)
            sgn : 1
            """,
            on_pre="""
            I_syn_post += w
            apre += a_plus
            w = clip(w + sgn * apost * mV, w_min, w_max)
            """,
            on_post="""
            apost += a_minus
            w = clip(w + sgn * apre * mV, w_min, w_max)
            """,
            namespace={
                "tau_stdp": cfg.stdp_tau_ms * b2.ms,
                "a_plus": cfg.stdp_a_plus,
                "a_minus": -cfg.stdp_a_minus,
                "w_min": cfg.w_min * b2.mV,
                "w_max": cfg.w_max * b2.mV,
                "mV": b2.mV,
            },
        )
        self.syn.connect(i=self.sources, j=self.targets)
        self.syn.w = self._fast.weights * b2.mV
        self.syn.sgn = self.signs
        self.monitor = b2.SpikeMonitor(self.group)
        self.network = b2.Network(self.group, self.syn, self.monitor)
        self._last_monitor_index = 0

    @property
    def synapse_count(self) -> int:
        return int(len(self.sources))

    def weights_vector(self) -> np.ndarray:
        return np.asarray(self.syn.w / self.b2.mV, dtype=np.float64)

    def _sync_fast(self) -> None:
        self._fast.set_weights(self.weights_vector())

    def snapshot(self) -> NetworkSnapshot:
        self._sync_fast()
        return self._fast.snapshot()

    def set_weights(self, weights: np.ndarray) -> None:
        self.syn.w = np.asarray(weights, dtype=np.float64) * self.b2.mV

    def advance(
        self,
        duration_ms: float,
        events: Iterable[StimEvent] | None = None,
        *,
        plasticity: bool = True,
        record: bool = True,
    ) -> ChannelActivity:
        if not plasticity:
            self._sync_fast()
            return self._fast.advance(duration_ms, events, plasticity=False, record=record)
        for event in events or []:
            drive = self.electrodes.stimulate(event) * self.cfg.stim_gain_mv_per_uA
            self.group.I_stim += drive * self.b2.mV
        self.network.run(float(duration_ms) * self.b2.ms)
        new_slice = slice(self._last_monitor_index, len(self.monitor.i))
        neuron_indices = np.asarray(self.monitor.i[new_slice], dtype=np.int64)
        times = np.asarray(self.monitor.t[new_slice] / self.b2.ms, dtype=np.float64)
        self._last_monitor_index = len(self.monitor.i)
        local_times = np.maximum(0.0, times - self.elapsed_ms)
        self.elapsed_ms += float(duration_ms)
        return self.electrodes.record(local_times, neuron_indices, duration_ms=duration_ms)

    def path_strength(self, input_channels: Iterable[int], target_channels: Iterable[int]) -> float:
        self._sync_fast()
        return self._fast.path_strength(input_channels, target_channels)

    def channel_connectivity_matrix(self) -> np.ndarray:
        self._sync_fast()
        return self._fast.channel_connectivity_matrix()


def build_network(cfg: CultureConfig, seed: int = 1):
    backend = cfg.backend.lower()
    if backend == "brian2":
        return Brian2CultureNetwork(cfg, seed=seed)
    if backend in {"numpy", "fast", "sparse"}:
        return CorticalCultureNetwork(cfg, seed=seed)
    raise ValueError(f"Unknown SNN reset backend: {cfg.backend}")
