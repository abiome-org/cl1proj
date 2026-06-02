from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cl1_clsdk_bridge import RESET_DYNAMICS_MODES, ResetSNNAdapter, is_reset_dynamics

from .._data_buffer import SPIKE_SAMPLES_TOTAL, SpikeRecord, StimRecord
from .config import TwinConfig
from .dynamics import PopulationDynamics
from .feedback import FeedbackStim
from .izhikevich import IzhikevichNetwork
from .maturation import CultureState, MaturationState
from .mea import MEAGeometry
from .noise import PinkNoiseState
from .plasticity import PlasticityState
from .profile import TwinProfile
from .spike_detector import DetectionBlankingWindow, RollingThresholdSpikeDetector
from .sparse_izhikevich import SparseIzhikevichNetwork
from .tissue import TissueTopology


@dataclass
class _SpikeEvent:
    timestamp: int
    channel: int
    amplitude: float


@dataclass
class _ArtifactEvent:
    timestamp: int
    channel: int
    amplitude: float


class SurrogateTwinModel:
    """
    First-generation biological twin core.

    This is not the final SNN.  It is the narrow, testable bridge from passive
    replay to a bidirectional twin: stimulation now perturbs latent excitability,
    creates realistic artifacts, and causes future evoked spikes through a
    spatial MEA coupling model.  The class boundary is designed so an Izhikevich
    or population SNN can later replace this surrogate while keeping the
    producer/shared-buffer integration stable.
    """

    def __init__(
        self,
        channel_count: int,
        frames_per_second: int,
        config: TwinConfig,
        profile: TwinProfile | None = None,
    ):
        self.channel_count = channel_count
        self.frames_per_second = frames_per_second
        self.config = config
        self.profile = profile or TwinProfile.default(
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
        )
        self.maturation = MaturationState.from_div(config.div)
        self.culture_state = CultureState.from_config(
            requested_state = config.culture_state,
            gaba_block      = config.gaba_block,
            excitability    = config.excitability,
        )
        self.geometry = MEAGeometry(channel_count=channel_count)
        self.rng = np.random.default_rng(config.seed)

        self._pending_spikes: list[_SpikeEvent] = []
        self._artifacts: list[_ArtifactEvent] = []
        self._eap_template = self._make_eap_template(config.eap_amplitude_sample_units)

        # Slow per-channel state gives the twin memory without committing to a
        # full cell-level model yet.  Stimulation increases excitability; each
        # render tick decays it back toward baseline.
        self._excitability = (
            np.ones(channel_count, dtype=np.float64)
            * config.excitability
            * self.maturation.excitability_scale
            * self.culture_state.excitability_scale
        )
        self._baseline_excitability = self._excitability.copy()
        self._baseline_rate_hz = self._profile_vector(
            self.profile.baseline_rate_hz_by_channel,
            default = config.baseline_rate_hz,
        ) * self.maturation.baseline_rate_scale * self.culture_state.baseline_rate_scale
        self._isi_median_frames = self._profile_int_vector(
            self.profile.isi_median_frames_by_channel,
            default = 0,
        )
        self._burst_rate_hz = self._profile_vector(
            self.profile.burst_rate_hz_by_channel,
            default = 0.0,
        ) * self.maturation.burst_rate_scale * self.culture_state.burst_rate_scale
        self._burst_median_duration_frames = self._profile_int_vector(
            self.profile.burst_median_duration_frames_by_channel,
            default = 0,
        )
        self._burst_spike_count_mean = self._profile_vector(
            self.profile.burst_spike_count_mean_by_channel,
            default = 0.0,
        ) * self.maturation.burst_count_scale * self.culture_state.burst_count_scale
        self._channel_confidence = self._profile_vector(
            self.profile.channel_confidence_by_channel,
            default = 1.0,
        )
        profile_noise = self._profile_vector(
            self.profile.noise_std_sample_units_by_channel,
            default = config.noise_std_sample_units,
        )
        self._noise_std = np.where(profile_noise > 0.0, profile_noise, config.noise_std_sample_units)
        self._noise = PinkNoiseState(
            channel_count   = channel_count,
            rng             = self.rng,
            std_by_channel  = self._noise_std,
            color           = config.noise_color,
        )
        self._spike_detector = RollingThresholdSpikeDetector(
            frames_per_second   = frames_per_second,
            channel_count       = channel_count,
            threshold_sigma     = config.spike_detection_threshold_sigma,
            refractory_frames   = config.spike_detection_refractory_frames,
            baseline_noise_std  = self._noise_std,
        )
        self._connectivity = self._profile_matrix(self.profile.connectivity, default_identity=True)
        self._stim_response_probability = self._profile_matrix(self.profile.stim_response_probability)
        self._stim_response_latency = self._profile_int_matrix(self.profile.stim_response_latency_frames)
        self._stim_response_confidence = self._profile_matrix(
            self.profile.stim_response_confidence,
            default_value = 1.0,
        )
        configured_snn_neuron_count = (
            int(self.profile.topology_neuron_count)
            if int(self.profile.topology_neuron_count) > 0
            else config.snn_neuron_count
        )
        dynamics_mode = config.dynamics_mode.lower()
        use_sparse_snn = (
            dynamics_mode in {"izhikevich", "snn"}
            and configured_snn_neuron_count >= config.snn_sparse_threshold
        )
        self._snn_topology = self._profile_topology(
            config.snn_neuron_count,
            config.snn_field_gamma,
            compute_pairwise_distances = not use_sparse_snn,
        )
        snn_neuron_count = (
            self._snn_topology.neuron_xy_um.shape[0]
            if self._snn_topology is not None
            else config.snn_neuron_count
        )
        self._plasticity = PlasticityState(
            channel_count = channel_count,
            mode          = config.plasticity_mode,
            dopamine      = config.dopamine,
        )
        self._dynamics = PopulationDynamics(
            channel_count       = channel_count,
            connectivity        = self._connectivity,
            mode                = config.dynamics_mode,
            coupling            = (
                config.recurrent_coupling
                * self.maturation.recurrent_coupling_scale
                * self.culture_state.recurrent_coupling_scale
            ),
            delay_frames        = self._matured_delay(config.propagation_delay_frames),
            refractory_frames   = config.refractory_frames,
            rng                 = self.rng,
        )
        if is_reset_dynamics(dynamics_mode):
            self._snn = ResetSNNAdapter(
                channel_count     = channel_count,
                neuron_count      = snn_neuron_count,
                frames_per_second = frames_per_second,
                rng               = self.rng,
                config            = config,
            )
        else:
            snn_class = SparseIzhikevichNetwork if use_sparse_snn else IzhikevichNetwork
            self._snn = snn_class(
                channel_count        = channel_count,
                neuron_count         = snn_neuron_count,
                frames_per_second    = frames_per_second,
                connectivity         = self._connectivity,
                rng                  = self.rng,
                excitatory_fraction  = config.snn_excitatory_fraction,
                coupling             = (
                    config.snn_coupling
                    * self.maturation.stim_coupling_scale
                    * self.culture_state.stim_coupling_scale
                ),
                field_gamma          = config.snn_field_gamma,
                connection_probability = (
                    config.snn_connection_probability
                    * self.maturation.connection_probability_scale
                ),
                length_constant_um   = config.snn_length_constant_um,
                plasticity_mode      = config.plasticity_mode,
                stdp_learning_rate   = config.snn_stdp_learning_rate,
                stdp_tau_frames      = config.snn_stdp_tau_frames,
                stp_recovery_frames  = config.snn_stp_recovery_frames,
                stp_facilitation_frames = config.snn_stp_facilitation_frames,
                stp_depression       = config.snn_stp_depression,
                stp_facilitation     = config.snn_stp_facilitation,
                refractory_frames     = config.snn_refractory_frames,
                min_propagation_delay_frames = self._matured_delay(config.snn_min_propagation_delay_frames),
                max_propagation_delay_frames = self._matured_delay(config.snn_max_propagation_delay_frames),
                gaba_block            = config.gaba_block,
                dopamine              = config.dopamine,
                topology              = self._snn_topology,
                **(
                    {"max_targets_per_source": config.snn_max_targets_per_source}
                    if use_sparse_snn
                    else {}
                ),
            )

    @staticmethod
    def _make_eap_template(amplitude: float) -> np.ndarray:
        """Create a 75-sample biphasic extracellular action-potential template."""
        t = np.arange(SPIKE_SAMPLES_TOTAL, dtype=np.float64)
        negative = np.exp(-0.5 * np.square((t - 25.0) / 3.0))
        positive = 0.45 * np.exp(-0.5 * np.square((t - 35.0) / 7.0))
        template = amplitude * (negative - positive)
        return template.astype(np.float32)

    def apply_stim(self, stim: StimRecord, current_uA: float = 1.0) -> None:
        """
        Couple an electrode stimulation into the simulated tissue state.

        A stimulation command has three immediate consequences: a recorded stim
        event, a decaying electrical artifact on nearby electrodes, and a set of
        probabilistic evoked spike events scheduled into the near future.
        """
        coupling = self.geometry.attenuation_from_channel(stim.channel)
        signed_strength = (
            abs(float(current_uA))
            * self.config.stim_coupling
            * self.maturation.stim_coupling_scale
            * self.culture_state.stim_coupling_scale
        )
        learned_pathways = np.clip(np.abs(self._connectivity[stim.channel]), 0.0, 1.0)
        confidence_scale = np.clip(self._stim_response_confidence[stim.channel], 0.25, 1.0)
        observed_response = np.clip(
            self._stim_response_probability[stim.channel] * confidence_scale,
            0.0,
            1.0,
        )
        profile_drive = np.maximum(coupling, learned_pathways * 0.75)
        local_drive = np.maximum(profile_drive, observed_response) * signed_strength
        local_drive *= self._plasticity.response_gain
        self._excitability += 0.15 * local_drive
        self._plasticity.on_stim(stim.timestamp, coupling)
        if self.config.dynamics_mode.lower() in {"izhikevich", "snn"} | set(RESET_DYNAMICS_MODES):
            if hasattr(self._snn, "apply_timed_stim"):
                self._snn.apply_timed_stim(stim.timestamp, stim.channel, local_drive)
            else:
                self._snn.apply_stim(stim.channel, local_drive)

        for channel, strength in enumerate(local_drive):
            if strength < 0.05:
                continue
            self._artifacts.append(_ArtifactEvent(
                timestamp = stim.timestamp,
                channel   = channel,
                amplitude = self.config.artifact_amplitude_sample_units * strength,
            ))

            p_spike = min(0.98, self.config.evoked_probability * strength * self._excitability[channel])
            if self.rng.random() <= p_spike:
                profiled_latency = self._stim_response_latency[stim.channel, channel]
                base_latency = (
                    int(profiled_latency)
                    if profiled_latency > 0
                    else self.config.evoked_latency_frames
                )
                jitter = int(self.rng.integers(
                    -self.config.evoked_jitter_frames,
                    self.config.evoked_jitter_frames + 1,
                ))
                self._pending_spikes.append(_SpikeEvent(
                    timestamp = stim.timestamp + base_latency + jitter,
                    channel   = channel,
                    amplitude = self.config.eap_amplitude_sample_units * max(0.35, strength),
                ))

    def apply_feedback(self, feedback: list[FeedbackStim]) -> None:
        """
        Apply a closed-loop feedback pattern through normal electrode coupling.

        Feedback protocols intentionally compile down to ordinary stim pulses so
        artifacts, evoked spikes, plasticity, and the optional SNN all experience
        the same bidirectional MEA path as user-delivered stimulation.
        """
        for event in feedback:
            self.apply_stim(
                StimRecord(timestamp=event.timestamp, channel=event.channel),
                current_uA=event.current_uA,
            )

    def _matured_delay(self, frames: int) -> int:
        """
        Scale propagation delays by DIV maturation while keeping them valid.

        This is intentionally a coarse developmental prior: immature cultures
        transmit recurrent activity more slowly, while mature cultures approach
        the configured delay values.
        """
        return max(1, int(round(int(frames) * self.maturation.propagation_delay_scale)))

    def render(
        self,
        from_timestamp: int,
        frame_count: int,
    ) -> tuple[np.ndarray, list[SpikeRecord]]:
        """
        Render raw electrode frames and detected spikes for a producer tick.

        The output is shaped exactly like the replay producer's output so the
        rest of the SDK can stay agnostic to whether data came from an H5 replay
        or from the twin model.
        """
        to_timestamp = from_timestamp + frame_count
        frames = self._noise.sample(frame_count)

        self._add_background_spikes(from_timestamp, to_timestamp)
        self._maybe_add_network_burst(from_timestamp, to_timestamp)
        self._add_snn_spikes(from_timestamp, frame_count)
        blanking_windows = self._artifact_blanking_windows(from_timestamp, to_timestamp)
        self._consume_spikes(from_timestamp, to_timestamp, frames)
        # Real pipelines detect spikes after amplifier blanking or artifact
        # removal, while user-visible raw frames still contain the artifact tail.
        # Keep those two views separate so artifacts can blind near-stim samples
        # without erasing legitimate evoked EAPs later in the response window.
        detection_frames = frames.copy()
        spike_records = self._spike_detector.detect(
            detection_frames,
            from_timestamp    = from_timestamp,
            blanking_windows  = blanking_windows,
        )
        self._apply_artifacts(from_timestamp, to_timestamp, frames)

        # Slow homeostatic decay prevents one strong stimulation from making the
        # model permanently hyper-excitable.
        self._excitability += (self._baseline_excitability - self._excitability) * 0.005
        self._plasticity.decay()
        self._dynamics.decay()

        return np.clip(np.rint(frames), -32768, 32767).astype(np.int16), spike_records

    def _add_snn_spikes(self, from_timestamp: int, frame_count: int) -> None:
        """Advance the optional cell-level SNN and queue its MEA spike outputs."""
        if self.config.dynamics_mode.lower() not in {"izhikevich", "snn"} | set(RESET_DYNAMICS_MODES):
            return
        for spike in self._snn.render(
            from_timestamp,
            frame_count,
            excitability  = self._excitability,
            response_gain = self._plasticity.response_gain,
        ):
            self._pending_spikes.append(_SpikeEvent(
                timestamp = spike.timestamp,
                channel   = spike.channel,
                amplitude = self.config.eap_amplitude_sample_units * max(0.35, spike.strength),
            ))

    def _add_background_spikes(self, from_timestamp: int, to_timestamp: int) -> None:
        """Generate sparse spontaneous spikes modulated by the slow latent state."""
        duration_sec = (to_timestamp - from_timestamp) / self.frames_per_second
        probabilities = self._baseline_rate_hz * duration_sec * self._excitability
        for channel, probability in enumerate(probabilities):
            if self.rng.random() <= min(0.5, probability):
                spike_ts = int(self.rng.integers(from_timestamp, to_timestamp))
                self._pending_spikes.append(_SpikeEvent(
                    timestamp = spike_ts,
                    channel   = channel,
                    amplitude = self.config.eap_amplitude_sample_units,
                ))
            self._maybe_add_background_burst(
                channel        = channel,
                from_timestamp = from_timestamp,
                to_timestamp   = to_timestamp,
                duration_sec   = duration_sec,
            )

    def _maybe_add_background_burst(
        self,
        *,
        channel: int,
        from_timestamp: int,
        to_timestamp: int,
        duration_sec: float,
    ) -> None:
        """Generate calibrated spontaneous burst packets when the profile has them."""
        confidence_scale = float(np.clip(self._channel_confidence[channel], 0.25, 1.0))
        burst_probability = (
            self._burst_rate_hz[channel]
            * duration_sec
            * self._excitability[channel]
            * confidence_scale
        )
        if burst_probability <= 0.0 or self.rng.random() > min(0.5, burst_probability):
            return

        spike_count = max(2, int(round(self._burst_spike_count_mean[channel])))
        burst_duration = int(self._burst_median_duration_frames[channel])
        if burst_duration <= 0:
            isi = int(self._isi_median_frames[channel])
            burst_duration = max(spike_count - 1, isi * max(1, spike_count - 1))
        latest_start = max(from_timestamp, to_timestamp - burst_duration - 1)
        start = int(self.rng.integers(from_timestamp, latest_start + 1))
        offsets = np.linspace(0, burst_duration, num=spike_count, dtype=int)
        for offset in offsets:
            timestamp = start + int(offset)
            if from_timestamp <= timestamp < to_timestamp:
                self._pending_spikes.append(_SpikeEvent(
                    timestamp = timestamp,
                    channel   = channel,
                    amplitude = self.config.eap_amplitude_sample_units,
                ))

    def _maybe_add_network_burst(self, from_timestamp: int, to_timestamp: int) -> None:
        """Generate synchronized culture-wide bursts in hyperexcitable states."""
        duration_sec = (to_timestamp - from_timestamp) / self.frames_per_second
        burst_probability = self.culture_state.network_burst_rate_hz * duration_sec
        if burst_probability <= 0.0 or self.rng.random() > min(0.75, burst_probability):
            return

        channel_count = max(1, int(round(
            self.channel_count * self.culture_state.network_burst_channel_fraction
        )))
        channels = self.rng.choice(self.channel_count, size=channel_count, replace=False)
        start = int(self.rng.integers(from_timestamp, max(from_timestamp + 1, to_timestamp)))
        # Synchronized bursts should look like near-simultaneous population
        # recruitment, with tiny jitter so all channels do not share one sample.
        for channel in channels.tolist():
            self._pending_spikes.append(_SpikeEvent(
                timestamp = min(to_timestamp - 1, start + int(self.rng.integers(0, 4))),
                channel   = int(channel),
                amplitude = self.config.eap_amplitude_sample_units,
            ))

    def _consume_spikes(
        self,
        from_timestamp: int,
        to_timestamp: int,
        frames: np.ndarray,
    ) -> None:
        """Move due biological spike events into raw frames for detection."""
        due: list[_SpikeEvent] = []
        remaining: list[_SpikeEvent] = []
        for event in self._pending_spikes:
            if from_timestamp <= event.timestamp < to_timestamp:
                due.append(event)
            else:
                remaining.append(event)
        self._pending_spikes = remaining

        recurrent_spikes: list[_SpikeEvent] = []
        for event in due:
            local_index = event.timestamp - from_timestamp
            template = self._eap_template * (event.amplitude / self.config.eap_amplitude_sample_units)
            self._add_template(frames, local_index, event.channel, template)
            self._plasticity.on_spike(event.timestamp, event.channel)
            for spike in self._dynamics.on_spike(
                timestamp     = event.timestamp,
                channel       = event.channel,
                excitability  = self._excitability,
                response_gain = self._plasticity.response_gain,
            ):
                recurrent_spikes.append(_SpikeEvent(
                    timestamp = spike.timestamp,
                    channel   = spike.channel,
                    amplitude = self.config.eap_amplitude_sample_units * max(0.35, spike.strength),
                ))
        self._pending_spikes.extend(recurrent_spikes)

    def _artifact_blanking_windows(
        self,
        from_timestamp: int,
        to_timestamp: int,
    ) -> list[DetectionBlankingWindow]:
        """Return detector blanking intervals caused by active stim artifacts."""
        if self.config.spike_detection_artifact_blank_frames <= 0:
            return []
        windows: list[DetectionBlankingWindow] = []
        for event in self._artifacts:
            start = event.timestamp
            end = start + self.config.spike_detection_artifact_blank_frames
            if end <= from_timestamp or start >= to_timestamp:
                continue
            windows.append(DetectionBlankingWindow(
                start_timestamp = start,
                end_timestamp   = end,
                channel         = event.channel,
            ))
        return windows

    def _apply_artifacts(
        self,
        from_timestamp: int,
        to_timestamp: int,
        frames: np.ndarray,
    ) -> None:
        """Superimpose stimulation artifacts and retain still-decaying tails."""
        retained: list[_ArtifactEvent] = []
        sample_times = np.arange(from_timestamp, to_timestamp)
        for event in self._artifacts:
            tail_age = to_timestamp - event.timestamp
            if tail_age < 0:
                retained.append(event)
                continue
            if tail_age < self.config.artifact_decay_frames * 8.0:
                retained.append(event)

            ages = sample_times - event.timestamp
            active = ages >= 0
            if not np.any(active):
                continue
            artifact = event.amplitude * np.exp(-ages[active] / self.config.artifact_decay_frames)
            frames[active, event.channel] += artifact
        self._artifacts = retained

    @staticmethod
    def _add_template(
        frames: np.ndarray,
        center_index: int,
        channel: int,
        template: np.ndarray,
    ) -> None:
        """Add a spike template centered on ``center_index`` into this frame block."""
        start = center_index - 25
        end = start + len(template)
        frame_start = max(0, start)
        frame_end = min(len(frames), end)
        if frame_start >= frame_end:
            return
        template_start = frame_start - start
        template_end = template_start + (frame_end - frame_start)
        frames[frame_start:frame_end, channel] += template[template_start:template_end]

    def _profile_vector(self, values: list[float], default: float) -> np.ndarray:
        """Return a profile vector, falling back to a uniform default."""
        if len(values) != self.channel_count:
            return np.full(self.channel_count, default, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def _profile_int_vector(self, values: list[int], default: int) -> np.ndarray:
        """Return an integer profile vector, falling back to a uniform default."""
        if len(values) != self.channel_count:
            return np.full(self.channel_count, default, dtype=np.int64)
        return np.asarray(values, dtype=np.int64)

    def _profile_matrix(
        self,
        values: list[list[float]],
        default_identity: bool = False,
        default_value: float = 0.0,
    ) -> np.ndarray:
        """Return a square profile matrix with safe defaults for missing data."""
        if len(values) != self.channel_count:
            if default_identity:
                return np.eye(self.channel_count, dtype=np.float64)
            return np.full((self.channel_count, self.channel_count), default_value, dtype=np.float64)
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (self.channel_count, self.channel_count):
            if default_identity:
                return np.eye(self.channel_count, dtype=np.float64)
            return np.full((self.channel_count, self.channel_count), default_value, dtype=np.float64)
        return matrix

    def _profile_int_matrix(self, values: list[list[int]]) -> np.ndarray:
        """Return a square integer matrix with zeros for missing latency data."""
        if len(values) != self.channel_count:
            return np.zeros((self.channel_count, self.channel_count), dtype=np.int64)
        matrix = np.asarray(values, dtype=np.int64)
        if matrix.shape != (self.channel_count, self.channel_count):
            return np.zeros((self.channel_count, self.channel_count), dtype=np.int64)
        return matrix

    def _profile_topology(
        self,
        configured_neuron_count: int,
        field_gamma: float,
        compute_pairwise_distances: bool = True,
    ) -> TissueTopology | None:
        """Build a calibrated SNN topology when the culture profile has a density prior."""
        density = self._profile_vector(
            self.profile.topology_channel_density,
            default = 1.0 / self.channel_count,
        )
        if len(self.profile.topology_channel_density) != self.channel_count:
            return None
        if np.sum(density) <= 0.0:
            return None
        profile_neuron_count = int(self.profile.topology_neuron_count)
        neuron_count = profile_neuron_count if profile_neuron_count > 0 else configured_neuron_count
        return TissueTopology.random(
            neuron_count                = neuron_count,
            mea                         = self.geometry,
            rng                         = self.rng,
            field_gamma                 = field_gamma,
            channel_density             = density,
            compute_pairwise_distances  = compute_pairwise_distances,
        )
