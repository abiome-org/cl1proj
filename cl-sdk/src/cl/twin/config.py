from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TwinConfig:
    """
    Runtime configuration for the biological twin backend.

    The defaults intentionally favor deterministic SDK testing over maximal
    biological variability.  More expressive north-star features such as DIV,
    pharmacology, and plasticity are represented here early so future producer
    implementations can extend behavior without changing the user-facing API.
    """

    seed: int = 1
    div: int = 21
    baseline_rate_hz: float = 0.15
    noise_std_sample_units: float = 5.0
    stim_coupling: float = 1.0
    evoked_probability: float = 0.9
    evoked_latency_frames: int = 18
    evoked_jitter_frames: int = 8
    artifact_amplitude_sample_units: float = 1200.0
    artifact_decay_frames: float = 35.0
    eap_amplitude_sample_units: float = -120.0
    noise_color: str = "pink"
    plasticity_mode: str = "off"
    dopamine: float = 0.0
    gaba_block: float = 0.0
    culture_state: str = "normal"
    excitability: float = 1.0
    profile_path: str | None = None
    dynamics_mode: str = "off"
    recurrent_coupling: float = 0.35
    propagation_delay_frames: int = 12
    refractory_frames: int = 20
    snn_neuron_count: int = 256
    snn_excitatory_fraction: float = 0.8
    snn_coupling: float = 1.0
    snn_connection_probability: float = 0.08
    snn_length_constant_um: float = 250.0
    snn_field_gamma: float = 1.25
    snn_sparse_threshold: int = 1024
    snn_max_targets_per_source: int = 64
    snn_stdp_learning_rate: float = 0.004
    snn_stdp_tau_frames: float = 500.0
    snn_stp_recovery_frames: float = 1500.0
    snn_stp_facilitation_frames: float = 500.0
    snn_stp_depression: float = 0.15
    snn_stp_facilitation: float = 0.08
    snn_refractory_frames: int = 25
    snn_min_propagation_delay_frames: int = 1
    snn_max_propagation_delay_frames: int = 25
    snn_reset_backend: str = "numpy"
    spike_detection_threshold_sigma: float = 4.5
    spike_detection_refractory_frames: int = 20
    spike_detection_artifact_blank_frames: int = 4

    @classmethod
    def from_env(cls) -> "TwinConfig":
        """Build a config from environment variables used by the simulator."""
        return cls(
            seed=int(os.getenv("CL_SDK_TWIN_SEED", "1")),
            div=int(os.getenv("CL_SDK_TWIN_DIV", "21")),
            baseline_rate_hz=float(os.getenv("CL_SDK_TWIN_BASELINE_RATE_HZ", "0.15")),
            noise_std_sample_units=float(os.getenv("CL_SDK_TWIN_NOISE_STD", "5.0")),
            stim_coupling=float(os.getenv("CL_SDK_TWIN_STIM_COUPLING", "1.0")),
            evoked_probability=float(os.getenv("CL_SDK_TWIN_EVOKED_PROBABILITY", "0.9")),
            evoked_latency_frames=int(os.getenv("CL_SDK_TWIN_EVOKED_LATENCY_FRAMES", "18")),
            evoked_jitter_frames=int(os.getenv("CL_SDK_TWIN_EVOKED_JITTER_FRAMES", "8")),
            artifact_amplitude_sample_units=float(os.getenv("CL_SDK_TWIN_ARTIFACT_AMPLITUDE", "1200.0")),
            artifact_decay_frames=float(os.getenv("CL_SDK_TWIN_ARTIFACT_DECAY_FRAMES", "35.0")),
            eap_amplitude_sample_units=float(os.getenv("CL_SDK_TWIN_EAP_AMPLITUDE", "-120.0")),
            noise_color=os.getenv("CL_SDK_TWIN_NOISE_COLOR", "pink"),
            plasticity_mode=os.getenv("CL_SDK_TWIN_PLASTICITY", "off"),
            dopamine=float(os.getenv("CL_SDK_TWIN_DOPAMINE", "0.0")),
            gaba_block=float(os.getenv("CL_SDK_TWIN_GABA_BLOCK", "0.0")),
            culture_state=os.getenv("CL_SDK_TWIN_CULTURE_STATE", "normal"),
            excitability=float(os.getenv("CL_SDK_TWIN_EXCITABILITY", "1.0")),
            profile_path=os.getenv("CL_SDK_TWIN_PROFILE_PATH", None),
            dynamics_mode=os.getenv("CL_SDK_TWIN_DYNAMICS", "off"),
            recurrent_coupling=float(os.getenv("CL_SDK_TWIN_RECURRENT_COUPLING", "0.35")),
            propagation_delay_frames=int(os.getenv("CL_SDK_TWIN_PROPAGATION_DELAY_FRAMES", "12")),
            refractory_frames=int(os.getenv("CL_SDK_TWIN_REFRACTORY_FRAMES", "20")),
            snn_neuron_count=int(os.getenv("CL_SDK_TWIN_SNN_NEURON_COUNT", "256")),
            snn_excitatory_fraction=float(os.getenv("CL_SDK_TWIN_SNN_EXCITATORY_FRACTION", "0.8")),
            snn_coupling=float(os.getenv("CL_SDK_TWIN_SNN_COUPLING", "1.0")),
            snn_connection_probability=float(os.getenv("CL_SDK_TWIN_SNN_CONNECTION_PROBABILITY", "0.08")),
            snn_length_constant_um=float(os.getenv("CL_SDK_TWIN_SNN_LENGTH_CONSTANT_UM", "250.0")),
            snn_field_gamma=float(os.getenv("CL_SDK_TWIN_SNN_FIELD_GAMMA", "1.25")),
            snn_sparse_threshold=int(os.getenv("CL_SDK_TWIN_SNN_SPARSE_THRESHOLD", "1024")),
            snn_max_targets_per_source=int(os.getenv("CL_SDK_TWIN_SNN_MAX_TARGETS_PER_SOURCE", "64")),
            snn_stdp_learning_rate=float(os.getenv("CL_SDK_TWIN_SNN_STDP_LEARNING_RATE", "0.004")),
            snn_stdp_tau_frames=float(os.getenv("CL_SDK_TWIN_SNN_STDP_TAU_FRAMES", "500.0")),
            snn_stp_recovery_frames=float(os.getenv("CL_SDK_TWIN_SNN_STP_RECOVERY_FRAMES", "1500.0")),
            snn_stp_facilitation_frames=float(os.getenv("CL_SDK_TWIN_SNN_STP_FACILITATION_FRAMES", "500.0")),
            snn_stp_depression=float(os.getenv("CL_SDK_TWIN_SNN_STP_DEPRESSION", "0.15")),
            snn_stp_facilitation=float(os.getenv("CL_SDK_TWIN_SNN_STP_FACILITATION", "0.08")),
            snn_refractory_frames=int(os.getenv("CL_SDK_TWIN_SNN_REFRACTORY_FRAMES", "25")),
            snn_min_propagation_delay_frames=int(os.getenv("CL_SDK_TWIN_SNN_MIN_PROPAGATION_DELAY_FRAMES", "1")),
            snn_max_propagation_delay_frames=int(os.getenv("CL_SDK_TWIN_SNN_MAX_PROPAGATION_DELAY_FRAMES", "25")),
            snn_reset_backend=os.getenv("CL_SDK_TWIN_SNN_RESET_BACKEND", "numpy"),
            spike_detection_threshold_sigma=float(os.getenv("CL_SDK_TWIN_SPIKE_THRESHOLD_SIGMA", "4.5")),
            spike_detection_refractory_frames=int(os.getenv("CL_SDK_TWIN_SPIKE_REFRACTORY_FRAMES", "20")),
            spike_detection_artifact_blank_frames=int(os.getenv("CL_SDK_TWIN_SPIKE_ARTIFACT_BLANK_FRAMES", "4")),
        )
