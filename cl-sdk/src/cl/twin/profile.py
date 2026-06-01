from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class TwinProfile:
    """
    Serializable calibration profile for a specific culture/session.

    This is the handoff point between observed CL1 data and the live simulator.
    It intentionally stores only plain Python data so profiles are safe to save,
    review, diff, and load in the producer subprocess.
    """

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 5

    schema_version: int = CURRENT_SCHEMA_VERSION
    source_file: str = ""
    channel_count: int = 64
    frames_per_second: int = 25_000
    duration_frames: int = 0
    baseline_rate_hz_by_channel: list[float] = field(default_factory=list)
    baseline_rate_ci95_hz_by_channel: list[list[float]] = field(default_factory=list)
    noise_std_sample_units_by_channel: list[float] = field(default_factory=list)
    connectivity: list[list[float]] = field(default_factory=list)
    stim_response_probability: list[list[float]] = field(default_factory=list)
    stim_response_probability_ci95: list[list[list[float]]] = field(default_factory=list)
    stim_response_latency_frames: list[list[int]] = field(default_factory=list)
    stim_response_latency_ci95_frames: list[list[list[int]]] = field(default_factory=list)
    stim_response_count: list[list[int]] = field(default_factory=list)
    stim_response_confidence: list[list[float]] = field(default_factory=list)
    isi_median_frames_by_channel: list[int] = field(default_factory=list)
    burst_rate_hz_by_channel: list[float] = field(default_factory=list)
    burst_median_duration_frames_by_channel: list[int] = field(default_factory=list)
    burst_spike_count_mean_by_channel: list[float] = field(default_factory=list)
    topology_neuron_count: int = 0
    topology_channel_density: list[float] = field(default_factory=list)
    channel_confidence_by_channel: list[float] = field(default_factory=list)
    field_confidence: dict[str, float] = field(default_factory=dict)
    dead_channels: list[int] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls, channel_count: int = 64, frames_per_second: int = 25_000) -> "TwinProfile":
        """Create a neutral profile used when no recording calibration is supplied."""
        return cls(
            channel_count                      = channel_count,
            frames_per_second                  = frames_per_second,
            baseline_rate_hz_by_channel        = [0.0] * channel_count,
            baseline_rate_ci95_hz_by_channel   = [[0.0, 0.0] for _ in range(channel_count)],
            noise_std_sample_units_by_channel  = [0.0] * channel_count,
            connectivity                       = np.eye(channel_count, dtype=float).tolist(),
            stim_response_probability          = np.zeros((channel_count, channel_count), dtype=float).tolist(),
            stim_response_probability_ci95      = cls._zero_interval_matrix(channel_count, as_int=False),
            stim_response_latency_frames       = np.zeros((channel_count, channel_count), dtype=int).tolist(),
            stim_response_latency_ci95_frames  = cls._zero_interval_matrix(channel_count, as_int=True),
            stim_response_count                = np.zeros((channel_count, channel_count), dtype=int).tolist(),
            stim_response_confidence           = np.ones((channel_count, channel_count), dtype=float).tolist(),
            isi_median_frames_by_channel       = [0] * channel_count,
            burst_rate_hz_by_channel           = [0.0] * channel_count,
            burst_median_duration_frames_by_channel = [0] * channel_count,
            burst_spike_count_mean_by_channel  = [0.0] * channel_count,
            topology_neuron_count              = 0,
            topology_channel_density           = [1.0 / channel_count] * channel_count,
            channel_confidence_by_channel      = [0.0] * channel_count,
            field_confidence                   = cls._neutral_field_confidence(),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TwinProfile":
        """Load a profile JSON file from disk."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**cls._migrate_data(data))

    @classmethod
    def from_recording_path(
        cls,
        recording_path: str | Path,
        *,
        bin_size_sec: float = 0.1,
        response_window_frames: int = 250,
        burst_isi_sec: float = 0.1,
        burst_min_spikes: int = 3,
    ) -> "TwinProfile":
        """Open a recording and calibrate a profile from it."""
        from ..util import RecordingView

        recording = RecordingView(str(recording_path))
        try:
            return cls.from_recording(
                recording,
                bin_size_sec           = bin_size_sec,
                response_window_frames = response_window_frames,
                burst_isi_sec          = burst_isi_sec,
                burst_min_spikes       = burst_min_spikes,
            )
        finally:
            recording.close()

    def save(self, path: str | Path) -> None:
        """Write this profile as stable, human-readable JSON."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_recording(
        cls,
        recording,
        *,
        bin_size_sec: float = 0.1,
        response_window_frames: int = 250,
        burst_isi_sec: float = 0.1,
        burst_min_spikes: int = 3,
    ) -> "TwinProfile":
        """
        Calibrate a profile from a ``RecordingView``.

        The fitter deliberately starts with robust MEA-level statistics instead
        of overfitting a complex hidden model: baseline firing, raw noise,
        functional coupling, and observed stim-triggered response timing.
        """
        channel_count = int(recording.attributes["channel_count"])
        frames_per_second = int(recording.attributes["frames_per_second"])
        duration_frames = int(recording.attributes["duration_frames"])
        duration_sec = max(duration_frames / frames_per_second, 1.0 / frames_per_second)

        spike_frames_by_channel = recording._analysis_cache.get_spike_frames_by_channel()
        baseline_rates = [
            float(len(spike_frames_by_channel[ch]) / duration_sec)
            for ch in range(channel_count)
        ]
        spike_counts = np.array(
            [len(spike_frames_by_channel[ch]) for ch in range(channel_count)],
            dtype=np.float64,
        )

        noise_std = cls._estimate_noise(recording, channel_count)
        connectivity = cls._estimate_connectivity(
            spike_frames_by_channel = spike_frames_by_channel,
            channel_count           = channel_count,
            frames_per_second       = frames_per_second,
            duration_frames         = duration_frames,
            bin_size_sec            = bin_size_sec,
        )
        baseline_rate_ci95 = cls._poisson_rate_ci95(
            spike_counts = spike_counts,
            duration_sec = duration_sec,
        )
        (
            response_probability,
            response_probability_ci95,
            response_latency,
            response_latency_ci95,
            response_count,
            response_confidence,
        ) = cls._estimate_stim_response(
            recording               = recording,
            spike_frames_by_channel = spike_frames_by_channel,
            channel_count           = channel_count,
            response_window_frames  = response_window_frames,
        )
        (
            isi_median,
            burst_rate,
            burst_duration,
            burst_spike_count,
        ) = cls._estimate_temporal_structure(
            spike_frames_by_channel = spike_frames_by_channel,
            channel_count           = channel_count,
            frames_per_second       = frames_per_second,
            duration_sec            = duration_sec,
            burst_isi_frames        = max(1, int(round(frames_per_second * burst_isi_sec))),
            burst_min_spikes        = burst_min_spikes,
        )
        topology_neuron_count, topology_channel_density = cls._estimate_topology(
            spike_counts              = spike_counts,
            channel_count             = channel_count,
            min_neuron_count          = channel_count,
            max_neuron_count          = 4096,
            stim_response_probability = response_probability,
            connectivity              = connectivity,
        )

        dead_channels = [
            ch for ch, rate in enumerate(baseline_rates)
            if rate == 0.0 and noise_std[ch] == 0.0
        ]
        stim_count = 0 if getattr(recording, "stims", None) is None else len(recording.stims)
        channel_confidence, field_confidence = cls._estimate_confidence(
            duration_sec    = duration_sec,
            spike_counts    = spike_counts,
            sample_count    = 0 if getattr(recording, "samples", None) is None else len(recording.samples),
            stim_count      = stim_count,
            burst_count     = np.array(burst_rate, dtype=np.float64) * duration_sec,
        )

        source_file = getattr(getattr(recording, "file", None), "filename", "")
        return cls(
            source_file                         = str(source_file),
            channel_count                       = channel_count,
            frames_per_second                   = frames_per_second,
            duration_frames                     = duration_frames,
            baseline_rate_hz_by_channel         = baseline_rates,
            baseline_rate_ci95_hz_by_channel    = baseline_rate_ci95,
            noise_std_sample_units_by_channel   = noise_std,
            connectivity                        = connectivity.tolist(),
            stim_response_probability           = response_probability.tolist(),
            stim_response_probability_ci95      = response_probability_ci95,
            stim_response_latency_frames        = response_latency.tolist(),
            stim_response_latency_ci95_frames   = response_latency_ci95,
            stim_response_count                 = response_count.tolist(),
            stim_response_confidence            = response_confidence.tolist(),
            isi_median_frames_by_channel        = isi_median,
            burst_rate_hz_by_channel            = burst_rate,
            burst_median_duration_frames_by_channel = burst_duration,
            burst_spike_count_mean_by_channel   = burst_spike_count,
            topology_neuron_count               = topology_neuron_count,
            topology_channel_density            = topology_channel_density,
            channel_confidence_by_channel       = channel_confidence,
            field_confidence                    = field_confidence,
            dead_channels                       = dead_channels,
            notes                               = {
                "bin_size_sec": bin_size_sec,
                "response_window_frames": response_window_frames,
                "burst_isi_sec": burst_isi_sec,
                "burst_min_spikes": burst_min_spikes,
            },
        )

    @staticmethod
    def _estimate_noise(recording, channel_count: int) -> list[float]:
        """Estimate per-channel raw noise from recorded samples when available."""
        if getattr(recording, "samples", None) is None:
            return [0.0] * channel_count
        samples = recording.samples
        if samples is None or len(samples) == 0:
            return [0.0] * channel_count
        # Limit the slice so calibration remains quick on long recordings.
        sample_count = min(len(samples), 25_000)
        sample_block = np.asarray(samples[:sample_count], dtype=np.float64)
        return np.std(sample_block, axis=0).astype(float).tolist()

    @staticmethod
    def _estimate_connectivity(
        *,
        spike_frames_by_channel: dict[int, np.ndarray],
        channel_count: int,
        frames_per_second: int,
        duration_frames: int,
        bin_size_sec: float,
    ) -> np.ndarray:
        """Estimate functional connectivity from binned spike-count correlation."""
        bin_size_frames = max(1, int(round(frames_per_second * bin_size_sec)))
        bin_count = max(2, int(np.ceil(duration_frames / bin_size_frames)))
        counts = np.zeros((channel_count, bin_count), dtype=np.float64)
        for ch in range(channel_count):
            spike_frames = spike_frames_by_channel[ch]
            if len(spike_frames) == 0:
                continue
            bins = np.clip(spike_frames // bin_size_frames, 0, bin_count - 1)
            np.add.at(counts[ch], bins, 1.0)
        connectivity = np.corrcoef(counts)
        np.nan_to_num(connectivity, nan=0.0, copy=False)
        np.fill_diagonal(connectivity, 1.0)
        return np.clip(connectivity, -1.0, 1.0)

    @staticmethod
    def _estimate_stim_response(
        *,
        recording,
        spike_frames_by_channel: dict[int, np.ndarray],
        channel_count: int,
        response_window_frames: int,
    ) -> tuple[
        np.ndarray,
        list[list[list[float]]],
        np.ndarray,
        list[list[list[int]]],
        np.ndarray,
        np.ndarray,
    ]:
        """Estimate stim-to-spike probability, uncertainty, support, and latency."""
        probability = np.zeros((channel_count, channel_count), dtype=np.float64)
        latency = np.zeros((channel_count, channel_count), dtype=np.int64)
        response_count = np.zeros((channel_count, channel_count), dtype=np.int64)
        confidence = np.zeros((channel_count, channel_count), dtype=np.float64)
        probability_ci95 = TwinProfile._zero_interval_matrix(channel_count, as_int=False)
        latency_ci95 = TwinProfile._zero_interval_matrix(channel_count, as_int=True)
        if getattr(recording, "stims", None) is None or recording.stims is None:
            return probability, probability_ci95, latency, latency_ci95, response_count, confidence

        stims_by_channel: list[list[int]] = [[] for _ in range(channel_count)]
        for stim in recording.stims:
            stims_by_channel[int(stim["channel"])].append(int(stim["timestamp"]))

        for stim_ch, stim_times in enumerate(stims_by_channel):
            if not stim_times:
                continue
            for target_ch in range(channel_count):
                target_spikes = spike_frames_by_channel[target_ch]
                if len(target_spikes) == 0:
                    continue
                hit_latencies: list[int] = []
                for stim_ts in stim_times:
                    left = np.searchsorted(target_spikes, stim_ts, side="left")
                    if left >= len(target_spikes):
                        continue
                    delta = int(target_spikes[left] - stim_ts)
                    if 0 <= delta <= response_window_frames:
                        hit_latencies.append(delta)
                if hit_latencies:
                    hit_count = len(hit_latencies)
                    probability[stim_ch, target_ch] = hit_count / len(stim_times)
                    probability_ci95[stim_ch][target_ch] = TwinProfile._binomial_ci95(
                        hit_count = hit_count,
                        trial_count = len(stim_times),
                    )
                    latency[stim_ch, target_ch] = int(np.median(hit_latencies))
                    latency_ci95[stim_ch][target_ch] = TwinProfile._latency_ci95(hit_latencies)
                    response_count[stim_ch, target_ch] = hit_count
                    confidence[stim_ch, target_ch] = TwinProfile._stim_pair_confidence(
                        stim_count = len(stim_times),
                        hit_count  = hit_count,
                    )
        return probability, probability_ci95, latency, latency_ci95, response_count, confidence

    @staticmethod
    def _estimate_temporal_structure(
        *,
        spike_frames_by_channel: dict[int, np.ndarray],
        channel_count: int,
        frames_per_second: int,
        duration_sec: float,
        burst_isi_frames: int,
        burst_min_spikes: int,
    ) -> tuple[list[int], list[float], list[int], list[float]]:
        """Estimate per-channel ISI and burst statistics from spike timestamps."""
        isi_median: list[int] = []
        burst_rate: list[float] = []
        burst_duration: list[int] = []
        burst_spike_count: list[float] = []

        for ch in range(channel_count):
            spikes = np.asarray(spike_frames_by_channel[ch], dtype=np.int64)
            if len(spikes) < 2:
                isi_median.append(0)
            else:
                isi = np.diff(spikes)
                isi_median.append(int(np.median(isi)))

            bursts = TwinProfile._find_bursts(
                spikes            = spikes,
                max_isi_frames    = burst_isi_frames,
                min_spikes        = burst_min_spikes,
            )
            if not bursts:
                burst_rate.append(0.0)
                burst_duration.append(0)
                burst_spike_count.append(0.0)
                continue

            durations = [int(burst[-1] - burst[0]) for burst in bursts]
            counts = [len(burst) for burst in bursts]
            burst_rate.append(float(len(bursts) / max(duration_sec, 1.0 / frames_per_second)))
            burst_duration.append(int(np.median(durations)))
            burst_spike_count.append(float(np.mean(counts)))

        return isi_median, burst_rate, burst_duration, burst_spike_count

    @staticmethod
    def _find_bursts(
        *,
        spikes: np.ndarray,
        max_isi_frames: int,
        min_spikes: int,
    ) -> list[np.ndarray]:
        """Group spike runs separated by short ISIs into simple burst events."""
        if len(spikes) < min_spikes:
            return []
        bursts: list[np.ndarray] = []
        start = 0
        isi = np.diff(spikes)
        for index, gap in enumerate(isi, start=1):
            if gap <= max_isi_frames:
                continue
            candidate = spikes[start:index]
            if len(candidate) >= min_spikes:
                bursts.append(candidate)
            start = index

        candidate = spikes[start:]
        if len(candidate) >= min_spikes:
            bursts.append(candidate)
        return bursts

    @classmethod
    def _migrate_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert older profile JSON into the current dataclass schema.

        Profiles are user-visible calibration artifacts, so loading must be
        tolerant of missing fields introduced by newer twin revisions.  Unknown
        fields are discarded after migration so forward experiments do not crash
        stable SDK consumers with unexpected constructor arguments.
        """
        migrated = dict(data)
        schema_version = int(migrated.get("schema_version", 1))
        if schema_version > cls.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported TwinProfile schema_version {schema_version}; "
                f"current loader supports up to {cls.CURRENT_SCHEMA_VERSION}"
            )

        channel_count = int(migrated.get("channel_count", 64))
        if schema_version < 2:
            migrated.setdefault("isi_median_frames_by_channel", [0] * channel_count)
            migrated.setdefault("burst_rate_hz_by_channel", [0.0] * channel_count)
            migrated.setdefault("burst_median_duration_frames_by_channel", [0] * channel_count)
            migrated.setdefault("burst_spike_count_mean_by_channel", [0.0] * channel_count)
            migrated.setdefault("channel_confidence_by_channel", [0.0] * channel_count)
            migrated.setdefault("field_confidence", cls._neutral_field_confidence())
        if schema_version < 3:
            migrated.setdefault("stim_response_count", np.zeros((channel_count, channel_count), dtype=int).tolist())
            migrated.setdefault(
                "stim_response_confidence",
                cls._legacy_stim_confidence(migrated, channel_count),
            )
        if schema_version < 4:
            migrated.setdefault("topology_neuron_count", 0)
            migrated.setdefault("topology_channel_density", [1.0 / channel_count] * channel_count)
        if schema_version < 5:
            migrated.setdefault("baseline_rate_ci95_hz_by_channel", [[0.0, 0.0] for _ in range(channel_count)])
            migrated.setdefault(
                "stim_response_probability_ci95",
                cls._zero_interval_matrix(channel_count, as_int=False),
            )
            migrated.setdefault(
                "stim_response_latency_ci95_frames",
                cls._zero_interval_matrix(channel_count, as_int=True),
            )

        migrated["schema_version"] = cls.CURRENT_SCHEMA_VERSION
        valid_fields = {field_def.name for field_def in fields(cls)}
        return {key: value for key, value in migrated.items() if key in valid_fields}

    @staticmethod
    def _legacy_stim_confidence(data: dict[str, Any], channel_count: int) -> list[list[float]]:
        """Preserve older calibrated response effects when pair confidence is absent."""
        probability = data.get("stim_response_probability", [])
        if len(probability) != channel_count:
            return np.ones((channel_count, channel_count), dtype=float).tolist()
        matrix = np.asarray(probability, dtype=np.float64)
        if matrix.shape != (channel_count, channel_count):
            return np.ones((channel_count, channel_count), dtype=float).tolist()
        # Older profiles had no way to express pairwise uncertainty.  Use full
        # confidence for nonzero calibrated responses to avoid silently erasing
        # prior profile behavior, and neutral confidence elsewhere.
        return np.where(matrix > 0.0, 1.0, 1.0).astype(float).tolist()

    @staticmethod
    def _neutral_field_confidence() -> dict[str, float]:
        """Return confidence metadata for uncalibrated/default profiles."""
        return {
            "baseline_rate_hz": 0.0,
            "noise_std_sample_units": 0.0,
            "connectivity": 0.0,
            "stim_response": 0.0,
            "temporal_structure": 0.0,
        }

    @staticmethod
    def _zero_interval_matrix(channel_count: int, *, as_int: bool) -> list[list[list[Any]]]:
        """Return a square ``[lower, upper]`` interval matrix for neutral profiles."""
        zero: Any = 0 if as_int else 0.0
        return [[[zero, zero] for _ in range(channel_count)] for _ in range(channel_count)]

    @staticmethod
    def _poisson_rate_ci95(*, spike_counts: np.ndarray, duration_sec: float) -> list[list[float]]:
        """
        Approximate 95% intervals for per-channel firing rates.

        Counts are modeled as Poisson observations.  The normal approximation is
        intentionally simple and deterministic; downstream consumers should use
        it as calibration uncertainty, not as a formal inferential guarantee.
        """
        duration = max(float(duration_sec), 1e-12)
        intervals: list[list[float]] = []
        for count in np.asarray(spike_counts, dtype=np.float64):
            rate = count / duration
            half_width = 1.96 * np.sqrt(max(count, 0.0)) / duration
            intervals.append([float(max(0.0, rate - half_width)), float(rate + half_width)])
        return intervals

    @staticmethod
    def _binomial_ci95(*, hit_count: int, trial_count: int) -> list[float]:
        """Wilson 95% interval for a stim-response hit probability."""
        if trial_count <= 0:
            return [0.0, 0.0]
        z = 1.96
        n = float(trial_count)
        p = float(hit_count) / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
        return [float(max(0.0, center - half)), float(min(1.0, center + half))]

    @staticmethod
    def _latency_ci95(latencies: list[int]) -> list[int]:
        """Approximate 95% interval for response latency from observed hits."""
        if not latencies:
            return [0, 0]
        values = np.asarray(latencies, dtype=np.float64)
        if len(values) == 1:
            latency = int(round(float(values[0])))
            return [latency, latency]
        center = float(np.median(values))
        half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
        return [max(0, int(np.floor(center - half_width))), int(np.ceil(center + half_width))]

    @staticmethod
    def _estimate_confidence(
        *,
        duration_sec: float,
        spike_counts: np.ndarray,
        sample_count: int,
        stim_count: int,
        burst_count: np.ndarray,
    ) -> tuple[list[float], dict[str, float]]:
        """
        Estimate simple bounded confidence scores for calibrated profile fields.

        These scores are not statistical guarantees; they are operational
        guardrails for downstream twin code and reviewers.  Scores increase with
        observation duration, spike support, sample support, and stimulation
        count, then stay in ``[0, 1]`` so profiles can be compared directly.
        """
        duration_confidence = 1.0 - np.exp(-duration_sec / 60.0)
        channel_confidence = (1.0 - np.exp(-spike_counts / 20.0)) * duration_confidence
        channel_confidence = np.clip(channel_confidence, 0.0, 1.0)

        active_channel_fraction = float(np.mean(spike_counts > 0.0)) if len(spike_counts) else 0.0
        mean_channel_confidence = float(np.mean(channel_confidence)) if len(channel_confidence) else 0.0
        sample_confidence = float(np.clip(sample_count / 25_000.0, 0.0, 1.0))
        stim_confidence = float(1.0 - np.exp(-max(stim_count, 0) / 20.0))
        burst_confidence = float(
            np.clip((1.0 - np.exp(-float(np.sum(burst_count)) / 10.0)) * duration_confidence, 0.0, 1.0)
        )

        field_confidence = {
            "baseline_rate_hz": float(np.clip(mean_channel_confidence, 0.0, 1.0)),
            "noise_std_sample_units": sample_confidence,
            "connectivity": float(np.clip(mean_channel_confidence * active_channel_fraction, 0.0, 1.0)),
            "stim_response": stim_confidence,
            "temporal_structure": burst_confidence,
        }
        return channel_confidence.astype(float).tolist(), field_confidence

    @staticmethod
    def _stim_pair_confidence(*, stim_count: int, hit_count: int) -> float:
        """Estimate support for one stimulated-channel to response-channel pair."""
        if stim_count <= 0 or hit_count <= 0:
            return 0.0
        trial_support = 1.0 - np.exp(-stim_count / 10.0)
        hit_support = 1.0 - np.exp(-hit_count / 3.0)
        return float(np.clip(trial_support * hit_support, 0.0, 1.0))

    @staticmethod
    def _estimate_topology(
        *,
        spike_counts: np.ndarray,
        channel_count: int,
        min_neuron_count: int,
        max_neuron_count: int,
        stim_response_probability: np.ndarray | None = None,
        connectivity: np.ndarray | None = None,
    ) -> tuple[int, list[float]]:
        """
        Estimate a coarse spatial tissue prior from available MEA evidence.

        Spike counts are the strongest direct evidence.  Stim-response support
        adds channels that are biologically recruitable even when their baseline
        firing is sparse, and connectivity adds channels that participate in
        functional assemblies.  This is still a coarse prior, but it uses the
        richer spatial observations already present in calibrated recordings.
        """
        counts = np.asarray(spike_counts, dtype=np.float64)
        if counts.shape != (channel_count,):
            return 0, [1.0 / channel_count] * channel_count

        activity_support = TwinProfile._normalize_support(counts)
        response_support = TwinProfile._topology_response_support(
            stim_response_probability,
            channel_count,
        )
        connectivity_support = TwinProfile._topology_connectivity_support(
            connectivity,
            channel_count,
        )
        supports = [activity_support]
        weights = [0.60]
        if response_support is not None:
            supports.append(response_support)
            weights.append(0.25)
        if connectivity_support is not None:
            supports.append(connectivity_support)
            weights.append(0.15)

        weight_array = np.asarray(weights, dtype=np.float64)
        weight_array /= np.sum(weight_array)
        density = np.zeros(channel_count, dtype=np.float64)
        for weight, support in zip(weight_array, supports, strict=False):
            density += weight * support
        density += 1e-6
        density /= np.sum(density)
        active_channels = int(np.count_nonzero(counts > 0.0))
        recruited_channels = int(np.count_nonzero(density > (1.5 / channel_count)))
        total_spikes = float(np.sum(np.maximum(counts, 0.0)))
        support_neurons = max(active_channels, recruited_channels) * 16 + int(np.sqrt(total_spikes) * 8.0)
        neuron_count = int(np.clip(support_neurons, min_neuron_count, max_neuron_count))
        return neuron_count, density.astype(float).tolist()

    @staticmethod
    def _normalize_support(values: np.ndarray) -> np.ndarray:
        """Normalize non-negative per-channel support with a smoothing floor."""
        support = np.asarray(values, dtype=np.float64)
        support = np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)
        support = np.maximum(support, 0.0)
        if np.sum(support) <= 0.0:
            return np.ones_like(support, dtype=np.float64) / max(1, support.size)
        positive = support[support > 0.0]
        support = support + max(float(np.mean(positive)) * 0.05, 1.0)
        return support / np.sum(support)

    @staticmethod
    def _topology_response_support(
        stim_response_probability: np.ndarray | None,
        channel_count: int,
    ) -> np.ndarray | None:
        """Convert pairwise stim responses into per-channel recruitability support."""
        if stim_response_probability is None:
            return None
        matrix = np.asarray(stim_response_probability, dtype=np.float64)
        if matrix.shape != (channel_count, channel_count):
            return None
        # Both outgoing recruitment and incoming response imply nearby viable
        # tissue, so combine row and column support symmetrically.
        support = np.maximum(0.0, matrix).sum(axis=0) + np.maximum(0.0, matrix).sum(axis=1)
        return TwinProfile._normalize_support(support)

    @staticmethod
    def _topology_connectivity_support(
        connectivity: np.ndarray | None,
        channel_count: int,
    ) -> np.ndarray | None:
        """Convert functional coupling strength into per-channel topology support."""
        if connectivity is None:
            return None
        matrix = np.asarray(connectivity, dtype=np.float64)
        if matrix.shape != (channel_count, channel_count):
            return None
        support = np.abs(matrix).sum(axis=0) + np.abs(matrix).sum(axis=1)
        return TwinProfile._normalize_support(support)
