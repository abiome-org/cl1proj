from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .profile import TwinProfile


@dataclass(frozen=True)
class TwinValidationReport:
    """
    Compact report comparing simulated activity against a calibrated profile.

    The report intentionally mirrors fields that `TwinProfile` can estimate from
    biological recordings.  This keeps validation anchored to observable CL1
    data instead of hidden simulator state.
    """

    duration_frames: int
    spike_count: int
    rate_mae_hz: float
    isi_median_mae_frames: float
    burst_rate_mae_hz: float
    stim_response_probability_mae: float
    stim_response_latency_mae_frames: float
    artifact_blank_fraction: float
    artifact_peak_abs_sample_units: float
    passed: bool
    tolerances: dict[str, float]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and notebooks."""
        return asdict(self)


class TwinValidator:
    """Validation gates for profile-vs-simulation biological twin checks."""

    DEFAULT_TOLERANCES = {
        "rate_mae_hz": 1.0,
        "isi_median_mae_frames": 250.0,
        "burst_rate_mae_hz": 0.5,
        "stim_response_probability_mae": 0.25,
        "stim_response_latency_mae_frames": 25.0,
        "artifact_blank_fraction_min": 0.75,
        "artifact_peak_abs_sample_units_min": 100.0,
    }

    @classmethod
    def validate_spikes(
        cls,
        *,
        profile: TwinProfile,
        spikes: Iterable[Any],
        duration_frames: int | None = None,
        burst_isi_frames: int | None = None,
        burst_min_spikes: int = 3,
        stims: Iterable[Any] | None = None,
        response_window_frames: int | None = None,
        raw_frames: np.ndarray | None = None,
        raw_frame_start: int = 0,
        artifact_blank_window_frames: int | None = None,
        artifact_threshold_sample_units: float | None = None,
        tolerances: dict[str, float] | None = None,
    ) -> TwinValidationReport:
        """
        Compare simulated spike events with the firing targets in a profile.

        Spike events may be dict-like records or SDK dataclass/object records as
        long as they expose `timestamp` and `channel`.  Tolerances are deliberately
        explicit so scientific workflows can tighten them as the twin matures.
        """
        channel_count = int(profile.channel_count)
        frames_per_second = int(profile.frames_per_second)
        resolved_duration = int(duration_frames or profile.duration_frames or 0)
        spike_frames_by_channel = cls._spike_frames_by_channel(
            spikes        = spikes,
            channel_count = channel_count,
        )
        if resolved_duration <= 0:
            resolved_duration = cls._infer_duration(spike_frames_by_channel)
        duration_sec = max(resolved_duration / frames_per_second, 1.0 / frames_per_second)

        simulated_rates = np.array([
            len(spike_frames_by_channel[ch]) / duration_sec
            for ch in range(channel_count)
        ], dtype=np.float64)
        target_rates = cls._profile_float_vector(
            profile.baseline_rate_hz_by_channel,
            channel_count,
        )

        (
            simulated_isi,
            simulated_burst_rate,
            simulated_burst_duration,
            simulated_burst_count,
        ) = TwinProfile._estimate_temporal_structure(
            spike_frames_by_channel = spike_frames_by_channel,
            channel_count           = channel_count,
            frames_per_second       = frames_per_second,
            duration_sec            = duration_sec,
            burst_isi_frames        = burst_isi_frames or max(1, int(round(frames_per_second * 0.1))),
            burst_min_spikes        = burst_min_spikes,
        )
        target_isi = cls._profile_int_vector(profile.isi_median_frames_by_channel, channel_count)
        target_burst_rate = cls._profile_float_vector(profile.burst_rate_hz_by_channel, channel_count)

        rate_mae = cls._mean_absolute_error(simulated_rates, target_rates)
        isi_mae = cls._mean_absolute_error(np.asarray(simulated_isi), target_isi)
        burst_rate_mae = cls._mean_absolute_error(np.asarray(simulated_burst_rate), target_burst_rate)
        (
            stim_probability_mae,
            stim_latency_mae,
            stim_metrics,
        ) = cls._stim_response_metrics(
            profile                 = profile,
            spike_frames_by_channel = spike_frames_by_channel,
            stims                   = stims,
            response_window_frames  = response_window_frames,
        )
        (
            artifact_blank_fraction,
            artifact_peak_abs,
            artifact_metrics,
        ) = cls._artifact_metrics(
            frames                          = raw_frames,
            stims                           = stims,
            raw_frame_start                 = raw_frame_start,
            channel_count                   = channel_count,
            artifact_blank_window_frames    = artifact_blank_window_frames,
            artifact_threshold_sample_units = artifact_threshold_sample_units,
        )
        resolved_tolerances = dict(cls.DEFAULT_TOLERANCES)
        if tolerances:
            resolved_tolerances.update(tolerances)

        artifact_gate_passed = (
            artifact_metrics["artifact_windows_evaluated"] == 0
            or (
                artifact_blank_fraction >= resolved_tolerances["artifact_blank_fraction_min"]
                and artifact_peak_abs >= resolved_tolerances["artifact_peak_abs_sample_units_min"]
            )
        )
        passed = (
            rate_mae <= resolved_tolerances["rate_mae_hz"]
            and isi_mae <= resolved_tolerances["isi_median_mae_frames"]
            and burst_rate_mae <= resolved_tolerances["burst_rate_mae_hz"]
            and stim_probability_mae <= resolved_tolerances["stim_response_probability_mae"]
            and stim_latency_mae <= resolved_tolerances["stim_response_latency_mae_frames"]
            and artifact_gate_passed
        )

        return TwinValidationReport(
            duration_frames       = resolved_duration,
            spike_count           = int(sum(len(v) for v in spike_frames_by_channel.values())),
            rate_mae_hz           = rate_mae,
            isi_median_mae_frames = isi_mae,
            burst_rate_mae_hz     = burst_rate_mae,
            stim_response_probability_mae = stim_probability_mae,
            stim_response_latency_mae_frames = stim_latency_mae,
            artifact_blank_fraction = artifact_blank_fraction,
            artifact_peak_abs_sample_units = artifact_peak_abs,
            passed                = passed,
            tolerances            = resolved_tolerances,
            metrics               = {
                "simulated_rate_hz_by_channel": simulated_rates.astype(float).tolist(),
                "simulated_isi_median_frames_by_channel": [int(v) for v in simulated_isi],
                "simulated_burst_rate_hz_by_channel": [float(v) for v in simulated_burst_rate],
                "simulated_burst_median_duration_frames_by_channel": [int(v) for v in simulated_burst_duration],
                "simulated_burst_spike_count_mean_by_channel": [float(v) for v in simulated_burst_count],
                **stim_metrics,
                **artifact_metrics,
            },
        )

    @classmethod
    def _stim_response_metrics(
        cls,
        *,
        profile: TwinProfile,
        spike_frames_by_channel: dict[int, np.ndarray],
        stims: Iterable[Any] | None,
        response_window_frames: int | None,
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Compare simulated stim-triggered responses against profile targets.

        The gate is opt-in by evidence: when no stims are supplied, or the
        profile contains no non-zero stim-response targets, the metric returns
        zero error and records that zero pairs were evaluated.
        """
        channel_count = int(profile.channel_count)
        target_probability = cls._profile_matrix(profile.stim_response_probability, channel_count)
        target_latency = cls._profile_int_matrix(profile.stim_response_latency_frames, channel_count)
        target_mask = target_probability > 0.0
        stim_events = list(stims or [])
        if not stim_events or not np.any(target_mask):
            return 0.0, 0.0, {
                "stim_response_pairs_evaluated": 0,
                "simulated_stim_response_probability": np.zeros(
                    (channel_count, channel_count), dtype=float,
                ).tolist(),
                "simulated_stim_response_latency_frames": np.zeros(
                    (channel_count, channel_count), dtype=int,
                ).tolist(),
            }

        window = int(response_window_frames or cls._default_response_window(profile))
        attempts = np.zeros(channel_count, dtype=np.int64)
        responses = np.zeros((channel_count, channel_count), dtype=np.int64)
        latency_sum = np.zeros((channel_count, channel_count), dtype=np.float64)

        for stim in stim_events:
            stim_channel = int(cls._event_value(stim, "channel"))
            stim_timestamp = int(cls._event_value(stim, "timestamp"))
            if not 0 <= stim_channel < channel_count:
                continue
            attempts[stim_channel] += 1
            window_start = stim_timestamp
            window_stop = stim_timestamp + window
            for response_channel, frames in spike_frames_by_channel.items():
                if len(frames) == 0:
                    continue
                first_index = int(np.searchsorted(frames, window_start, side="left"))
                if first_index < len(frames) and frames[first_index] <= window_stop:
                    responses[stim_channel, response_channel] += 1
                    latency_sum[stim_channel, response_channel] += float(frames[first_index] - stim_timestamp)

        probability = np.zeros((channel_count, channel_count), dtype=np.float64)
        latency = np.zeros((channel_count, channel_count), dtype=np.int64)
        rows_with_attempts = attempts > 0
        if np.any(rows_with_attempts):
            probability[rows_with_attempts] = (
                responses[rows_with_attempts]
                / attempts[rows_with_attempts, np.newaxis]
            )
        response_mask = responses > 0
        latency[response_mask] = np.rint(latency_sum[response_mask] / responses[response_mask]).astype(np.int64)

        evaluated = target_mask & (attempts[:, np.newaxis] > 0)
        if not np.any(evaluated):
            probability_mae = 0.0
            latency_mae = 0.0
        else:
            probability_mae = cls._mean_absolute_error(probability[evaluated], target_probability[evaluated])
            latency_evaluated = evaluated & (target_latency > 0)
            latency_mae = (
                cls._mean_absolute_error(latency[latency_evaluated], target_latency[latency_evaluated])
                if np.any(latency_evaluated)
                else 0.0
            )

        return probability_mae, latency_mae, {
            "stim_response_pairs_evaluated": int(np.count_nonzero(evaluated)),
            "simulated_stim_response_probability": probability.astype(float).tolist(),
            "simulated_stim_response_latency_frames": latency.astype(int).tolist(),
        }

    @classmethod
    def _artifact_metrics(
        cls,
        *,
        frames: np.ndarray | None,
        stims: Iterable[Any] | None,
        raw_frame_start: int,
        channel_count: int,
        artifact_blank_window_frames: int | None,
        artifact_threshold_sample_units: float | None,
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Measure whether stim-adjacent raw frames contain a blanking artifact.

        Real MEA recordings are briefly dominated by amplifier artifact after
        stimulation.  The twin should preserve that nuisance because downstream
        closed-loop code must tolerate or blank those windows.
        """
        if frames is None:
            return 1.0, 0.0, {
                "artifact_windows_evaluated": 0,
                "artifact_blank_fraction": 1.0,
                "artifact_peak_abs_sample_units": 0.0,
            }
        frame_block = np.asarray(frames)
        stim_events = list(stims or [])
        if frame_block.ndim != 2 or not stim_events:
            return 1.0, 0.0, {
                "artifact_windows_evaluated": 0,
                "artifact_blank_fraction": 1.0,
                "artifact_peak_abs_sample_units": 0.0,
            }

        window = max(1, int(artifact_blank_window_frames or 5))
        threshold = float(artifact_threshold_sample_units or 100.0)
        blanked = 0
        evaluated = 0
        peak_abs = 0.0
        for stim in stim_events:
            channel = int(cls._event_value(stim, "channel"))
            timestamp = int(cls._event_value(stim, "timestamp"))
            if not 0 <= channel < min(channel_count, frame_block.shape[1]):
                continue
            start = max(0, timestamp - int(raw_frame_start))
            stop = min(frame_block.shape[0], start + window)
            if start >= stop:
                continue
            segment = np.abs(frame_block[start:stop, channel].astype(np.float64))
            evaluated += 1
            local_peak = float(np.max(segment)) if segment.size else 0.0
            peak_abs = max(peak_abs, local_peak)
            if local_peak >= threshold:
                blanked += 1

        blank_fraction = float(blanked / evaluated) if evaluated else 1.0
        return blank_fraction, peak_abs, {
            "artifact_windows_evaluated": int(evaluated),
            "artifact_blank_fraction": blank_fraction,
            "artifact_peak_abs_sample_units": peak_abs,
        }

    @staticmethod
    def _spike_frames_by_channel(
        *,
        spikes: Iterable[Any],
        channel_count: int,
    ) -> dict[int, np.ndarray]:
        """Group arbitrary spike records by channel as sorted frame arrays."""
        grouped: dict[int, list[int]] = {ch: [] for ch in range(channel_count)}
        for spike in spikes:
            channel = int(TwinValidator._event_value(spike, "channel"))
            timestamp = int(TwinValidator._event_value(spike, "timestamp"))
            if 0 <= channel < channel_count:
                grouped[channel].append(timestamp)
        return {
            ch: np.asarray(sorted(frames), dtype=np.int64)
            for ch, frames in grouped.items()
        }

    @staticmethod
    def _event_value(event: Any, key: str) -> Any:
        """Read a field from either dict-like or object-like event records."""
        if isinstance(event, dict):
            return event[key]
        return getattr(event, key)

    @staticmethod
    def _infer_duration(spike_frames_by_channel: dict[int, np.ndarray]) -> int:
        """Infer a minimal validation duration when the profile does not specify one."""
        latest = 0
        for frames in spike_frames_by_channel.values():
            if len(frames):
                latest = max(latest, int(frames[-1]))
        return latest + 1

    @staticmethod
    def _profile_float_vector(values: list[float], channel_count: int) -> np.ndarray:
        """Return a float profile vector with a zero fallback."""
        if len(values) != channel_count:
            return np.zeros(channel_count, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _profile_int_vector(values: list[int], channel_count: int) -> np.ndarray:
        """Return an integer profile vector with a zero fallback."""
        if len(values) != channel_count:
            return np.zeros(channel_count, dtype=np.int64)
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _profile_matrix(values: list[list[float]], channel_count: int) -> np.ndarray:
        """Return a square float profile matrix with a zero fallback."""
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (channel_count, channel_count):
            return np.zeros((channel_count, channel_count), dtype=np.float64)
        return matrix

    @staticmethod
    def _profile_int_matrix(values: list[list[int]], channel_count: int) -> np.ndarray:
        """Return a square integer profile matrix with a zero fallback."""
        matrix = np.asarray(values, dtype=np.int64)
        if matrix.shape != (channel_count, channel_count):
            return np.zeros((channel_count, channel_count), dtype=np.int64)
        return matrix

    @staticmethod
    def _default_response_window(profile: TwinProfile) -> int:
        """Choose a response window from profile latencies or a 10 ms fallback."""
        latencies = np.asarray(profile.stim_response_latency_frames, dtype=np.int64)
        positive = latencies[latencies > 0]
        if positive.size:
            return max(1, int(np.max(positive) * 2))
        return max(1, int(round(profile.frames_per_second * 0.01)))

    @staticmethod
    def _mean_absolute_error(observed: np.ndarray, target: np.ndarray) -> float:
        """Compute MAE while preserving zero-target channels in the gate."""
        if observed.shape != target.shape or observed.size == 0:
            return 0.0
        return float(np.mean(np.abs(observed.astype(np.float64) - target.astype(np.float64))))
