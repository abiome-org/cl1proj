from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._data_buffer import SPIKE_SAMPLES_TOTAL, SpikeRecord


@dataclass(frozen=True)
class DetectionBlankingWindow:
    """Frame interval where amplifier artifact should suppress spike detection."""

    start_timestamp: int
    end_timestamp: int
    channel: int


class RollingThresholdSpikeDetector:
    """
    Detect MEA spikes from raw voltage frames using a local negative threshold.

    Real MEA stacks usually derive formal spike events from raw extracellular
    voltage, not from hidden simulator labels.  This detector keeps the twin on
    that side of the interface: SNN or surrogate models write biphasic EAPs into
    frames, then this component emits `SpikeRecord` objects when a channel
    crosses a rolling noise threshold outside artifact-blanking windows.
    """

    def __init__(
        self,
        *,
        frames_per_second: int,
        channel_count: int,
        threshold_sigma: float = 4.5,
        refractory_frames: int = 20,
        baseline_noise_std: np.ndarray | None = None,
    ):
        self.frames_per_second = frames_per_second
        self.channel_count = channel_count
        self.threshold_sigma = max(1.0, float(threshold_sigma))
        self.refractory_frames = max(1, int(refractory_frames))
        if baseline_noise_std is None:
            self.baseline_noise_std = np.ones(channel_count, dtype=np.float64)
        else:
            values = np.asarray(baseline_noise_std, dtype=np.float64)
            if values.shape != (channel_count,):
                values = np.ones(channel_count, dtype=np.float64)
            self.baseline_noise_std = np.maximum(values, 1e-3)
        self._last_detection_ts = np.full(channel_count, -1_000_000_000, dtype=np.int64)

    def detect(
        self,
        frames: np.ndarray,
        *,
        from_timestamp: int,
        blanking_windows: list[DetectionBlankingWindow] | None = None,
    ) -> list[SpikeRecord]:
        """Return formal spikes whose waveforms are present in `frames`."""
        if frames.size == 0:
            return []

        frame_values = np.asarray(frames, dtype=np.float64)
        noise_std = self._estimate_noise_std(frame_values)
        thresholds = -self.threshold_sigma * noise_std
        blanked = self._blanking_mask(
            frame_count=len(frame_values),
            from_timestamp=from_timestamp,
            windows=blanking_windows or [],
        )

        spikes: list[SpikeRecord] = []
        for channel in range(self.channel_count):
            trace = frame_values[:, channel]
            below = trace <= thresholds[channel]
            if not np.any(below):
                continue
            candidate_indices = np.flatnonzero(below & ~blanked[:, channel])
            for local_index in candidate_indices.tolist():
                timestamp = from_timestamp + int(local_index)
                if timestamp - self._last_detection_ts[channel] < self.refractory_frames:
                    continue
                # Align to the local trough so a threshold crossing reports the
                # EAP peak rather than the first shoulder sample.
                aligned_index = self._align_to_negative_peak(trace, local_index)
                timestamp = from_timestamp + aligned_index
                if timestamp - self._last_detection_ts[channel] < self.refractory_frames:
                    continue
                if blanked[aligned_index, channel]:
                    continue
                samples = self._extract_samples(frame_values[:, channel], aligned_index)
                spikes.append(SpikeRecord(
                    timestamp           = timestamp,
                    channel             = channel,
                    channel_mean_sample = float(samples.mean()),
                    samples             = samples,
                ))
                self._last_detection_ts[channel] = timestamp
        return spikes

    def _estimate_noise_std(self, frames: np.ndarray) -> np.ndarray:
        """
        Estimate per-channel noise with a robust floor from the culture profile.

        Median absolute deviation prevents one EAP or artifact tail from inflating
        the threshold so much that the detector misses the actual spike.
        """
        median = np.median(frames, axis=0)
        mad = np.median(np.abs(frames - median), axis=0)
        robust_std = mad / 0.67448975
        return np.maximum(robust_std, self.baseline_noise_std)

    def _blanking_mask(
        self,
        *,
        frame_count: int,
        from_timestamp: int,
        windows: list[DetectionBlankingWindow],
    ) -> np.ndarray:
        """Build a per-sample/channel artifact mask for this render block."""
        mask = np.zeros((frame_count, self.channel_count), dtype=bool)
        block_end = from_timestamp + frame_count
        for window in windows:
            start = max(window.start_timestamp, from_timestamp)
            end = min(window.end_timestamp, block_end)
            if end <= start or not 0 <= window.channel < self.channel_count:
                continue
            mask[start - from_timestamp:end - from_timestamp, window.channel] = True
        return mask

    @staticmethod
    def _align_to_negative_peak(trace: np.ndarray, local_index: int) -> int:
        """Move a crossing to the most negative sample in a short neighborhood."""
        start = max(0, local_index - 4)
        end = min(len(trace), local_index + 5)
        if end <= start:
            return int(local_index)
        return int(start + np.argmin(trace[start:end]))

    @staticmethod
    def _extract_samples(trace: np.ndarray, center_index: int) -> np.ndarray:
        """Return the 75-sample waveform window expected by `SpikeRecord`."""
        left = 25
        start = center_index - left
        end = start + SPIKE_SAMPLES_TOTAL
        samples = np.zeros(SPIKE_SAMPLES_TOTAL, dtype=np.float32)
        trace_start = max(0, start)
        trace_end = min(len(trace), end)
        if trace_start >= trace_end:
            return samples
        sample_start = trace_start - start
        sample_end = sample_start + (trace_end - trace_start)
        samples[sample_start:sample_end] = trace[trace_start:trace_end].astype(np.float32)
        return samples
