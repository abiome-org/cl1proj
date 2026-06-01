from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackStim:
    """A scheduled stimulation pulse produced by a closed-loop feedback pattern."""

    timestamp: int
    channel: int
    current_uA: float


class TwinFeedbackProtocol:
    """
    Generate structured or chaotic stimulation patterns for closed-loop tasks.

    Cortical Labs-style wetware training depends on the difference between
    predictable feedback and disorganizing feedback.  This helper keeps that
    distinction explicit while routing both cases through the same stimulation
    coupling path as normal SDK calls.
    """

    def __init__(self, channel_count: int, frames_per_second: int):
        self.channel_count = int(channel_count)
        self.frames_per_second = int(frames_per_second)

    def structured(
        self,
        *,
        timestamp: int,
        sensory_channel: int,
        motor_channel: int | None = None,
        current_uA: float = 1.0,
        pairing_delay_ms: float = 8.0,
    ) -> list[FeedbackStim]:
        """
        Return a predictable local sensory-to-motor feedback pair.

        The first pulse represents stable sensory feedback.  The optional second
        pulse lands after a fixed delay on a motor/output channel so STDP-capable
        engines can reinforce a repeatable causal path.
        """
        sensory = self._channel(sensory_channel)
        events = [FeedbackStim(timestamp=int(timestamp), channel=sensory, current_uA=abs(float(current_uA)))]
        if motor_channel is not None:
            delay_frames = self._ms_to_frames(pairing_delay_ms)
            events.append(FeedbackStim(
                timestamp  = int(timestamp) + delay_frames,
                channel    = self._channel(motor_channel),
                current_uA = abs(float(current_uA)) * 0.75,
            ))
        return events

    def chaotic(
        self,
        *,
        timestamp: int,
        current_uA: float = 0.35,
        burst_count: int = 3,
        inter_burst_ms: float = 2.0,
    ) -> list[FeedbackStim]:
        """
        Return a broad high-frequency stimulation pattern across the MEA.

        Chaotic feedback intentionally lacks a stable channel relationship.  The
        deterministic channel staggering prevents every pulse from sharing one
        timestamp while still producing the amplifier-artifact and network-wide
        perturbation pressure expected from disorganizing feedback.
        """
        events: list[FeedbackStim] = []
        burst_stride = self._ms_to_frames(inter_burst_ms)
        for burst_index in range(max(1, int(burst_count))):
            burst_ts = int(timestamp) + burst_index * burst_stride
            for channel in range(self.channel_count):
                events.append(FeedbackStim(
                    timestamp  = burst_ts + channel % 4,
                    channel    = channel,
                    current_uA = abs(float(current_uA)),
                ))
        return events

    def from_outcome(
        self,
        *,
        timestamp: int,
        correct: bool,
        sensory_channel: int,
        motor_channel: int | None = None,
        current_uA: float = 1.0,
    ) -> list[FeedbackStim]:
        """Map a task outcome onto structured or chaotic feedback."""
        if correct:
            return self.structured(
                timestamp       = timestamp,
                sensory_channel = sensory_channel,
                motor_channel   = motor_channel,
                current_uA      = current_uA,
            )
        return self.chaotic(
            timestamp  = timestamp,
            current_uA = max(0.1, abs(float(current_uA)) * 0.35),
        )

    def _channel(self, channel: int) -> int:
        """Normalize arbitrary channel-like integers into the MEA channel range."""
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        return int(channel) % self.channel_count

    def _ms_to_frames(self, duration_ms: float) -> int:
        """Convert a millisecond interval to at least one CL sample frame."""
        return max(1, int(round(float(duration_ms) * self.frames_per_second / 1000.0)))
