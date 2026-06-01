"""
Biological digital-twin producer subprocess.

This module is the SDK integration boundary for the twin.  It mirrors the
process/shared-memory shape of ``_data_producer.py`` while replacing passive H5
replay with a stateful model that stimulation can perturb.
"""
from __future__ import annotations

import logging
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import override

import numpy as np

from ._base_producer import BaseProducer, BaseProducerWorker
from ._data_buffer import StimRecord
from ._data_producer import (
    DEFAULT_TICK_RATE_HZ,
    InterruptCommand,
    ShutdownCommand,
    STALE_THRESHOLD_NS,
    StimCommand,
)
from .twin import SurrogateTwinModel, TwinConfig, TwinProfile

_logger = logging.getLogger("cl.twin_producer")


def _producer_main(
    replay_file_path: str,
    start_timestamp: int,
    channel_count: int,
    frames_per_second: int,
    tick_rate_hz: int,
    accelerated_time: bool,
    command_queue: Queue,
    name_prefix: str,
) -> None:
    """Subprocess entry point used by ``BiologicalTwinProducer``."""
    try:
        producer = BiologicalTwinProducerWorker(
            replay_file_path   = replay_file_path,
            start_timestamp    = start_timestamp,
            channel_count      = channel_count,
            frames_per_second  = frames_per_second,
            tick_rate_hz       = tick_rate_hz,
            accelerated_time   = accelerated_time,
            command_queue      = command_queue,
            name_prefix        = name_prefix,
            config             = TwinConfig.from_env(),
        )
        producer.run()
    except Exception as e:
        import traceback
        print(f"Twin producer subprocess failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


class BiologicalTwinProducerWorker(BaseProducerWorker):
    """
    Worker process that advances the biological twin and fills shared memory.

    The current worker runs the first-generation ``SurrogateTwinModel``.  The
    comments and class boundary intentionally describe the north-star shape:
    later iterations can swap in an Izhikevich/population SNN, STDP, STP, and
    calibrated pharmacology while preserving producer commands and buffer writes.
    """

    def __init__(
        self,
        replay_file_path: str,
        start_timestamp: int,
        channel_count: int,
        frames_per_second: int,
        tick_rate_hz: int,
        accelerated_time: bool,
        command_queue: Queue,
        name_prefix: str,
        config: TwinConfig,
    ):
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
            tick_rate_hz      = tick_rate_hz,
            command_queue     = command_queue,
            name_prefix       = name_prefix,
        )
        self._start_timestamp = start_timestamp
        self._current_timestamp = start_timestamp
        self._accelerated_time = accelerated_time
        self._config = config
        profile = (
            TwinProfile.load(config.profile_path)
            if config.profile_path
            else TwinProfile.default(
                channel_count     = channel_count,
                frames_per_second = frames_per_second,
            )
        )
        self._model = SurrogateTwinModel(
            channel_count      = channel_count,
            frames_per_second  = frames_per_second,
            config             = config,
            profile            = profile,
        )
        self._queued_stims: list[StimCommand] = []
        self._channel_available_from = np.full(channel_count, start_timestamp, dtype=np.int64)

    def run(self) -> None:
        """Main producer loop: process commands, advance biology, write frames."""
        BaseProducerWorker.set_process_priority()
        self.attach_buffer()
        BaseProducerWorker.disable_gc()

        assert self._buffer is not None
        self._running = True
        self._buffer.producer_ready = True
        start_wall_ns = time.perf_counter_ns()
        tick_count = 0

        _logger.info(
            "Twin producer started: %d frames/tick, accelerated=%s, plasticity=%s",
            self._frames_per_tick,
            self._accelerated_time,
            self._config.plasticity_mode,
        )

        while self._running:
            if self._buffer.pause_flag:
                time.sleep(0.01)
                start_wall_ns = time.perf_counter_ns()
                tick_count = 0
                continue

            if self._buffer.shutdown_flag:
                break

            self._process_commands()

            if self._accelerated_time:
                requested_ts = self._buffer.requested_timestamp
                while self._current_timestamp >= requested_ts:
                    # Keep draining commands while accelerated time is parked.
                    # Closed-loop user code commonly queues a stim and then
                    # immediately requests future frames; without this drain,
                    # the producer can wake on the read request before the
                    # multiprocessing feeder has made the stim visible.
                    self._process_commands()
                    if self._check_heartbeat_stale():
                        time.sleep(0.01)
                        continue
                    if self._buffer.shutdown_flag:
                        self._running = False
                        break
                    time.sleep(0.0001)
                    requested_ts = self._buffer.requested_timestamp
                if not self._running:
                    break
                self._process_commands()
            elif tick_count % 10 == 0:
                while self._check_heartbeat_stale():
                    time.sleep(0.01)

            from_ts = self._current_timestamp
            to_ts = from_ts + self._frames_per_tick
            stims = self._process_stims(from_ts, to_ts)
            frames, spikes = self._model.render(from_ts, self._frames_per_tick)

            self.write_spikes_to_buffer(spikes)
            self.write_stims_to_buffer(stims)
            # Frames advance the buffer's write timestamp, which consumers use
            # as "data ready".  Write event side-buffers first so a read that
            # waits for frames can immediately see the corresponding spikes and
            # stims for that same biological interval.
            self._buffer.write_frames(frames, from_ts)

            if not self._accelerated_time:
                self.sleep_until_next_tick(start_wall_ns, tick_count)

            self._current_timestamp = to_ts
            tick_count += 1

        self.cleanup()

    def _check_heartbeat_stale(self) -> bool:
        """Pause twin advancement while the main process appears debugger-stopped."""
        if self._buffer is None:
            return False
        heartbeat_ns = self._buffer.main_process_heartbeat_ns
        if heartbeat_ns == 0:
            return False
        return (time.perf_counter_ns() - heartbeat_ns) > STALE_THRESHOLD_NS

    def _process_commands(self) -> None:
        """Drain main-process commands into the worker's local queues."""
        while True:
            cmd = self.get_next_command()
            if cmd is None:
                break
            if isinstance(cmd, StimCommand) or all(
                hasattr(cmd, attr) for attr in ("timestamp", "channel", "end_timestamp")
            ):
                self._queued_stims.append(cmd)
                self._queued_stims.sort()
                self._channel_available_from[cmd.channel] = cmd.end_timestamp
            elif isinstance(cmd, InterruptCommand) or all(
                hasattr(cmd, attr) for attr in ("timestamp", "channels")
            ):
                channels = set(cmd.channels)
                self._queued_stims = [
                    stim for stim in self._queued_stims
                    if not (stim.channel in channels and stim.timestamp >= cmd.timestamp)
                ]
                for channel in channels:
                    self._channel_available_from[channel] = cmd.timestamp
            elif isinstance(cmd, ShutdownCommand):
                self._running = False

    def _process_stims(self, from_timestamp: int, to_timestamp: int) -> list[StimRecord]:
        """
        Deliver due stimulation to the model and return buffer stim records.

        Accelerated mode accepts slightly late commands because closed-loop code
        can enqueue stimulation while the producer is waiting on requested time.
        """
        due: list[StimCommand] = []
        remaining: list[StimCommand] = []
        late_tolerance = max(self._frames_per_tick, 200) if self._accelerated_time else 50
        for stim in self._queued_stims:
            if stim.timestamp < to_timestamp:
                if self._accelerated_time or stim.timestamp >= from_timestamp - late_tolerance:
                    due.append(stim)
            else:
                remaining.append(stim)
        self._queued_stims = remaining

        records: list[StimRecord] = []
        for stim in due:
            actual_ts = stim.timestamp if self._accelerated_time else max(stim.timestamp, from_timestamp)
            record = StimRecord(timestamp=actual_ts, channel=stim.channel)
            self._model.apply_stim(record, current_uA=stim.current_uA)
            records.append(record)
        return records


class BiologicalTwinProducer(BaseProducer):
    """Main-process interface for the biological twin subprocess."""

    def __init__(
        self,
        replay_file_path: str | Path,
        start_timestamp: int,
        channel_count: int,
        frames_per_second: int,
        tick_rate_hz: int = DEFAULT_TICK_RATE_HZ,
        accelerated_time: bool = False,
    ):
        super().__init__(
            replay_file_path  = replay_file_path,
            channel_count     = channel_count,
            frames_per_second = frames_per_second,
        )
        self._start_timestamp = start_timestamp
        self._tick_rate_hz = tick_rate_hz
        self._accelerated_time = accelerated_time
        self._channel_available_from = np.full(channel_count, start_timestamp, dtype=np.int64)

    @override
    def start(self, timeout: float = 15.0, start_timestamp: int = 0) -> None:
        """Start the twin using its configured CL timestamp."""
        super().start(timeout=timeout, start_timestamp=self._start_timestamp)

    @override
    def _create_process(self) -> Process:
        return Process(
            target = _producer_main,
            args   = (
                self._replay_file_path,
                self._start_timestamp,
                self._channel_count,
                self._frames_per_second,
                self._tick_rate_hz,
                self._accelerated_time,
                self._command_queue,
                self._name_prefix,
            ),
            daemon = True,
            name   = "cl-biological-twin-producer",
        )

    @override
    def _send_shutdown(self) -> None:
        self._command_queue.put(ShutdownCommand())

    def queue_stim(
        self,
        timestamp: int,
        channel: int,
        end_timestamp: int,
        current_uA: float = 1.0,
    ) -> None:
        """Queue an electrode stimulation for the twin's forward MEA path."""
        self._command_queue.put(StimCommand(
            timestamp     = timestamp,
            channel       = channel,
            end_timestamp = end_timestamp,
            current_uA    = current_uA,
        ))
        self._channel_available_from[channel] = end_timestamp

        # In accelerated mode, stimulation is often followed immediately by a
        # read for the affected future window.  Nudge the worker far enough to
        # ingest the command before that read can advance past the stim and make
        # the causal artifact/spike response unrecoverable.
        if self._accelerated_time and self._buffer is not None:
            time.sleep(0.001)
            initial_stim_count = self._buffer._read_header().stim_count
            requested_ts = max(self._buffer.requested_timestamp, timestamp + 1)
            for _ in range(4):
                self._buffer.requested_timestamp = requested_ts
                self._buffer.wait_for_timestamp(requested_ts, timeout_seconds=2.0)
                if self._buffer._read_header().stim_count > initial_stim_count:
                    break
                # If the worker advanced before the queue feeder exposed the
                # command, ask for one more tick so the late command can still
                # be consumed and coupled into future simulated biology.
                requested_ts = max(requested_ts + 1, self._buffer.write_timestamp + 1)

    def interrupt_channels(self, timestamp: int, channels: list[int]) -> None:
        """Cancel queued stimulation on selected channels."""
        self._command_queue.put(InterruptCommand(timestamp=timestamp, channels=channels))
        for channel in channels:
            self._channel_available_from[channel] = timestamp

    def get_channel_available_from(self, channel: int) -> int:
        """Return when a channel becomes stimulation-available."""
        return int(self._channel_available_from[channel])

    def set_channel_available_from(self, channel: int, timestamp: int) -> None:
        """Set channel availability for sync-compatible local bookkeeping."""
        self._channel_available_from[channel] = timestamp
