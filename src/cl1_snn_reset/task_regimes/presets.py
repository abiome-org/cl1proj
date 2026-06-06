from __future__ import annotations

from .specs import ProbeSpec, TaskRegime, TrainingTrialSpec, stim_event_ms


def evoked_channel_response(
    *,
    input_channel: int = 8,
    target_channel: int = 8,
    input_current_uA: float = 80.0,
    pulse_width_us: int = 160,
    duration_ms: float = 80.0,
    response_window_ms: tuple[float, float] = (0.0, 35.0),
    criterion_score: float = 0.5,
    eval_repetitions: int = 16,
) -> TaskRegime:
    """Direct evoked response task: channel A should respond to A stimulation."""
    input_event = stim_event_ms(0.0, (input_channel,), input_current_uA, pulse_width_us)
    return TaskRegime(
        name="evoked_channel_response",
        description="Stimulate one electrode and measure same-channel evoked activity over sham.",
        training_trials=(),
        probes=(
            ProbeSpec(
                name="input",
                events=(input_event,),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="positive",
            ),
            ProbeSpec(
                name="sham",
                events=(),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="negative",
            ),
        ),
        criterion_score=criterion_score,
        max_training_repetitions=0,
        eval_repetitions=eval_repetitions,
    )


def conditioned_electrode_association(
    *,
    input_channel: int = 8,
    target_channel: int = 9,
    input_current_uA: float = 80.0,
    target_current_uA: float = 80.0,
    pair_delay_ms: float = 12.0,
    duration_ms: float = 100.0,
    response_window_ms: tuple[float, float] = (4.0, 45.0),
    criterion_score: float = 0.4,
    max_training_repetitions: int = 80,
    eval_repetitions: int = 16,
) -> TaskRegime:
    """A -> B conditioned association with input-only test probes."""
    input_event = stim_event_ms(0.0, (input_channel,), input_current_uA)
    target_event = stim_event_ms(pair_delay_ms, (target_channel,), target_current_uA)
    return TaskRegime(
        name="conditioned_electrode_association",
        description="Train input electrode A paired with target electrode B; test A-alone response at B.",
        training_trials=(
            TrainingTrialSpec(
                name="paired_a_to_b",
                events=(input_event, target_event),
                duration_ms=duration_ms,
            ),
        ),
        probes=(
            ProbeSpec(
                name="a_to_b",
                events=(input_event,),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="positive",
            ),
            ProbeSpec(
                name="sham_b",
                events=(),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="negative",
            ),
        ),
        criterion_score=criterion_score,
        max_training_repetitions=max_training_repetitions,
        eval_repetitions=eval_repetitions,
    )


def delayed_conditioned_response(
    *,
    input_channel: int = 8,
    target_channel: int = 55,
    input_current_uA: float = 80.0,
    target_current_uA: float = 90.0,
    pair_delay_ms: float = 75.0,
    duration_ms: float = 180.0,
    response_window_ms: tuple[float, float] = (50.0, 150.0),
    criterion_score: float = 0.35,
    max_training_repetitions: int = 100,
    eval_repetitions: int = 16,
) -> TaskRegime:
    """Delayed A -> B association requiring a later response window."""
    input_event = stim_event_ms(0.0, (input_channel,), input_current_uA)
    target_event = stim_event_ms(pair_delay_ms, (target_channel,), target_current_uA)
    return TaskRegime(
        name="delayed_conditioned_response",
        description="Train A then delayed B; test whether A evokes B in a delayed window.",
        training_trials=(
            TrainingTrialSpec(
                name="delayed_pair",
                events=(input_event, target_event),
                duration_ms=duration_ms,
            ),
        ),
        probes=(
            ProbeSpec(
                name="a_delayed_b",
                events=(input_event,),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="positive",
            ),
            ProbeSpec(
                name="sham_delayed_b",
                events=(),
                target_channels=(target_channel,),
                response_window_ms=response_window_ms,
                duration_ms=duration_ms,
                expected="negative",
            ),
        ),
        criterion_score=criterion_score,
        max_training_repetitions=max_training_repetitions,
        eval_repetitions=eval_repetitions,
    )


def pattern_discrimination(
    *,
    input_a: int = 8,
    input_b: int = 16,
    target_a: int = 9,
    target_b: int = 17,
    input_current_uA: float = 80.0,
    target_current_uA: float = 80.0,
    pair_delay_ms: float = 12.0,
    duration_ms: float = 110.0,
    response_window_ms: tuple[float, float] = (4.0, 50.0),
    criterion_score: float = 0.35,
    max_training_repetitions: int = 100,
    eval_repetitions: int = 16,
) -> TaskRegime:
    """Two input electrodes should evoke different target electrodes."""
    a_event = stim_event_ms(0.0, (input_a,), input_current_uA)
    b_event = stim_event_ms(0.0, (input_b,), input_current_uA)
    target_a_event = stim_event_ms(pair_delay_ms, (target_a,), target_current_uA)
    target_b_event = stim_event_ms(pair_delay_ms, (target_b,), target_current_uA)
    return TaskRegime(
        name="pattern_discrimination",
        description="Train A1->B1 and A2->B2; score correct targets over crossed targets.",
        training_trials=(
            TrainingTrialSpec("a_to_target_a", (a_event, target_a_event), duration_ms),
            TrainingTrialSpec("b_to_target_b", (b_event, target_b_event), duration_ms),
        ),
        probes=(
            ProbeSpec("a_correct", (a_event,), (target_a,), response_window_ms, duration_ms, "positive"),
            ProbeSpec("b_correct", (b_event,), (target_b,), response_window_ms, duration_ms, "positive"),
            ProbeSpec("a_crossed", (a_event,), (target_b,), response_window_ms, duration_ms, "negative"),
            ProbeSpec("b_crossed", (b_event,), (target_a,), response_window_ms, duration_ms, "negative"),
        ),
        criterion_score=criterion_score,
        max_training_repetitions=max_training_repetitions,
        eval_repetitions=eval_repetitions,
    )


def temporal_order_discrimination(
    *,
    channel_a: int = 8,
    channel_b: int = 16,
    target_ab: int = 9,
    target_ba: int = 17,
    input_current_uA: float = 80.0,
    target_current_uA: float = 80.0,
    order_gap_ms: float = 20.0,
    pair_delay_ms: float = 45.0,
    duration_ms: float = 140.0,
    response_window_ms: tuple[float, float] = (25.0, 95.0),
    criterion_score: float = 0.3,
    max_training_repetitions: int = 120,
    eval_repetitions: int = 16,
) -> TaskRegime:
    """Same electrodes in different temporal order map to different targets."""
    a0 = stim_event_ms(0.0, (channel_a,), input_current_uA)
    b0 = stim_event_ms(0.0, (channel_b,), input_current_uA)
    a1 = stim_event_ms(order_gap_ms, (channel_a,), input_current_uA)
    b1 = stim_event_ms(order_gap_ms, (channel_b,), input_current_uA)
    target_ab_event = stim_event_ms(pair_delay_ms, (target_ab,), target_current_uA)
    target_ba_event = stim_event_ms(pair_delay_ms, (target_ba,), target_current_uA)
    ab_probe = (a0, b1)
    ba_probe = (b0, a1)
    return TaskRegime(
        name="temporal_order_discrimination",
        description="Train A-then-B and B-then-A to separate targets; score order-specific responses.",
        training_trials=(
            TrainingTrialSpec("ab_to_target_ab", (*ab_probe, target_ab_event), duration_ms),
            TrainingTrialSpec("ba_to_target_ba", (*ba_probe, target_ba_event), duration_ms),
        ),
        probes=(
            ProbeSpec("ab_correct", ab_probe, (target_ab,), response_window_ms, duration_ms, "positive"),
            ProbeSpec("ba_correct", ba_probe, (target_ba,), response_window_ms, duration_ms, "positive"),
            ProbeSpec("ab_crossed", ab_probe, (target_ba,), response_window_ms, duration_ms, "negative"),
            ProbeSpec("ba_crossed", ba_probe, (target_ab,), response_window_ms, duration_ms, "negative"),
        ),
        criterion_score=criterion_score,
        max_training_repetitions=max_training_repetitions,
        eval_repetitions=eval_repetitions,
    )
