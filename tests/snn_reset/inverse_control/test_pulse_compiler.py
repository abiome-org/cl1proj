import pytest

from cl1_snn_reset.inverse_control import (
    InvalidStimProgramError,
    compile_program_to_stim_events,
    estimate_energy_cost,
)
from cl1_snn_reset.inverse_control.blocks import (
    ActuatorPositiveControlBlock,
    AntiSTDPPairingBlock,
    StimConstraints,
    StimProgram,
    TaskInputDriveBlock,
)


def test_compiler_emits_ordered_events_and_energy():
    program = StimProgram(
        blocks=(
            AntiSTDPPairingBlock(
                first_electrode=9,
                second_electrode=8,
                delay_ms=5.0,
                inter_pair_interval_ms=50.0,
                repeat_count=3,
                amplitude_uA=1.0,
                duration_s=0.2,
            ),
        ),
        constraints=StimConstraints(max_energy_cost=1.0, max_total_duration_s=1.0),
        random_seed=1,
    )

    events = compile_program_to_stim_events(program)

    assert events
    assert [event.time_us for event in events] == sorted(event.time_us for event in events)
    assert estimate_energy_cost(program, events) > 0.0


def test_compiler_rejects_amplitude_and_energy_violations():
    program = StimProgram(
        blocks=(
            AntiSTDPPairingBlock(
                first_electrode=9,
                second_electrode=8,
                delay_ms=5.0,
                inter_pair_interval_ms=10.0,
                repeat_count=10,
                amplitude_uA=3.0,
                duration_s=0.2,
            ),
        ),
        constraints=StimConstraints(max_amplitude_uA=1.0, max_energy_cost=1.0),
    )

    with pytest.raises(InvalidStimProgramError):
        compile_program_to_stim_events(program)


def test_compiler_expands_actuator_positive_control():
    program = StimProgram(
        blocks=(
            ActuatorPositiveControlBlock(
                electrodes=(0, 1, 2, 3),
                frequency_hz=100.0,
                duration_s=0.1,
                amplitude_uA=0.8,
            ),
        ),
        constraints=StimConstraints(
            max_amplitude_uA=0.8,
            max_energy_cost=1.0,
            max_total_duration_s=0.2,
            max_events_per_electrode=20,
        ),
    )

    events = compile_program_to_stim_events(program)

    assert len(events) == 10
    assert all(event.channels == (0, 1, 2, 3) for event in events)


def test_compiler_expands_task_input_drive():
    program = StimProgram(
        blocks=(
            TaskInputDriveBlock(
                electrodes=(8,),
                frequency_hz=100.0,
                duration_s=0.1,
                amplitude_uA=8.0,
            ),
        ),
        constraints=StimConstraints(
            max_amplitude_uA=16.0,
            max_energy_cost=1.0,
            max_total_duration_s=0.2,
            max_events_per_electrode=20,
        ),
    )

    events = compile_program_to_stim_events(program)

    assert len(events) == 10
    assert all(event.channels == (8,) for event in events)
