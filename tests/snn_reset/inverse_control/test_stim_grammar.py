import numpy as np

from cl1_snn_reset.inverse_control import (
    AntiSTDPPairingBlock,
    ActuatorPositiveControlBlock,
    StimConstraints,
    StimProgram,
    TaskInputDriveBlock,
    sample_stim_programs,
    stim_program_features,
)
from cl1_snn_reset.inverse_control.stim_grammar import block_from_json, block_to_json


def test_block_serialization_round_trips():
    block = AntiSTDPPairingBlock(
        first_electrode=9,
        second_electrode=8,
        delay_ms=5.0,
        inter_pair_interval_ms=50.0,
        repeat_count=3,
        amplitude_uA=1.0,
    )

    loaded = block_from_json(block_to_json(block))

    assert loaded == block


def test_sampling_is_seed_stable_and_features_have_fixed_shape():
    constraints = StimConstraints(max_energy_cost=1.0, max_total_duration_s=0.5)
    a = sample_stim_programs(
        count=5,
        constraints=constraints,
        input_channel=8,
        target_channel=9,
        rng=np.random.default_rng(7),
        duration_s=(0.1, 0.2),
    )
    b = sample_stim_programs(
        count=5,
        constraints=constraints,
        input_channel=8,
        target_channel=9,
        rng=np.random.default_rng(7),
        duration_s=(0.1, 0.2),
    )

    assert [program.to_json() for program in a] == [program.to_json() for program in b]
    assert stim_program_features(a[0]).shape == stim_program_features(b[0]).shape


def test_program_json_round_trip_preserves_constraints():
    program = StimProgram(
        blocks=(
            AntiSTDPPairingBlock(
                first_electrode=9,
                second_electrode=8,
                delay_ms=2.0,
                inter_pair_interval_ms=40.0,
                repeat_count=2,
                amplitude_uA=0.8,
            ),
        ),
        constraints=StimConstraints(max_energy_cost=1.0),
        random_seed=12,
    )

    loaded = StimProgram.from_json(program.to_json())

    assert loaded.to_json() == program.to_json()


def test_positive_control_block_serializes():
    block = ActuatorPositiveControlBlock(
        electrodes=tuple(range(4)),
        frequency_hz=100.0,
        duration_s=0.5,
        amplitude_uA=0.8,
    )

    loaded = block_from_json(block_to_json(block))

    assert loaded == block


def test_task_input_drive_block_serializes():
    block = TaskInputDriveBlock(
        electrodes=(8,),
        frequency_hz=110.0,
        duration_s=0.5,
        amplitude_uA=12.0,
        drive_mode="single_input",
    )

    loaded = block_from_json(block_to_json(block))

    assert loaded == block
