"""Single entrypoint for every SNN reset task grid.

Each task is one entry in ``TASK_BUILDERS``: a function that turns parsed CLI
args into a library ``TaskRegime``. The task-identity choices that live here
(criterion score, training length, channel wiring) are experiment decisions; the
mechanism they configure lives in ``cl1_snn_reset``. Run one task with
``tasks.py --task <name>``; ``run_grid.py`` imports this registry to run a suite.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from cl1_snn_reset import (
    conditioned_electrode_association,
    delayed_conditioned_response,
    evoked_channel_response,
    multi_association_mapping,
    overlapping_shared_input_association,
    overlapping_shared_target_association,
    pattern_discrimination,
    temporal_order_discrimination,
    xor_electrode_classification,
)

from common import RESULTS_DIR, run_task_grid, task_parser


def _evoked_channel_response(args: argparse.Namespace):
    return evoked_channel_response(
        input_channel=args.input_channel,
        target_channel=args.input_channel,
        input_current_uA=args.input_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.5,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _conditioned_electrode_association(args: argparse.Namespace):
    return conditioned_electrode_association(
        input_channel=args.input_channel,
        target_channel=args.target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.4,
        max_training_repetitions=args.training_repetitions or 80,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _delayed_conditioned_response(args: argparse.Namespace):
    return delayed_conditioned_response(
        input_channel=args.input_channel,
        target_channel=args.target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.35,
        max_training_repetitions=args.training_repetitions or 100,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _pattern_discrimination(args: argparse.Namespace):
    return pattern_discrimination(
        input_a=args.input_channel,
        input_b=args.second_input_channel,
        target_a=args.target_channel,
        target_b=args.second_target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.35,
        max_training_repetitions=args.training_repetitions or 100,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _overlapping_shared_target_association(args: argparse.Namespace):
    return overlapping_shared_target_association(
        input_a=args.input_channel,
        input_b=args.second_input_channel,
        shared_target=args.target_channel,
        distractor_target=args.second_target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.35,
        max_training_repetitions=args.training_repetitions or 100,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _overlapping_shared_input_association(args: argparse.Namespace):
    return overlapping_shared_input_association(
        input_channel=args.input_channel,
        context_channel=args.second_input_channel,
        target_a=args.target_channel,
        target_context=args.second_target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.25,
        max_training_repetitions=args.training_repetitions or 120,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _multi_association_mapping(args: argparse.Namespace):
    return multi_association_mapping(
        input_channels=(
            args.input_channel,
            args.second_input_channel,
            args.third_input_channel,
            args.fourth_input_channel,
        ),
        target_channels=(
            args.target_channel,
            args.second_target_channel,
            args.third_target_channel,
            args.fourth_target_channel,
        ),
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.3,
        max_training_repetitions=args.training_repetitions or 120,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _xor_electrode_classification(args: argparse.Namespace):
    return xor_electrode_classification(
        input_a=args.input_channel,
        input_b=args.second_input_channel,
        xor_target=args.target_channel,
        conjunction_target=args.second_target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.25,
        max_training_repetitions=args.training_repetitions or 140,
        eval_repetitions=args.eval_repetitions or 16,
    )


def _temporal_order_discrimination(args: argparse.Namespace):
    return temporal_order_discrimination(
        channel_a=args.input_channel,
        channel_b=args.second_input_channel,
        target_ab=args.target_channel,
        target_ba=args.second_target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.3,
        max_training_repetitions=args.training_repetitions or 120,
        eval_repetitions=args.eval_repetitions or 16,
    )


TASK_BUILDERS = {
    "evoked_channel_response": _evoked_channel_response,
    "conditioned_electrode_association": _conditioned_electrode_association,
    "delayed_conditioned_response": _delayed_conditioned_response,
    "multi_association_mapping": _multi_association_mapping,
    "overlapping_shared_input_association": _overlapping_shared_input_association,
    "overlapping_shared_target_association": _overlapping_shared_target_association,
    "pattern_discrimination": _pattern_discrimination,
    "temporal_order_discrimination": _temporal_order_discrimination,
    "xor_electrode_classification": _xor_electrode_classification,
}


def main() -> None:
    parser = task_parser("Run one SNN reset task grid.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_BUILDERS))
    args = parser.parse_args()
    regime = TASK_BUILDERS[args.task](args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{regime.name}_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or RESULTS_DIR / run_id
    run_task_grid(args, regime, output_dir=output_dir, run_id=run_id)


if __name__ == "__main__":
    main()
