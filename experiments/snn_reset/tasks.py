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
    pattern_discrimination,
    temporal_order_discrimination,
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
    "pattern_discrimination": _pattern_discrimination,
    "temporal_order_discrimination": _temporal_order_discrimination,
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
