from __future__ import annotations

import argparse

from cl1_snn_reset import temporal_order_discrimination

from common import run_task_cli


def build_regime(args: argparse.Namespace):
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


if __name__ == "__main__":
    run_task_cli(
        description="Run the temporal order discrimination SNN reset task grid.",
        build_regime=build_regime,
    )
