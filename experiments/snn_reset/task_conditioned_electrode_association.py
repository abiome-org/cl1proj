from __future__ import annotations

import argparse

from cl1_snn_reset import conditioned_electrode_association

from common import run_task_cli


def build_regime(args: argparse.Namespace):
    return conditioned_electrode_association(
        input_channel=args.input_channel,
        target_channel=args.target_channel,
        input_current_uA=args.input_current_uA,
        target_current_uA=args.target_current_uA,
        criterion_score=args.criterion_score if args.criterion_score is not None else 0.4,
        max_training_repetitions=args.training_repetitions or 80,
        eval_repetitions=args.eval_repetitions or 16,
    )


if __name__ == "__main__":
    run_task_cli(
        description="Run the conditioned electrode association SNN reset task grid.",
        build_regime=build_regime,
    )
