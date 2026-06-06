from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from common import EXPERIMENT_DIR, RESULTS_DIR, add_common_task_args, collect_task_summaries, git_commit


TASK_SCRIPTS = {
    "evoked_channel_response": "task_evoked_channel_response.py",
    "conditioned_electrode_association": "task_conditioned_electrode_association.py",
    "delayed_conditioned_response": "task_delayed_conditioned_response.py",
    "pattern_discrimination": "task_pattern_discrimination.py",
    "temporal_order_discrimination": "task_temporal_order_discrimination.py",
}
DEFAULT_TASKS = [
    "evoked_channel_response",
    "conditioned_electrode_association",
    "pattern_discrimination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular SNN reset grid-search task suite.")
    add_common_task_args(parser)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_SCRIPTS), default=DEFAULT_TASKS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def forwarded_args(args: argparse.Namespace, *, task_output_dir: Path) -> list[str]:
    values: list[str] = [
        "--output-dir",
        str(task_output_dir),
        "--neurons",
        str(args.neurons),
        "--mean-out-degree",
        str(args.mean_out_degree),
        "--max-out-degree",
        str(args.max_out_degree),
        "--local-candidate-multiplier",
        str(args.local_candidate_multiplier),
        "--background-noise-mv",
        str(args.background_noise_mv),
        "--spontaneous-rate-hz",
        str(args.spontaneous_rate_hz),
        "--homeostasis-rate",
        str(args.homeostasis_rate),
        "--homeostasis-interval-ms",
        str(args.homeostasis_interval_ms),
        "--backend",
        args.backend,
        "--build-workers",
        str(args.build_workers),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--workers",
        str(args.workers),
        "--executor",
        args.executor,
        "--warmup-s",
        str(args.warmup_s),
        "--consolidation-rest-s",
        str(args.consolidation_rest_s),
        "--input-current-uA",
        str(args.input_current_uA),
        "--target-current-uA",
        str(args.target_current_uA),
        "--input-channel",
        str(args.input_channel),
        "--target-channel",
        str(args.target_channel),
        "--second-input-channel",
        str(args.second_input_channel),
        "--second-target-channel",
        str(args.second_target_channel),
    ]
    if args.limit_protocols is not None:
        values.extend(["--limit-protocols", str(args.limit_protocols)])
    if args.training_repetitions is not None:
        values.extend(["--training-repetitions", str(args.training_repetitions)])
    if args.eval_repetitions is not None:
        values.extend(["--eval-repetitions", str(args.eval_repetitions)])
    if args.criterion_score is not None:
        values.extend(["--criterion-score", str(args.criterion_score)])
    if args.stop_at_criterion:
        values.append("--stop-at-criterion")
    if args.measure_relearning:
        values.append("--measure-relearning")
    if args.relearn_only_if_forgot:
        values.append("--relearn-only-if-forgot")
    if args.relearn_repetitions is not None:
        values.extend(["--relearn-repetitions", str(args.relearn_repetitions)])
    if args.resume:
        values.append("--resume")
    values.extend(["--progress-interval", str(args.progress_interval)])
    return values


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("snn_reset_grid_search_%Y%m%dT%H%M%SZ")
    output_root = args.output_root or args.output_dir or RESULTS_DIR / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    task_results = []
    failed = False

    for task in args.tasks:
        task_output_dir = output_root / task
        script = EXPERIMENT_DIR / TASK_SCRIPTS[task]
        command = [sys.executable, str(script), *forwarded_args(args, task_output_dir=task_output_dir)]
        print(json.dumps({"event": "task_start", "task": task, "command": command}), flush=True)
        completed = subprocess.run(command, cwd=EXPERIMENT_DIR.parents[1], check=False)
        task_results.append(
            {
                "task": task,
                "script": str(script),
                "output_dir": str(task_output_dir),
                "returncode": int(completed.returncode),
            }
        )
        if completed.returncode != 0:
            failed = True
            if args.stop_on_failure:
                break

    all_summary = collect_task_summaries(output_root) if not failed else None
    if all_summary is not None and not all_summary.empty:
        all_summary.to_csv(output_root / "all_task_summary.csv", index=False)
    metadata = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": perf_counter() - started,
        "git_commit": git_commit(),
        "tasks": task_results,
        "argv": sys.argv,
        "output_root": str(output_root),
        "status": "failed" if failed else "complete",
        "outputs": [
            "metadata.json",
            *(
                ["all_task_summary.csv"]
                if (output_root / "all_task_summary.csv").exists()
                else []
            ),
        ],
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "suite_failed" if failed else "suite_complete",
                "output_root": str(output_root),
                "tasks": len(task_results),
                "elapsed_s": metadata["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
