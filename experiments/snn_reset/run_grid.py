"""Run the modular SNN reset grid-search suite: one task after another.

This is pure orchestration. Each task's protocol x seed grid is executed by
``common.run_task_grid`` (which parallelises internally); the suite just picks
the tasks, gives each its own output directory, and aggregates the summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from common import RESULTS_DIR, add_common_task_args, collect_task_summaries, git_commit, run_task_grid
from tasks import TASK_BUILDERS

DEFAULT_TASKS = [
    "evoked_channel_response",
    "conditioned_electrode_association",
    "pattern_discrimination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular SNN reset grid-search task suite.")
    add_common_task_args(parser)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_BUILDERS), default=DEFAULT_TASKS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


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
        print(json.dumps({"event": "task_start", "task": task}), flush=True)
        try:
            run_task_grid(args, TASK_BUILDERS[task](args), output_dir=task_output_dir, run_id=task)
            returncode = 0
        except Exception:
            traceback.print_exc()
            returncode = 1
        task_results.append({"task": task, "output_dir": str(task_output_dir), "returncode": returncode})
        if returncode != 0:
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
            *(["all_task_summary.csv"] if (output_root / "all_task_summary.csv").exists() else []),
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
