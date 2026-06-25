from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "cl1-snn-reset-final-grid-chunks"
VOLUME_NAME = "cl1-snn-reset-final-results"
REMOTE_PROJECT = Path("/root/cl1proj")
REMOTE_RESULTS = Path("/results")
DEFAULT_TASKS = (
    "overlapping_shared_target_association",
    "overlapping_shared_input_association",
    "multi_association_mapping",
    "xor_electrode_classification",
)


def _ignore_experiment_outputs(path: Path) -> bool:
    return "results" in path.parts or "__pycache__" in path.parts


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_file("pyproject.toml", str(REMOTE_PROJECT / "pyproject.toml"), copy=True)
    .add_local_dir("src", str(REMOTE_PROJECT / "src"), copy=True)
    .add_local_dir(
        "experiments/snn_reset",
        str(REMOTE_PROJECT / "experiments" / "snn_reset"),
        copy=True,
        ignore=_ignore_experiment_outputs,
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_PROJECT / 'src'}:{REMOTE_PROJECT / 'experiments' / 'snn_reset'}",
            "MPLCONFIGDIR": "/tmp/mpl-cache",
            "XDG_CACHE_HOME": "/tmp/xdg-cache",
        }
    )
    .workdir(str(REMOTE_PROJECT))
)

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _namespace(
    *,
    task: str,
    output_root_name: str,
    seed: int,
    training_repetitions: int,
    eval_repetitions: int,
    relearn_repetitions: int,
    input_current_uA: float,
    target_current_uA: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        backend="numpy",
        background_noise_mv=1.0,
        build_workers=1,
        consolidation_rest_s=1.0,
        criterion_score=None,
        eval_repetitions=eval_repetitions,
        executor="thread",
        fourth_input_channel=48,
        fourth_target_channel=57,
        homeostasis_interval_ms=100.0,
        homeostasis_rate=0.0,
        input_channel=8,
        input_current_uA=input_current_uA,
        limit_protocols=None,
        local_candidate_multiplier=6,
        max_out_degree=96,
        mean_out_degree=64,
        measure_relearning=True,
        neurons=10_000,
        output_dir=None,
        output_root=REMOTE_RESULTS / output_root_name,
        progress_interval=60,
        relearn_only_if_forgot=False,
        relearn_repetitions=relearn_repetitions,
        resume=True,
        run_id=f"{task}_seed{int(seed):03d}",
        second_input_channel=24,
        second_target_channel=33,
        seeds=[int(seed)],
        spontaneous_rate_hz=0.0,
        stop_at_criterion=False,
        target_channel=17,
        target_current_uA=target_current_uA,
        tasks=[task],
        third_input_channel=40,
        third_target_channel=49,
        training_repetitions=training_repetitions,
        warmup_s=0.5,
        workers=1,
    )


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS): results_volume},
    cpu=(2.0, 2.0),
    memory=8192,
    timeout=60 * 60 * 12,
    max_containers=240,
)
def run_grid_seed(
    task: str,
    seed: int,
    output_root_name: str,
    training_repetitions: int = 80,
    eval_repetitions: int = 8,
    relearn_repetitions: int = 80,
    input_current_uA: float = 120.0,
    target_current_uA: float = 120.0,
) -> dict[str, Any]:
    sys.path.insert(0, str(REMOTE_PROJECT / "experiments" / "snn_reset"))
    sys.path.insert(0, str(REMOTE_PROJECT / "src"))

    from common import run_task_grid
    from tasks import TASK_BUILDERS

    args = _namespace(
        task=task,
        output_root_name=output_root_name,
        seed=int(seed),
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        relearn_repetitions=relearn_repetitions,
        input_current_uA=input_current_uA,
        target_current_uA=target_current_uA,
    )
    output_dir = REMOTE_RESULTS / output_root_name / task / f"seed_{int(seed):03d}"
    try:
        metadata = run_task_grid(args, TASK_BUILDERS[task](args), output_dir=output_dir, run_id=args.run_id)
    finally:
        results_volume.commit()
    return {
        "task": task,
        "seed": int(seed),
        "output_dir": str(output_dir),
        "rows": metadata.get("completed_jobs"),
        "elapsed_s": metadata.get("elapsed_s"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.local_entrypoint()
def main(
    tasks: str = ",".join(DEFAULT_TASKS),
    output_root_name: str = "",
    seed_start: int = 1,
    seed_stop: int = 50,
) -> None:
    selected_tasks = [task.strip() for task in tasks.replace(" ", ",").split(",") if task.strip()]
    if not selected_tasks:
        raise ValueError("No tasks selected.")
    output_root_name = output_root_name or f"snn_reset_final_n50_modal_chunks_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    seeds = list(range(int(seed_start), int(seed_stop) + 1))
    calls = [
        run_grid_seed.spawn(task, seed, output_root_name)
        for task in selected_tasks
        for seed in seeds
    ]
    print(
        json.dumps(
            {
                "event": "submitted",
                "output_root_name": output_root_name,
                "tasks": selected_tasks,
                "seeds": seeds,
                "calls": len(calls),
            }
        ),
        flush=True,
    )
    for call in calls:
        print(json.dumps(call.get(), sort_keys=True), flush=True)
    print(json.dumps({"event": "complete", "output_root_name": output_root_name, "tasks": selected_tasks}))
