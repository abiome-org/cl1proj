from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "cl1-snn-reset-final-manifold"
VOLUME_NAME = "cl1-snn-reset-final-results"
REMOTE_PROJECT = Path("/root/cl1proj")
REMOTE_RESULTS = Path("/results")
DEFAULT_TASKS = (
    "conditioned_electrode_association",
    "pattern_discrimination",
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.function(
    image=image,
    volumes={str(REMOTE_RESULTS): results_volume},
    cpu=(32.0, 32.0),
    memory=65536,
    timeout=60 * 60 * 12,
)
def run_manifold_task(
    task: str,
    output_root_name: str,
    seeds: list[int],
    workers: int,
    alpha_count: int = 16,
    training_repetitions: int = 80,
    eval_repetitions: int = 8,
    input_current_uA: float = 120.0,
    target_current_uA: float = 120.0,
    progress_interval: int = 10,
) -> dict[str, Any]:
    output_dir = REMOTE_RESULTS / output_root_name / task
    command = [
        sys.executable,
        "-u",
        "experiments/snn_reset/manifold_analysis.py",
        "--out",
        str(output_dir),
        "--tasks",
        task,
        "--neurons",
        "10000",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--workers",
        str(int(workers)),
        "--executor",
        "process",
        "--training-repetitions",
        str(int(training_repetitions)),
        "--eval-repetitions",
        str(int(eval_repetitions)),
        "--input-current-uA",
        str(float(input_current_uA)),
        "--target-current-uA",
        str(float(target_current_uA)),
        "--progress-interval",
        str(int(progress_interval)),
        "--alpha-count",
        str(int(alpha_count)),
        "--resume",
    ]
    try:
        subprocess.run(command, cwd=REMOTE_PROJECT, check=True)
    finally:
        results_volume.commit()

    progress = _read_json(output_dir / "progress.json")
    metadata = _read_json(output_dir / "metadata.json")
    return {
        "task": task,
        "output_dir": str(output_dir),
        "completed_jobs": progress.get("completed_jobs", metadata.get("completed_jobs")),
        "elapsed_s": progress.get("elapsed_s", metadata.get("elapsed_s")),
        "status": metadata.get("status", progress.get("status")),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.local_entrypoint()
def main(
    tasks: str = ",".join(DEFAULT_TASKS),
    output_root_name: str = "",
    workers: int = 32,
    seed_start: int = 1,
    seed_stop: int = 50,
    alpha_count: int = 16,
) -> None:
    selected_tasks = [task.strip() for task in tasks.replace(" ", ",").split(",") if task.strip()]
    if not selected_tasks:
        raise ValueError("No tasks selected.")
    output_root_name = output_root_name or f"manifold_analysis_final_n50_modal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    seeds = list(range(int(seed_start), int(seed_stop) + 1))
    calls = [
        run_manifold_task.spawn(
            task,
            output_root_name,
            seeds,
            int(workers),
            int(alpha_count),
        )
        for task in selected_tasks
    ]
    print(
        json.dumps(
            {
                "event": "submitted",
                "output_root_name": output_root_name,
                "tasks": selected_tasks,
                "alpha_count": int(alpha_count),
            }
        )
    )
    for call in calls:
        print(json.dumps(call.get(), sort_keys=True))
    print(json.dumps({"event": "complete", "output_root_name": output_root_name, "tasks": selected_tasks}))
