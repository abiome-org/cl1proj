"""Suite result IO: validate a completed grid output and load it into one frame
with the standard derived analysis columns. Shared by the experiment figure
scripts and the relearning analysis so the loading/validation rules live in one
place. Presentation (labels, palette, output paths) stays in the experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def energy_cost_uC(frame: pd.DataFrame) -> pd.Series:
    """Nominal injected charge per evaluation (uC), summed over both pulse phases."""
    return (
        frame["current_uA"].abs()
        * frame["pulse_width_us"].astype(float)
        * 1e-6
        * frame["total_pulses"].astype(float)
        * 2.0
    )


def validate_suite_dir(suite_dir: Path) -> None:
    """Raise if a suite directory is missing metadata or has incomplete tasks."""
    metadata_path = suite_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Suite metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_results = metadata.get("tasks", [])
    if metadata.get("status", "complete") == "failed":
        raise ValueError(f"Suite is marked failed: {suite_dir}")
    failed_tasks = [task for task in task_results if int(task.get("returncode", 1)) != 0]
    if failed_tasks:
        raise ValueError(f"Suite contains failed task scripts: {failed_tasks}")
    for task in task_results:
        task_output_dir = Path(task["output_dir"])
        if not task_output_dir.is_absolute() and not task_output_dir.exists():
            task_output_dir = suite_dir / str(task["task"])
        progress_path = task_output_dir / "progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"Task progress metadata is missing: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_jobs", -1))
        expected = int(progress.get("job_count", -2))
        pending = int(progress.get("pending_jobs", -1))
        if completed != expected or pending != 0:
            raise ValueError(f"Task output is incomplete: {progress_path}")


def suite_neuron_count(suite_dir: Path) -> int:
    """Read the neuron count from per-task metadata, falling back to suite argv."""
    for metadata_path in sorted(suite_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        if "neurons" in config:
            return int(config["neurons"])
    metadata = json.loads((suite_dir / "metadata.json").read_text(encoding="utf-8"))
    argv = list(metadata.get("argv", []))
    if "--neurons" in argv:
        index = argv.index("--neurons")
        if index + 1 < len(argv):
            return int(argv[index + 1])
    return 10_000


def load_suite(suite_dir: Path) -> pd.DataFrame:
    """Validate a suite, concatenate its per-task ``raw_trials.csv`` files, and
    attach the standard derived analysis columns. Display labels are not added
    here; presentation code maps ``task_name``/``schedule`` to labels itself.
    """
    validate_suite_dir(suite_dir)
    frames = []
    for raw_path in sorted(suite_dir.glob("*/raw_trials.csv")):
        frame = pd.read_csv(raw_path).copy()
        task_output_dir = pd.Series([str(raw_path.parent)] * len(frame), name="task_output_dir")
        frames.append(pd.concat([frame, task_output_dir], axis=1))
    if not frames:
        raise FileNotFoundError(f"No task raw_trials.csv files found under {suite_dir}")
    raw = pd.concat(frames, ignore_index=True).copy()
    n_neurons = suite_neuron_count(suite_dir)
    raw["score_dent"] = raw["no_reset_score"] - raw["reset_score"]
    raw["criterion_margin"] = raw["reset_score"] - raw["criterion_score"]
    raw["no_reset_margin"] = raw["no_reset_score"] - raw["criterion_score"]
    raw["margin_consumed"] = np.where(
        raw["no_reset_margin"] > 1e-12,
        raw["score_dent"] / raw["no_reset_margin"],
        np.nan,
    )
    raw["abs_weight_effect"] = raw["reset_minus_no_reset_weight_norm"].abs()
    raw["log_weight_effect"] = np.log10(1.0 + raw["abs_weight_effect"])
    raw["extra_spikes_per_s"] = raw["reset_window_neuron_spikes_delta"] / raw["duration_s"].clip(lower=1e-9)
    raw["extra_spikes_per_neuron_s"] = raw["extra_spikes_per_s"] / max(float(n_neurons), 1.0)
    raw["energy_cost_uC"] = energy_cost_uC(raw)
    raw["erasure_parallel_norm"] = (raw["erasure_projection_reset_vs_no_reset"] * raw["trained_delta_norm"]).abs()
    parallel_sq = np.square(raw["erasure_parallel_norm"].fillna(0.0))
    total_sq = np.square(raw["reset_minus_no_reset_weight_norm"].fillna(0.0))
    raw["erasure_orthogonal_norm"] = np.sqrt(np.maximum(total_sq - parallel_sq, 0.0))
    raw["relearn_trials"] = raw["relearn_trials"].fillna(0.0)
    raw["relearn_savings"] = raw["relearn_savings"].fillna(1.0)
    raw["criterion_forget"] = raw["criterion_forget"].astype(bool)
    raw["score_drop"] = raw["score_drop"].astype(bool)
    raw["n_neurons"] = n_neurons
    return raw
