from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from cl1_snn_reset import (
    CultureConfig,
    ResetProtocol,
    TaskRegime,
    coarse_protocol_grid,
    run_regime_seed_protocols,
    summarize_regime_grid,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
TaskJob = tuple[
    CultureConfig,
    TaskRegime,
    list[ResetProtocol],
    int,
    float,
    float,
    int | None,
    int | None,
    bool,
    bool,
    bool,
    int | None,
]
RESUME_IGNORED_CONFIG_KEYS = {
    "executor",
    "output_dir",
    "progress_interval",
    "resume",
    "run_id",
    "workers",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_DIR.parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def add_common_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--neurons", type=int, default=10_000)
    parser.add_argument("--mean-out-degree", type=int, default=64)
    parser.add_argument("--max-out-degree", type=int, default=96)
    parser.add_argument("--local-candidate-multiplier", type=int, default=6)
    parser.add_argument("--background-noise-mv", type=float, default=1.0)
    parser.add_argument("--spontaneous-rate-hz", type=float, default=0.0)
    parser.add_argument("--homeostasis-rate", type=float, default=0.0)
    parser.add_argument("--homeostasis-interval-ms", type=float, default=100.0)
    parser.add_argument("--backend", choices=["numpy", "brian2"], default="numpy")
    parser.add_argument("--build-workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--consolidation-rest-s", type=float, default=1.0)
    parser.add_argument("--limit-protocols", type=int, default=None)
    parser.add_argument("--training-repetitions", type=int, default=None)
    parser.add_argument("--eval-repetitions", type=int, default=None)
    parser.add_argument("--stop-at-criterion", action="store_true")
    parser.add_argument("--measure-relearning", action="store_true")
    parser.add_argument("--relearn-only-if-forgot", action="store_true")
    parser.add_argument("--relearn-repetitions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--input-current-uA", type=float, default=120.0)
    parser.add_argument("--target-current-uA", type=float, default=120.0)
    parser.add_argument("--criterion-score", type=float, default=None)
    parser.add_argument("--input-channel", type=int, default=8)
    parser.add_argument("--target-channel", type=int, default=17)
    parser.add_argument("--second-input-channel", type=int, default=24)
    parser.add_argument("--second-target-channel", type=int, default=33)
    parser.add_argument("--third-input-channel", type=int, default=40)
    parser.add_argument("--third-target-channel", type=int, default=49)
    parser.add_argument("--fourth-input-channel", type=int, default=48)
    parser.add_argument("--fourth-target-channel", type=int, default=57)


def culture_from_args(args: argparse.Namespace) -> CultureConfig:
    return CultureConfig(
        n_neurons=args.neurons,
        mean_out_degree=args.mean_out_degree,
        max_out_degree=args.max_out_degree,
        local_candidate_multiplier=args.local_candidate_multiplier,
        background_noise_mv=args.background_noise_mv,
        spontaneous_rate_hz=args.spontaneous_rate_hz,
        homeostasis_rate=args.homeostasis_rate,
        homeostasis_interval_ms=args.homeostasis_interval_ms,
        build_workers=args.build_workers,
        backend=args.backend,
    )


def protocols_from_args(args: argparse.Namespace):
    protocols = coarse_protocol_grid()
    if args.limit_protocols is not None:
        protocols = protocols[: args.limit_protocols]
    return protocols


def metadata_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def resume_relevant_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata_config(args).items()
        if key not in RESUME_IGNORED_CONFIG_KEYS
    }


def resume_fingerprint(args: argparse.Namespace, regime: TaskRegime, protocols: list[ResetProtocol]) -> str:
    payload = {
        "config": resume_relevant_config(args),
        "protocol_ids": [protocol.id for protocol in protocols],
        "task_regime": regime.to_metadata(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_resume_metadata(output_dir: Path, expected_fingerprint: str) -> None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Cannot resume {output_dir}: metadata.json is missing.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_fingerprint = metadata.get("resume_fingerprint")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Cannot resume into an output directory with a different task/protocol/config fingerprint."
        )


def task_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_task_args(parser)
    return parser


def task_jobs(
    culture: CultureConfig,
    regime: TaskRegime,
    protocols: list[ResetProtocol],
    args: argparse.Namespace,
    completed: set[tuple[str, int]] | None = None,
) -> list[TaskJob]:
    completed = completed or set()
    jobs = []
    for seed in args.seeds:
        pending_protocols = [
            protocol
            for protocol in protocols
            if (protocol.id, int(seed)) not in completed
        ]
        if not pending_protocols:
            continue
        jobs.append(
            (
                culture,
                regime,
                pending_protocols,
                int(seed),
                float(args.warmup_s),
                float(args.consolidation_rest_s),
                args.training_repetitions,
                args.eval_repetitions,
                bool(args.stop_at_criterion),
                bool(args.measure_relearning),
                bool(args.relearn_only_if_forgot),
                args.relearn_repetitions,
            )
        )
    return jobs


def run_task_job(job: TaskJob) -> list[dict[str, Any]]:
    (
        culture,
        regime,
        protocols,
        seed,
        warmup_s,
        consolidation_rest_s,
        training_repetitions,
        eval_repetitions,
        stop_at_criterion,
        measure_relearning,
        relearn_only_if_forgot,
        relearn_repetitions,
    ) = job
    return run_regime_seed_protocols(
        culture,
        regime,
        protocols,
        seed=int(seed),
        warmup_s=float(warmup_s),
        consolidation_rest_s=float(consolidation_rest_s),
        training_repetitions=training_repetitions,
        eval_repetitions=eval_repetitions,
        stop_at_criterion=bool(stop_at_criterion),
        measure_relearning=bool(measure_relearning),
        relearn_only_if_forgot=bool(relearn_only_if_forgot),
        relearn_repetitions=relearn_repetitions,
    )


def completed_keys(raw_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    if not raw_path.exists():
        return [], set()
    frame = pd.read_csv(raw_path)
    rows = frame.to_dict(orient="records")
    keys = {
        (str(row["protocol_id"]), int(row["seed"]))
        for row in rows
        if "protocol_id" in row and "seed" in row
    }
    return rows, keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    if raw.empty:
        raw.to_csv(output_dir / "raw_trials.csv", index=False)
        summary = pd.DataFrame()
    else:
        raw = raw.sort_values(["protocol_id", "seed"]).reset_index(drop=True)
        raw.to_csv(output_dir / "raw_trials.csv", index=False)
        summary = summarize_regime_grid(raw)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def run_task_grid(
    args: argparse.Namespace,
    regime: TaskRegime,
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Run one task's protocol x seed grid into ``output_dir`` and return its metadata."""
    protocols = protocols_from_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_trials.csv"
    started = perf_counter()
    fingerprint = resume_fingerprint(args, regime, protocols)
    if args.resume and (raw_path.exists() or (output_dir / "metadata.json").exists()):
        validate_resume_metadata(output_dir, fingerprint)
    rows, completed = completed_keys(raw_path) if args.resume else ([], set())
    jobs = task_jobs(culture_from_args(args), regime, protocols, args, completed)
    job_count = len(protocols) * len(args.seeds)
    pending_count = sum(len(job[2]) for job in jobs)
    metadata = {
        "run_id": run_id,
        "task_name": regime.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "argv": sys.argv,
        "config": metadata_config(args),
        "resume_fingerprint": fingerprint,
        "seeds": list(map(int, args.seeds)),
        "protocol_count": len(protocols),
        "job_count": int(job_count),
        "seed_job_count": int(len(jobs)),
        "completed_jobs": int(len(rows)),
        "pending_jobs": int(pending_count),
        "task_regime": regime.to_metadata(),
        "outputs": ["metadata.json", "progress.json", "raw_trials.csv", "summary.csv"],
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "progress.json", metadata)
    if jobs:
        executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        completed_count = len(rows)
        completed_at_start = len(rows)
        last_reported = completed_count
        with executor_cls(max_workers=int(args.workers)) as executor:
            future_to_job = {executor.submit(run_task_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                batch = future.result()
                for row in batch:
                    rows.append(row)
                    append_row(raw_path, row)
                completed_count += len(batch)
                elapsed = perf_counter() - started
                jobs_per_s = (completed_count - completed_at_start) / max(elapsed, 1e-9)
                remaining = max(job_count - completed_count, 0)
                last_row = batch[-1] if batch else {}
                metadata.update(
                    {
                        "completed_jobs": int(completed_count),
                        "pending_jobs": int(remaining),
                        "elapsed_s": float(elapsed),
                        "jobs_per_s": float(jobs_per_s),
                        "eta_s": float(remaining / max(jobs_per_s, 1e-9)),
                        "last_completed_protocol_id": last_row.get("protocol_id"),
                        "last_completed_seed": int(last_row.get("seed", -1)),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_json(output_dir / "progress.json", metadata)
                should_report = completed_count - last_reported >= max(1, args.progress_interval)
                if should_report or completed_count == job_count:
                    last_reported = completed_count
                    print(
                        json.dumps(
                            {
                                "event": "task_progress",
                                "task_name": regime.name,
                                "completed": completed_count,
                                "total": job_count,
                                "jobs_per_s": jobs_per_s,
                                "eta_s": metadata["eta_s"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    summary = write_outputs(output_dir, rows)
    metadata.update(
        {
            "elapsed_s": perf_counter() - started,
            "completed_jobs": int(len(rows)),
            "pending_jobs": 0,
            "summary_rows": int(len(summary)),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "progress.json", metadata)
    print(
        json.dumps(
            {
                "event": "task_complete",
                "task_name": regime.name,
                "output_dir": str(output_dir),
                "rows": int(len(rows)),
                "elapsed_s": metadata["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metadata


def collect_task_summaries(output_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(output_root.glob("*/summary.csv")):
        frame = pd.read_csv(path)
        frame["task_output_dir"] = str(path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
