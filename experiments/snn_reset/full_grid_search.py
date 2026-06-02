from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    TaskConfig,
    coarse_protocol_grid,
    pareto_front,
    rank_protocols,
    run_trial,
    summarize_sweep,
)
from cl1_snn_reset.protocols import ResetProtocol


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _protocol_record(protocol: ResetProtocol) -> dict[str, Any]:
    return {
        "protocol_id": protocol.id,
        "beta": protocol.beta,
        "duration_s": protocol.duration_s,
        "current_uA": protocol.current_uA,
        "pulse_width_us": protocol.pulse_width_us,
        "schedule": protocol.schedule,
        "spatial_mode": protocol.spatial_mode,
        "burst_rate_hz": protocol.burst_rate_hz,
        "epoch_s": protocol.epoch_s,
        "pause_s": protocol.pause_s,
    }


def _run_grid_job(args: tuple[ExperimentConfig, ResetProtocol, int]) -> dict[str, Any]:
    cfg, protocol, seed = args
    started = perf_counter()
    row = run_trial(replace(cfg, seed=int(seed)), protocol, int(seed)).to_row()
    row["job_elapsed_s"] = perf_counter() - started
    return row


def _make_config(args: argparse.Namespace) -> ExperimentConfig:
    build_workers = args.build_workers
    if build_workers is None:
        build_workers = 1 if args.workers > 1 else -1
    return ExperimentConfig(
        culture=CultureConfig(
            n_neurons=args.neurons,
            mean_out_degree=args.mean_out_degree,
            max_out_degree=args.max_out_degree,
            local_candidate_multiplier=args.local_candidate_multiplier,
            build_workers=build_workers,
            spontaneous_rate_hz=args.spontaneous_rate_hz,
            backend=args.backend,
        ),
        task=TaskConfig(
            input_channels=tuple(args.input_channels),
            target_channels=tuple(args.target_channels),
            max_trials=args.max_trials,
            eval_interval_trials=args.eval_interval_trials,
            eval_trials=args.eval_trials,
            inter_trial_ms=args.inter_trial_ms,
            criterion_response_probability=args.criterion_response_probability,
        ),
        readout_window_s=args.readout_window_s,
        warmup_s=args.warmup_s,
        seed=args.seeds[0],
        keep_snapshots=False,
    )


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    completed = {
        (str(row["protocol_id"]), int(row["seed"]))
        for row in rows
        if "protocol_id" in row and "seed" in row
    }
    return rows, completed


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].head(limit).copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.4g}")
    view = view.fillna("")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy().tolist()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), divider, *(format_row(row) for row in rows)])


def _group_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    metrics = [
        "reset_score",
        "weight_erasure",
        "path_erasure",
        "residual_performance",
        "savings",
        "trace_auc",
        "health",
        "energy_cost",
    ]
    return (
        df.groupby(by, as_index=False)
        .agg({metric: "mean" for metric in metrics} | {"protocol_id": "count"})
        .rename(columns={"protocol_id": "protocols"})
        .sort_values("reset_score", ascending=False)
        .reset_index(drop=True)
    )


def _write_report(
    *,
    output_dir: Path,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    ranked: pd.DataFrame,
    pareto: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    report_path = output_dir / "report.md"
    initial_zero = int((raw["initial_trials"] <= 0).sum()) if "initial_trials" in raw else 0
    relearn_zero = int((raw["relearn_trials"] <= 0).sum()) if "relearn_trials" in raw else 0
    reached_training_floor = int((raw["initial_trials"] >= metadata["config"]["max_trials"]).sum()) if "initial_trials" in raw else 0
    top_columns = [
        "protocol_id",
        "reset_score",
        "weight_erasure",
        "path_erasure",
        "residual_performance",
        "savings",
        "trace_auc",
        "health",
        "energy_cost",
        "replicates",
    ]
    protocol_columns = [
        "beta",
        "schedule",
        "spatial_mode",
        "duration_s",
        "current_uA",
        "reset_score",
        "weight_erasure",
        "residual_performance",
        "savings",
        "trace_auc",
        "health",
    ]
    lines = [
        "# Full SNN Reset Grid Search",
        "",
        "## Run Metadata",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Git commit: `{metadata['git_commit']}`",
        f"- Neurons: `{metadata['config']['neurons']}`",
        f"- Mean out-degree: `{metadata['config']['mean_out_degree']}`",
        f"- Protocols: `{metadata['protocol_count']}`",
        f"- Seeds: `{metadata['seeds']}`",
        f"- Jobs completed: `{metadata['completed_jobs']} / {metadata['job_count']}`",
        f"- Workers: `{metadata['workers']}` via `{metadata['executor']}` executor",
        f"- Elapsed seconds: `{metadata['elapsed_s']:.2f}`",
        f"- Jobs per second: `{metadata['jobs_per_s']:.4g}`",
        "",
        "## Data Quality",
        "",
        f"- Initial-at-criterion rows: `{initial_zero}`",
        f"- Relearn-at-criterion rows: `{relearn_zero}`",
        f"- Initial max-trial rows: `{reached_training_floor}`",
        "",
        "## Objective",
        "",
        f"This run screens the full coarse reset protocol grid on a {metadata['config']['neurons']}-neuron",
        "CL1-style SNN.  Protocols are compared on true simulation-only weight",
        "erasure and CL1-like readouts: behavior after reset, relearning savings,",
        "trace detectability, health, criticality distance, and stimulation cost.",
        "",
        "## Top Ranked Protocols",
        "",
        _markdown_table(ranked, top_columns, 20),
        "",
        "## Pareto Front",
        "",
        _markdown_table(pareto, top_columns, 30),
        "",
        "## Best Settings By Factor",
        "",
        "### By Beta",
        "",
        _markdown_table(_group_table(ranked, "beta"), ["beta", "protocols"] + top_columns[1:9], 10),
        "",
        "### By Schedule",
        "",
        _markdown_table(_group_table(ranked, "schedule"), ["schedule", "protocols"] + top_columns[1:9], 10),
        "",
        "### By Spatial Mode",
        "",
        _markdown_table(_group_table(ranked, "spatial_mode"), ["spatial_mode", "protocols"] + top_columns[1:9], 10),
        "",
        "## Protocol Parameter View",
        "",
        _markdown_table(ranked, protocol_columns, 30),
        "",
        "## Artifacts",
        "",
        "- `raw_trials.csv`: every protocol x seed trial.",
        "- `summary.csv`: replicate means by protocol.",
        "- `ranked.csv`: summary with scalar screen score.",
        "- `pareto.csv`: nondominated protocol rows.",
        "- `metadata.json`: run configuration and progress metadata.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _finalize_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    raw = pd.DataFrame(rows)
    raw = raw.sort_values(["protocol_id", "seed"]).reset_index(drop=True)
    raw.to_csv(output_dir / "raw_trials.csv", index=False)
    summary = summarize_sweep(raw)
    ranked = rank_protocols(summary)
    pareto = pareto_front(ranked)
    summary.to_csv(output_dir / "summary.csv", index=False)
    ranked.to_csv(output_dir / "ranked.csv", index=False)
    pareto.to_csv(output_dir / "pareto.csv", index=False)
    _write_report(
        output_dir=output_dir,
        raw=raw,
        summary=summary,
        ranked=ranked,
        pareto=pareto,
        metadata=metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SNN reset protocol grid.")
    parser.add_argument("--neurons", type=int, default=10_000)
    parser.add_argument("--mean-out-degree", type=int, default=64)
    parser.add_argument("--max-out-degree", type=int, default=96)
    parser.add_argument("--local-candidate-multiplier", type=int, default=6)
    parser.add_argument("--spontaneous-rate-hz", type=float, default=0.12)
    parser.add_argument("--backend", choices=["numpy", "brian2"], default="numpy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--build-workers", type=int, default=None)
    parser.add_argument("--input-channels", type=int, nargs="+", default=[8])
    parser.add_argument("--target-channels", type=int, nargs="+", default=[9])
    parser.add_argument("--max-trials", type=int, default=120)
    parser.add_argument("--eval-interval-trials", type=int, default=8)
    parser.add_argument("--eval-trials", type=int, default=8)
    parser.add_argument("--inter-trial-ms", type=float, default=70.0)
    parser.add_argument("--criterion-response-probability", type=float, default=0.875)
    parser.add_argument("--readout-window-s", type=float, default=1.5)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-protocols", type=int, default=None)
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("full_grid_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("experiments/snn_reset/results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    protocols = coarse_protocol_grid()
    if args.limit_protocols is not None:
        protocols = protocols[: args.limit_protocols]
    cfg = _make_config(args)
    jobs = [(cfg, protocol, int(seed)) for protocol in protocols for seed in args.seeds]
    if args.limit_jobs is not None:
        jobs = jobs[: args.limit_jobs]

    raw_path = output_dir / "raw_trials.csv"
    rows, completed = _load_completed(raw_path) if args.resume else ([], set())
    pending = [
        job
        for job in jobs
        if (job[1].id, int(job[2])) not in completed
    ]

    metadata = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "neurons": args.neurons,
            "mean_out_degree": args.mean_out_degree,
            "max_out_degree": args.max_out_degree,
            "local_candidate_multiplier": args.local_candidate_multiplier,
            "spontaneous_rate_hz": args.spontaneous_rate_hz,
            "backend": args.backend,
            "input_channels": args.input_channels,
            "target_channels": args.target_channels,
            "max_trials": args.max_trials,
            "eval_interval_trials": args.eval_interval_trials,
            "eval_trials": args.eval_trials,
            "criterion_response_probability": args.criterion_response_probability,
            "readout_window_s": args.readout_window_s,
            "warmup_s": args.warmup_s,
        },
        "seeds": list(map(int, args.seeds)),
        "protocol_count": len(protocols),
        "job_count": len(jobs),
        "completed_jobs": len(rows),
        "pending_jobs": len(pending),
        "workers": int(args.workers),
        "executor": args.executor,
        "protocol_grid": [_protocol_record(protocol) for protocol in protocols],
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "metadata.json", metadata)

    print(
        json.dumps(
            {
                "event": "start",
                "run_id": run_id,
                "output_dir": str(output_dir),
                "protocols": len(protocols),
                "seeds": args.seeds,
                "jobs": len(jobs),
                "pending": len(pending),
                "workers": args.workers,
                "executor": args.executor,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        metadata["elapsed_s"] = 0.0
        metadata["jobs_per_s"] = 0.0
        _finalize_outputs(output_dir, rows, metadata)
        _write_json(output_dir / "metadata.json", metadata)
        print(json.dumps({"event": "complete", "output_dir": str(output_dir)}, sort_keys=True), flush=True)
        return

    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    started = perf_counter()
    total = len(jobs)
    completed_count = len(rows)
    with executor_cls(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(_run_grid_job, job): job for job in pending}
        for future in as_completed(future_to_job):
            row = future.result()
            rows.append(row)
            _append_row(raw_path, row)
            completed_count += 1
            elapsed = perf_counter() - started
            remaining = max(total - completed_count, 0)
            jobs_per_s = (completed_count - len(completed)) / max(elapsed, 1e-9)
            eta_s = remaining / max(jobs_per_s, 1e-9)
            metadata.update(
                {
                    "completed_jobs": completed_count,
                    "pending_jobs": remaining,
                    "elapsed_s": elapsed,
                    "jobs_per_s": jobs_per_s,
                    "eta_s": eta_s,
                    "last_completed_protocol_id": row.get("protocol_id"),
                    "last_completed_seed": int(row.get("seed", -1)),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(output_dir / "progress.json", metadata)
            if completed_count % max(1, args.progress_interval) == 0 or completed_count == total:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed": completed_count,
                            "total": total,
                            "elapsed_s": round(elapsed, 2),
                            "jobs_per_s": round(jobs_per_s, 5),
                            "eta_s": round(eta_s, 2),
                            "last_protocol": row.get("protocol_id"),
                            "last_seed": int(row.get("seed", -1)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    elapsed = perf_counter() - started
    metadata.update(
        {
            "completed_jobs": len(rows),
            "pending_jobs": 0,
            "elapsed_s": elapsed,
            "jobs_per_s": len(pending) / max(elapsed, 1e-9),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _finalize_outputs(output_dir, rows, metadata)
    _write_json(output_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(output_dir),
                "elapsed_s": round(elapsed, 2),
                "jobs": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
