from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from time import perf_counter

import pandas as pd

from .config import ExperimentConfig
from .experiment import run_trial
from .protocols import ResetProtocol


def _run_one(args):
    cfg, protocol, seed = args
    return run_trial(cfg, protocol, seed).to_row()


def run_sweep(
    cfg: ExperimentConfig,
    protocols: list[ResetProtocol],
    *,
    seeds: list[int] | tuple[int, ...],
    workers: int = 1,
) -> pd.DataFrame:
    """Run protocol x seed grid, optionally across worker threads."""
    jobs = [(replace(cfg, seed=int(seed)), protocol, int(seed)) for protocol in protocols for seed in seeds]
    started = perf_counter()
    if workers <= 1:
        rows = [_run_one(job) for job in jobs]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = perf_counter() - started
    df.attrs["workers"] = int(workers)
    df.attrs["jobs"] = len(jobs)
    return df


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate rows by protocol."""
    metrics = [
        "weight_erasure",
        "residual_performance",
        "savings",
        "trace_auc",
        "criticality_distance",
        "health",
        "energy_cost",
        "path_erasure",
    ]
    return (
        df.groupby("protocol_id", as_index=False)
        .agg({metric: "mean" for metric in metrics} | {"seed": "count"})
        .rename(columns={"seed": "replicates"})
    )
