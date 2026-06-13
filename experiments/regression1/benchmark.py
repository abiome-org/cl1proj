from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import pandas as pd

from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    TaskConfig,
    build_network,
    protocol_events,
    rank_protocols,
    run_sweep,
    summarize_sweep,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def network_benchmark(neurons: int, sim_seconds: float, seed: int, degree: int) -> dict:
    cfg = CultureConfig(
        n_neurons=neurons,
        mean_out_degree=degree,
        max_out_degree=max(degree, min(128, degree * 2)),
        local_candidate_multiplier=4,
        spontaneous_rate_hz=0.02,
        backend="numpy",
    )
    t0 = perf_counter()
    net = build_network(cfg, seed=seed)
    build_s = perf_counter() - t0
    protocol = ResetProtocol(
        beta=0,
        duration_s=sim_seconds,
        current_uA=1.8,
        pulse_width_us=160,
        schedule="static",
        spatial_mode="independent",
        burst_rate_hz=45,
    )
    events = protocol_events(protocol, n_channels=64, rng=net.rng)
    t1 = perf_counter()
    activity = net.advance(sim_seconds * 1000.0, events, plasticity=True, record=True)
    run_s = perf_counter() - t1
    return {
        "neurons": neurons,
        "synapses": net.synapse_count,
        "build_s": build_s,
        "run_s": run_s,
        "simulated_s": sim_seconds,
        "x_realtime": sim_seconds / max(run_s, 1e-9),
        "spikes": int(activity.total_neuron_spikes),
        "pulses": int(sum(len(event.channels) for event in events)),
    }


def sweep_benchmark(neurons: int, workers: int, seeds: int, degree: int) -> tuple[pd.DataFrame, dict]:
    cfg = ExperimentConfig(
        culture=CultureConfig(
            n_neurons=neurons,
            mean_out_degree=degree,
            max_out_degree=max(degree, min(96, degree * 2)),
            local_candidate_multiplier=4,
            build_workers=1 if workers > 1 else -1,
            spontaneous_rate_hz=0.03,
            backend="numpy",
        ),
        task=TaskConfig(
            input_channels=(8,),
            target_channels=(55,),
            max_trials=16,
            eval_interval_trials=8,
            eval_trials=4,
            inter_trial_ms=45.0,
            criterion_response_probability=0.5,
        ),
        readout_window_s=0.2,
        warmup_s=0.05,
    )
    protocols = [
        ResetProtocol(-1, 0.25, 1.0, 160, "static", "independent", burst_rate_hz=40),
        ResetProtocol(0, 0.25, 1.4, 160, "static", "correlated", burst_rate_hz=45),
        ResetProtocol(1, 0.25, 1.4, 160, "epoch_pause", "shared", burst_rate_hz=35, epoch_s=0.12, pause_s=0.04),
        ResetProtocol(2, 0.25, 2.0, 160, "alternating_blue_red", "phase_shifted", burst_rate_hz=35),
    ]
    seed_values = tuple(range(1, seeds + 1))
    df = run_sweep(cfg, protocols, seeds=seed_values, workers=workers)
    metadata = {
        "workers": workers,
        "jobs": int(df.attrs.get("jobs", len(df))),
        "elapsed_s": float(df.attrs.get("elapsed_s", 0.0)),
        "jobs_per_s": float(df.attrs.get("jobs", len(df)) / max(df.attrs.get("elapsed_s", 0.0), 1e-9)),
    }
    return df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression benchmark for cl1_snn_reset.")
    parser.add_argument("--network-neurons", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--network-sim-seconds", type=float, default=1.0)
    parser.add_argument("--sweep-neurons", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "benchmark_sweep.csv",
        help="Sweep CSV path (default: experiments/regression/results/benchmark_sweep.csv)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    network_rows = [
        network_benchmark(neurons, args.network_sim_seconds, seed=10 + index, degree=args.degree)
        for index, neurons in enumerate(args.network_neurons)
    ]
    single_df, single_meta = sweep_benchmark(args.sweep_neurons, workers=1, seeds=args.seeds, degree=args.degree)
    multi_df, multi_meta = sweep_benchmark(args.sweep_neurons, workers=args.workers, seeds=args.seeds, degree=args.degree)
    multi_df.to_csv(args.output, index=False)
    ranked = rank_protocols(summarize_sweep(multi_df)).head(5)

    payload = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "network": network_rows,
        "sweep_single_worker": single_meta,
        "sweep_multi_worker": multi_meta,
        "parallel_speedup": single_meta["elapsed_s"] / max(multi_meta["elapsed_s"], 1e-9),
        "top_protocols": ranked.to_dict(orient="records"),
        "output_csv": str(args.output),
    }
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
