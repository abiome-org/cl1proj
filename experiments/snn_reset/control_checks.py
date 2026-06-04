from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    TaskConfig,
    apply_reset_protocol,
    build_network,
    build_trial_artifacts,
    capture_phase,
    colored_noise,
    compute_trial_metrics,
    evaluate_task,
    protocol_events,
    train_to_criterion,
    trace_auc_proxy,
)


def selected_protocols() -> list[ResetProtocol]:
    """Representative protocols spanning the previous 10k grid outcomes."""
    return [
        ResetProtocol(
            beta=2,
            duration_s=0.75,
            current_uA=0.8,
            pulse_width_us=160,
            schedule="epoch_pause",
            spatial_mode="independent",
            protocol_id="low_burden_0.75s",
        ),
        ResetProtocol(
            beta=0,
            duration_s=1.5,
            current_uA=0.8,
            pulse_width_us=160,
            schedule="static",
            spatial_mode="shared",
            protocol_id="mid_burden_1.5s",
        ),
        ResetProtocol(
            beta=-2,
            duration_s=3.0,
            current_uA=2.6,
            pulse_width_us=160,
            schedule="static",
            spatial_mode="shared",
            protocol_id="high_burden_3s",
        ),
    ]


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        culture=CultureConfig(
            n_neurons=args.neurons,
            mean_out_degree=args.mean_out_degree,
            max_out_degree=args.max_out_degree,
            local_candidate_multiplier=args.local_candidate_multiplier,
            build_workers=args.build_workers,
            spontaneous_rate_hz=args.spontaneous_rate_hz,
            backend="numpy",
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
        keep_snapshots=False,
    )


def estimate_power_slope(signal: np.ndarray) -> float:
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(signal.size)
    power = np.abs(spectrum) ** 2
    valid = (freqs > 0.0) & np.isfinite(power) & (power > 0.0)
    if np.count_nonzero(valid) < 8:
        return 0.0
    lo = max(1, int(np.floor(np.count_nonzero(valid) * 0.02)))
    hi = max(lo + 8, int(np.floor(np.count_nonzero(valid) * 0.85)))
    x = np.log(freqs[valid][lo:hi])
    y = np.log(power[valid][lo:hi])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def noise_diagnostics(*, seed: int, repeats: int, n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for beta in [-2, -1, 0, 1, 2]:
        slopes = [
            estimate_power_slope(colored_noise(beta, n_samples, rng))
            for _ in range(repeats)
        ]
        for spatial_mode in ["shared", "independent", "correlated", "phase_shifted"]:
            protocol = ResetProtocol(
                beta=beta,
                duration_s=3.0,
                current_uA=1.0,
                pulse_width_us=160,
                schedule="static",
                spatial_mode=spatial_mode,
            )
            events = protocol_events(protocol, n_channels=64, rng=rng)
            channel_counts = np.bincount(
                [channel for event in events for channel in event.channels],
                minlength=64,
            )
            rows.append(
                {
                    "beta": beta,
                    "target_power_slope": -float(beta),
                    "estimated_power_slope_mean": float(np.mean(slopes)),
                    "estimated_power_slope_std": float(np.std(slopes)),
                    "spatial_mode": spatial_mode,
                    "event_count": len(events),
                    "total_pulses": int(channel_counts.sum()),
                    "active_channels": int(np.count_nonzero(channel_counts)),
                    "mean_pulses_per_active_channel": float(
                        channel_counts[channel_counts > 0].mean()
                        if np.any(channel_counts > 0)
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_trained_control(args: tuple[ExperimentConfig, ResetProtocol, int, str]) -> dict[str, Any]:
    cfg, protocol, seed, mode = args
    net = build_network(replace(cfg.culture, build_workers=1), seed=seed)
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)

    baseline = capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=False,
    )
    initial = train_to_criterion(net, cfg.task)
    trained = capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=False,
    )

    if mode == "reset":
        reset_activity, total_pulses = apply_reset_protocol(net, protocol, seed=seed + 10_000)
        trial_protocol = protocol
    elif mode == "no_reset":
        reset_activity = net.advance(protocol.duration_s * 1000.0, [], plasticity=True, record=True)
        total_pulses = 0
        trial_protocol = replace(protocol, protocol_id=f"no_reset_{protocol.duration_s:g}s")
    else:
        raise ValueError(f"Unknown trained control mode: {mode}")

    post = capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=False,
    )
    post_behavior = evaluate_task(net, cfg.task, trials=cfg.task.eval_trials)
    relearn = train_to_criterion(net, cfg.task)
    artifacts = build_trial_artifacts(
        baseline=baseline,
        trained=trained,
        post=post,
        initial=initial,
        relearn=relearn,
        post_behavior=post_behavior,
        protocol=trial_protocol,
        seed=seed,
        total_pulses=total_pulses,
    )
    row = compute_trial_metrics(artifacts).to_row()
    row.update(
        {
            "control_mode": mode,
            "source_protocol_id": protocol.id,
            "trained_delta_norm": float(np.linalg.norm(trained.weights - baseline.weights)),
            "post_delta_norm": float(np.linalg.norm(post.weights - baseline.weights)),
            "post_minus_trained_norm": float(np.linalg.norm(post.weights - trained.weights)),
            "baseline_weight_norm": float(np.linalg.norm(baseline.weights)),
            "reset_window_neuron_spikes": int(reset_activity.total_neuron_spikes),
        }
    )
    return row


def run_untrained_control(args: tuple[ExperimentConfig, ResetProtocol, int, str]) -> dict[str, Any]:
    cfg, protocol, seed, mode = args
    net = build_network(replace(cfg.culture, build_workers=1), seed=seed)
    if cfg.warmup_s > 0:
        net.advance(cfg.warmup_s * 1000.0, [], plasticity=False, record=False)

    baseline = capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=False,
    )
    pre_behavior = evaluate_task(net, cfg.task, trials=cfg.task.eval_trials)
    if mode == "reset":
        reset_activity, total_pulses = apply_reset_protocol(net, protocol, seed=seed + 10_000)
    elif mode == "no_reset":
        reset_activity = net.advance(protocol.duration_s * 1000.0, [], plasticity=True, record=True)
        total_pulses = 0
    else:
        raise ValueError(f"Unknown untrained control mode: {mode}")

    post = capture_phase(
        net,
        cfg.task,
        readout_window_s=cfg.readout_window_s,
        keep_snapshots=False,
    )
    post_behavior = evaluate_task(net, cfg.task, trials=cfg.task.eval_trials)
    drift_norm = float(np.linalg.norm(post.weights - baseline.weights))
    return {
        "control_mode": mode,
        "source_protocol_id": protocol.id,
        "seed": int(seed),
        "beta": float(protocol.beta),
        "schedule": protocol.schedule,
        "spatial_mode": protocol.spatial_mode,
        "duration_s": float(protocol.duration_s),
        "current_uA": float(protocol.current_uA),
        "pulse_width_us": int(protocol.pulse_width_us),
        "total_pulses": int(total_pulses),
        "pre_behavior": float(pre_behavior),
        "post_behavior": float(post_behavior),
        "weight_drift_norm": drift_norm,
        "weight_drift_rel_to_baseline": drift_norm / (float(np.linalg.norm(baseline.weights)) + 1e-9),
        "path0": float(baseline.path_strength),
        "path_post": float(post.path_strength),
        "path_delta": float(post.path_strength - baseline.path_strength),
        "trace_auc": float(trace_auc_proxy(baseline.activity, post.activity)),
        "reset_window_neuron_spikes": int(reset_activity.total_neuron_spikes),
    }


def run_parallel(jobs, worker, workers: int) -> pd.DataFrame:
    if workers <= 1:
        return pd.DataFrame([worker(job) for job in jobs])
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def summarize_reset_vs_no_reset(trained: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "weight_erasure",
        "residual_performance",
        "savings",
        "trace_auc",
        "health",
        "post_delta_norm",
        "reset_window_neuron_spikes",
    ]
    reset = trained[trained["control_mode"] == "reset"]
    no_reset = trained[trained["control_mode"] == "no_reset"]
    merged = reset.merge(
        no_reset,
        on=["source_protocol_id", "seed"],
        suffixes=("_reset", "_no_reset"),
    )
    rows = []
    for _, row in merged.iterrows():
        result: dict[str, Any] = {
            "source_protocol_id": row["source_protocol_id"],
            "seed": int(row["seed"]),
            "duration_s": float(row["duration_s_reset"]),
        }
        for metric in metrics:
            result[f"{metric}_reset"] = row[f"{metric}_reset"]
            result[f"{metric}_no_reset"] = row[f"{metric}_no_reset"]
            result[f"{metric}_delta"] = row[f"{metric}_reset"] - row[f"{metric}_no_reset"]
        rows.append(result)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].head(limit).copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.5g}")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy().tolist()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    return "\n".join(
        [
            format_row(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(format_row(row) for row in rows),
        ]
    )


def write_report(
    *,
    output_dir: Path,
    noise: pd.DataFrame,
    trained: pd.DataFrame,
    comparison: pd.DataFrame,
    untrained: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    noise_summary = (
        noise.groupby("beta", as_index=False)
        .agg(
            target_power_slope=("target_power_slope", "first"),
            estimated_power_slope_mean=("estimated_power_slope_mean", "first"),
            estimated_power_slope_std=("estimated_power_slope_std", "first"),
            event_count_mean=("event_count", "mean"),
            total_pulses_mean=("total_pulses", "mean"),
        )
        .sort_values("beta")
    )
    trained_summary = (
        trained.groupby(["control_mode", "source_protocol_id"], as_index=False)
        .agg(
            weight_erasure=("weight_erasure", "mean"),
            residual_performance=("residual_performance", "mean"),
            savings=("savings", "mean"),
            trace_auc=("trace_auc", "mean"),
            post_delta_norm=("post_delta_norm", "mean"),
            reset_window_neuron_spikes=("reset_window_neuron_spikes", "mean"),
            total_pulses=("total_pulses", "mean"),
        )
        .sort_values(["source_protocol_id", "control_mode"])
    )
    comparison_summary = (
        comparison.groupby("source_protocol_id", as_index=False)
        .agg(
            weight_erasure_delta=("weight_erasure_delta", "mean"),
            residual_performance_delta=("residual_performance_delta", "mean"),
            savings_delta=("savings_delta", "mean"),
            trace_auc_delta=("trace_auc_delta", "mean"),
            post_delta_norm_delta=("post_delta_norm_delta", "mean"),
            reset_window_neuron_spikes_delta=("reset_window_neuron_spikes_delta", "mean"),
        )
        .sort_values("source_protocol_id")
    )
    untrained_summary = (
        untrained.groupby(["control_mode", "source_protocol_id"], as_index=False)
        .agg(
            weight_drift_norm=("weight_drift_norm", "mean"),
            weight_drift_rel_to_baseline=("weight_drift_rel_to_baseline", "mean"),
            pre_behavior=("pre_behavior", "mean"),
            post_behavior=("post_behavior", "mean"),
            trace_auc=("trace_auc", "mean"),
            reset_window_neuron_spikes=("reset_window_neuron_spikes", "mean"),
            total_pulses=("total_pulses", "mean"),
        )
        .sort_values(["source_protocol_id", "control_mode"])
    )
    lines = [
        "# SNN Reset Control Checks",
        "",
        "## Setup",
        "",
        f"- Neurons: `{metadata['neurons']}`",
        f"- Seeds: `{metadata['seeds']}`",
        f"- Protocols: `{metadata['protocol_ids']}`",
        "",
        "## Noise Spectra",
        "",
        markdown_table(
            noise_summary,
            [
                "beta",
                "target_power_slope",
                "estimated_power_slope_mean",
                "estimated_power_slope_std",
                "event_count_mean",
                "total_pulses_mean",
            ],
        ),
        "",
        "## Trained Reset Versus No Reset",
        "",
        markdown_table(
            trained_summary,
            [
                "control_mode",
                "source_protocol_id",
                "weight_erasure",
                "residual_performance",
                "savings",
                "trace_auc",
                "post_delta_norm",
                "reset_window_neuron_spikes",
                "total_pulses",
            ],
        ),
        "",
        "## Reset Minus No Reset",
        "",
        markdown_table(
            comparison_summary,
            [
                "source_protocol_id",
                "weight_erasure_delta",
                "residual_performance_delta",
                "savings_delta",
                "trace_auc_delta",
                "post_delta_norm_delta",
                "reset_window_neuron_spikes_delta",
            ],
        ),
        "",
        "## Untrained Reset Controls",
        "",
        markdown_table(
            untrained_summary,
            [
                "control_mode",
                "source_protocol_id",
                "weight_drift_norm",
                "weight_drift_rel_to_baseline",
                "pre_behavior",
                "post_behavior",
                "trace_auc",
                "reset_window_neuron_spikes",
                "total_pulses",
            ],
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SNN reset noise/control diagnostics.")
    parser.add_argument("--neurons", type=int, default=10_000)
    parser.add_argument("--mean-out-degree", type=int, default=64)
    parser.add_argument("--max-out-degree", type=int, default=96)
    parser.add_argument("--local-candidate-multiplier", type=int, default=6)
    parser.add_argument("--build-workers", type=int, default=1)
    parser.add_argument("--spontaneous-rate-hz", type=float, default=0.12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 4])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--input-channels", type=int, nargs="+", default=[8])
    parser.add_argument("--target-channels", type=int, nargs="+", default=[9])
    parser.add_argument("--max-trials", type=int, default=120)
    parser.add_argument("--eval-interval-trials", type=int, default=8)
    parser.add_argument("--eval-trials", type=int, default=8)
    parser.add_argument("--inter-trial-ms", type=float, default=70.0)
    parser.add_argument("--criterion-response-probability", type=float, default=0.875)
    parser.add_argument("--readout-window-s", type=float, default=1.5)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--noise-repeats", type=int, default=32)
    parser.add_argument("--noise-samples", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocols = selected_protocols()
    output_dir = args.output_dir or (
        Path("experiments/snn_reset/results")
        / datetime.now(timezone.utc).strftime("control_checks_%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "neurons": args.neurons,
        "seeds": list(map(int, args.seeds)),
        "workers": int(args.workers),
        "protocol_ids": [protocol.id for protocol in protocols],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    started = perf_counter()
    noise = noise_diagnostics(
        seed=11,
        repeats=args.noise_repeats,
        n_samples=args.noise_samples,
    )
    noise.to_csv(output_dir / "noise_diagnostics.csv", index=False)

    trained_jobs = [
        (replace(cfg, seed=int(seed)), protocol, int(seed), mode)
        for seed in args.seeds
        for protocol in protocols
        for mode in ["reset", "no_reset"]
    ]
    trained = run_parallel(trained_jobs, run_trained_control, args.workers)
    trained = trained.sort_values(["source_protocol_id", "control_mode", "seed"])
    trained.to_csv(output_dir / "trained_controls.csv", index=False)

    comparison = summarize_reset_vs_no_reset(trained)
    comparison.to_csv(output_dir / "reset_vs_no_reset.csv", index=False)

    untrained_jobs = [
        (replace(cfg, seed=int(seed)), protocol, int(seed), mode)
        for seed in args.seeds
        for protocol in protocols
        for mode in ["reset", "no_reset"]
    ]
    untrained = run_parallel(untrained_jobs, run_untrained_control, args.workers)
    untrained = untrained.sort_values(["source_protocol_id", "control_mode", "seed"])
    untrained.to_csv(output_dir / "untrained_controls.csv", index=False)

    metadata["elapsed_s"] = perf_counter() - started
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        output_dir=output_dir,
        noise=noise,
        trained=trained,
        comparison=comparison,
        untrained=untrained,
        metadata=metadata,
    )
    print(json.dumps({"output_dir": str(output_dir), "elapsed_s": metadata["elapsed_s"]}, sort_keys=True))


if __name__ == "__main__":
    main()
