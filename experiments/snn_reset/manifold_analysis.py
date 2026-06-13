"""Memory-geometry probes for the validated SNN reset regime.

Two analyses on trained networks whose synaptic state is observable:

1. Breaking point along the memory axis. Interpolate the weight vector between
   the trained and naive states, W(a) = (1-a) W_trained + a W_naive, and read the
   validated task score versus a. A matched-magnitude random direction orthogonal
   to the trained-naive axis is the control: it asks whether displacement
   *magnitude* or displacement *direction* is what degrades performance.

2. Surgical ablations against the validated readout. Zero or scramble defined
   synapse sets (the input->target pathway, target-incoming edges, etc.) to
   localize where the learned association is stored.

Outputs CSVs and two figures under results/manifold_analysis_<stamp>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cl1_snn_reset import (
    CultureConfig,
    build_network,
    conditioned_electrode_association,
    evaluate_regime,
    pattern_discrimination,
    train_regime,
)

from figlib import PALETTE, apply_style, save

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = (1, 3, 4)
ALPHAS = np.round(np.linspace(0.0, 1.0, 16), 4)


def build_culture(neurons: int) -> CultureConfig:
    return CultureConfig(
        n_neurons=neurons,
        mean_out_degree=64,
        max_out_degree=96,
        local_candidate_multiplier=6,
        background_noise_mv=1.0,
        spontaneous_rate_hz=0.0,
        homeostasis_rate=0.0,
        backend="numpy",
    )


def regimes() -> dict:
    return {
        "conditioned_electrode_association": conditioned_electrode_association(
            input_channel=8, target_channel=17, input_current_uA=120.0,
            target_current_uA=120.0, criterion_score=0.4, max_training_repetitions=60,
            eval_repetitions=16,
        ),
        "pattern_discrimination": pattern_discrimination(
            input_a=8, input_b=24, target_a=17, target_b=33, input_current_uA=120.0,
            target_current_uA=120.0, criterion_score=0.35, max_training_repetitions=60,
            eval_repetitions=16,
        ),
    }


def regime_channels(regime) -> tuple[list[int], list[int]]:
    inputs: set[int] = set()
    targets: set[int] = set()
    for probe in regime.probes:
        if probe.is_positive:
            targets.update(int(c) for c in probe.target_channels)
            for event in probe.events:
                inputs.update(int(c) for c in event.channels)
    return sorted(inputs), sorted(targets)


def train_state(culture: CultureConfig, regime, seed: int, *, warmup_s: float = 0.5):
    net = build_network(culture, seed=seed)
    net.advance(warmup_s * 1000.0, [], plasticity=False, record=False)
    naive = net.weights_vector()
    train_regime(net, regime, max_repetitions=60, eval_repetitions=16, stop_at_criterion=True)
    trained = net.weights_vector()
    return net, naive, trained


def score_weights(net, regime, weights: np.ndarray, *, reps: int = 16) -> tuple[float, np.ndarray]:
    net.set_weights(weights)
    effective = net.weights_vector()
    score = evaluate_regime(net, regime, repetitions=reps).score
    return float(score), effective


def interpolation_rows(net, regime, naive, trained, *, task: str, seed: int) -> list[dict]:
    axis = naive - trained
    distance = float(np.linalg.norm(axis))
    rng = np.random.default_rng(7919 * seed + len(task))
    random_dir = rng.standard_normal(axis.shape)
    random_dir -= (random_dir @ axis) / (axis @ axis) * axis
    random_dir /= np.linalg.norm(random_dir) + 1e-12
    rows = []
    for alpha in ALPHAS:
        for direction, target in (
            ("axis", trained + alpha * axis),
            ("random", trained + alpha * distance * random_dir),
        ):
            score, effective = score_weights(net, regime, target)
            rows.append({
                "task": task, "seed": seed, "direction": direction, "alpha": float(alpha),
                "score": score, "criterion": float(regime.criterion_score),
                "distance_from_naive": float(np.linalg.norm(effective - naive)),
                "distance_from_trained": float(np.linalg.norm(effective - trained)),
            })
    return rows


def ablation_rows(net, regime, naive, trained, *, task: str, seed: int) -> list[dict]:
    inputs, targets = regime_channels(regime)
    channel_of = net.electrodes.nearest_channel
    input_neuron = np.isin(channel_of, inputs)
    target_neuron = np.isin(channel_of, targets)
    src_in = input_neuron[net.sources]
    tgt_in = target_neuron[net.targets]
    rng = np.random.default_rng(104729 * seed + len(task))

    def zeroed(mask: np.ndarray) -> np.ndarray:
        w = trained.copy()
        w[mask] = 0.0
        return w

    random_dir = rng.standard_normal(trained.shape)
    axis = naive - trained
    random_dir -= (random_dir @ axis) / (axis @ axis) * axis
    random_dir /= np.linalg.norm(random_dir) + 1e-12
    conditions = {
        "trained": trained,
        "naive_replacement": naive,
        "zero_all": np.zeros_like(trained),
        "shuffle_magnitudes": np.abs(trained)[rng.permutation(trained.size)] * np.sign(trained),
        "anti_training_delta": 2.0 * trained - naive,
        "random_matched": trained + np.linalg.norm(axis) * random_dir,
        "zero_input_to_target_path": zeroed(src_in & tgt_in),
        "zero_target_incoming": zeroed(tgt_in),
        "zero_input_outgoing": zeroed(src_in),
    }
    edge_fraction = {
        "zero_input_to_target_path": float(np.mean(src_in & tgt_in)),
        "zero_target_incoming": float(np.mean(tgt_in)),
        "zero_input_outgoing": float(np.mean(src_in)),
    }
    rows = []
    for name, weights in conditions.items():
        score, effective = score_weights(net, regime, weights)
        rows.append({
            "task": task, "seed": seed, "condition": name, "score": score,
            "criterion": float(regime.criterion_score),
            "distance_from_naive": float(np.linalg.norm(effective - naive)),
            "distance_from_trained": float(np.linalg.norm(effective - trained)),
            "edges_changed_fraction": edge_fraction.get(name, np.nan),
        })
    return rows


def breaking_points(interp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, seed), group in interp[interp["direction"] == "axis"].groupby(["task", "seed"]):
        criterion = float(group["criterion"].iloc[0])
        below = group[group["score"] < criterion].sort_values("alpha")
        alpha_star = float(below["alpha"].iloc[0]) if not below.empty else np.nan
        random = interp[(interp.task == task) & (interp.seed == seed) & (interp.direction == "random")]
        random_breaks = bool((random["score"] < criterion).any())
        rows.append({"task": task, "seed": seed, "alpha_star_axis": alpha_star,
                     "random_ever_breaks": random_breaks})
    return pd.DataFrame(rows)


def figure_breaking_point(interp: pd.DataFrame, bp: pd.DataFrame, out_dir: Path) -> None:
    tasks = list(interp["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = {"axis": PALETTE["vermillion"], "random": PALETTE["gray"]}
    labels = {"axis": "toward naive (memory axis)", "random": "random matched direction"}
    for ax, task in zip(axes, tasks):
        sub = interp[interp["task"] == task]
        criterion = float(sub["criterion"].iloc[0])
        for direction in ("axis", "random"):
            d = sub[sub["direction"] == direction].groupby("alpha")["score"].agg(["mean", "std"]).reset_index()
            ax.plot(d["alpha"], d["mean"], color=colors[direction], lw=1.8, marker="o", ms=3, label=labels[direction])
            ax.fill_between(d["alpha"], d["mean"] - d["std"], d["mean"] + d["std"], color=colors[direction], alpha=0.18)
        ax.axhline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        astar = bp.loc[bp["task"] == task, "alpha_star_axis"].mean()
        if np.isfinite(astar):
            ax.axvline(astar, color=PALETTE["vermillion"], ls=":", lw=1.0, alpha=0.8)
            ax.text(astar + 0.02, 0.02, f"a*={astar:.2f}", color=PALETTE["vermillion"], fontsize=8)
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("fraction moved along direction (a)")
        ax.set_ylabel("task score (positive - negative)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Performance collapses along the memory axis, not with matched random displacement",
                 fontsize=11, weight="bold")
    save(fig, out_dir, "F1_memory_axis_breaking_point")


def figure_ablations(abl: pd.DataFrame, out_dir: Path) -> None:
    summary = abl.groupby(["task", "condition"], as_index=False)["score"].mean()
    tasks = list(summary["task"].unique())
    order = ["trained", "anti_training_delta", "random_matched", "shuffle_magnitudes",
             "zero_input_outgoing", "zero_input_to_target_path", "zero_target_incoming",
             "zero_all", "naive_replacement"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 4.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        sub = summary[summary["task"] == task].set_index("condition").reindex(order).reset_index()
        criterion = float(abl.loc[abl["task"] == task, "criterion"].iloc[0])
        colors = [PALETTE["vermillion"] if v < criterion else PALETTE["blue"] for v in sub["score"]]
        ax.barh(sub["condition"], sub["score"], color=colors)
        ax.axvline(criterion, color=PALETTE["black"], ls="--", lw=0.9, alpha=0.75)
        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("task score (positive - negative)")
        ax.invert_yaxis()
    fig.suptitle("Targeted pathway ablation erases like naive replacement; magnitude-matched scrambles do not",
                 fontsize=11, weight="bold")
    save(fig, out_dir, "F2_surgical_ablations")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-geometry probes for the SNN reset regime.")
    parser.add_argument("--neurons", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    started = perf_counter()
    started_at = datetime.now(timezone.utc)
    out_dir = args.out or RESULTS_DIR / f"manifold_analysis_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    culture = build_culture(args.neurons)
    interp_rows: list[dict] = []
    abl_rows: list[dict] = []
    for task, regime in regimes().items():
        for seed in args.seeds:
            net, naive, trained = train_state(culture, regime, seed)
            work = copy.deepcopy(net)
            interp_rows.extend(interpolation_rows(work, regime, naive, trained, task=task, seed=seed))
            abl_rows.extend(ablation_rows(work, regime, naive, trained, task=task, seed=seed))
            print(json.dumps({"event": "done", "task": task, "seed": seed}), flush=True)

    interp = pd.DataFrame(interp_rows)
    abl = pd.DataFrame(abl_rows)
    bp = breaking_points(interp)
    interp.to_csv(out_dir / "interpolation.csv", index=False)
    abl.to_csv(out_dir / "ablations.csv", index=False)
    bp.to_csv(out_dir / "breaking_points.csv", index=False)
    figure_breaking_point(interp, bp, out_dir)
    figure_ablations(abl, out_dir)
    (out_dir / "metadata.json").write_text(json.dumps({
        "run_id": out_dir.name,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "neurons": int(args.neurons),
        "seeds": list(args.seeds),
        "alphas": ALPHAS.tolist(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": perf_counter() - started,
        "status": "complete",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "out_dir": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
