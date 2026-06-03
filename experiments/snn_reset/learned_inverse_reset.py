from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cl1_snn_reset.config import CultureConfig, ExperimentConfig, TaskConfig, to_dict
from cl1_snn_reset.inverse_control import (
    CMAESStimOptimizer,
    CausalDeltaDataset,
    CausalDeltaDatasetBuilder,
    HybridStateProjector,
    InverseResetObjective,
    RandomSearchOptimizer,
    RidgeDeltaModel,
    StimConstraints,
    StimProgram,
    analyze_controllability,
    build_target_state,
    evaluate_forward_model,
)
from cl1_snn_reset.inverse_control.inverse_optimizer import CandidateProtocol
from cl1_snn_reset.inverse_control.reporting import (
    write_candidate_csv,
    write_controllability_artifacts,
    write_inverse_reset_report,
)
from cl1_snn_reset.inverse_control.validation import validate_candidates_against_no_reset


def main() -> None:
    args = _parse_args()
    payload = _load_yaml(args.config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("inverse_reset_%Y%m%dT%H%M%SZ")
    output_root = Path(payload.get("run", {}).get("output_dir", "experiments/snn_reset/results"))
    output_dir = args.output_dir or output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = _experiment_config(payload)
    constraints = StimConstraints.from_dict(payload.get("constraints", {}))
    projector = _projector(payload, experiment_config)
    metadata = _metadata(run_id, args.mode, output_dir, payload, experiment_config, constraints)
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"event": "start", "run_id": run_id, "mode": args.mode, "output_dir": str(output_dir)}), flush=True)

    dataset: CausalDeltaDataset | None = None
    model: RidgeDeltaModel | None = None
    candidates: list[CandidateProtocol] = []
    validation = None

    if args.mode in {"dataset", "full"}:
        dataset = _build_dataset(payload, experiment_config, constraints, projector)
        dataset.save(output_dir / "dataset")
        metadata["dataset_examples"] = len(dataset.examples)
        _write_json(output_dir / "metadata.json", metadata)
        print(json.dumps({"event": "dataset", "examples": len(dataset.examples)}), flush=True)

    if args.mode in {"controllability", "optimize", "validate"}:
        dataset = CausalDeltaDataset.load(args.dataset_dir or output_dir / "dataset")

    if args.mode in {"controllability", "optimize", "full"}:
        assert dataset is not None
        model = _fit_model(payload, dataset)
        _save_model(model, output_dir / "models" / "linear_delta_model.pkl")
        diagnostics = evaluate_forward_model(model, dataset, state_spec=dataset.state_spec)
        report = analyze_controllability(
            model=model,
            dataset=dataset,
            state_spec=dataset.state_spec,
            target_mode=payload.get("target", {}).get("mode", "trace_removed"),
        )
        diagnostics["controllability_by_seed"] = _controllability_by_seed(
            model,
            dataset,
            target_mode=payload.get("target", {}).get("mode", "trace_removed"),
        )
        write_controllability_artifacts(
            report=report,
            dataset=dataset,
            diagnostics=diagnostics,
            output_dir=output_dir / "reports",
            metadata=metadata | {"dataset_examples": len(dataset.examples)},
        )
        metadata["model_diagnostics"] = diagnostics
        metadata["controllability"] = report.to_json_dict()
        _write_json(output_dir / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "event": "controllability",
                    "fraction": report.controllable_fraction,
                    "recommendation": report.recommendation,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.mode == "validate":
        model = _load_model(args.model_path or output_dir / "models" / "linear_delta_model.pkl")
        candidates = _load_candidates(args.candidates_path or output_dir / "candidates" / "optimized_protocols.jsonl")

    if args.mode in {"optimize", "full"}:
        assert dataset is not None and model is not None
        candidates = _optimize(payload, dataset, model, constraints)
        write_candidate_csv(candidates, output_dir / "candidates")
        print(json.dumps({"event": "optimize", "candidates": len(candidates)}), flush=True)

    if args.mode in {"validate", "full"}:
        assert dataset is not None
        if not candidates:
            candidates = _load_candidates(args.candidates_path or output_dir / "candidates" / "optimized_protocols.jsonl")
        validation_seeds = tuple(int(seed) for seed in payload.get("seeds", {}).get("heldout", ())) or tuple(
            int(seed) for seed in payload.get("seeds", {}).get("train", (1,))
        )
        validation = validate_candidates_against_no_reset(
            candidates=candidates,
            experiment_config=experiment_config,
            projector=projector,
            state_spec=dataset.state_spec,
            seeds=validation_seeds,
            target_mode=payload.get("target", {}).get("mode", "trace_removed"),
            output_dir=output_dir / "validation",
            limit=int(payload.get("validation", {}).get("candidate_limit", len(candidates))),
        )
        write_inverse_reset_report(
            candidates=candidates,
            validation=validation,
            output_dir=output_dir / "reports",
            metadata=metadata | {"dataset_examples": len(dataset.examples), "mode": args.mode},
        )
        print(json.dumps({"event": "validate", "rows": len(validation)}), flush=True)

    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"event": "complete", "output_dir": str(output_dir)}), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learned inverse reset controller experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["dataset", "controllability", "optimize", "validate", "full"],
        default="full",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--candidates-path", type=Path, default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _experiment_config(payload: dict[str, Any]) -> ExperimentConfig:
    network = payload.get("network", {})
    task = payload.get("task", {})
    culture = CultureConfig(
        n_neurons=int(network.get("n_neurons", 10_000)),
        mean_out_degree=int(network.get("mean_out_degree", 64)),
        max_out_degree=int(network.get("max_out_degree", max(96, int(network.get("mean_out_degree", 64))))),
        n_electrodes=int(network.get("n_electrodes", 64)),
        backend=str(network.get("backend", "numpy")),
        build_workers=int(network.get("build_workers", 1)),
        local_candidate_multiplier=int(network.get("local_candidate_multiplier", 6)),
        spontaneous_rate_hz=float(network.get("spontaneous_rate_hz", 0.12)),
        background_noise_mv=float(network.get("background_noise_mv", 1.4)),
        stim_gain_mv_per_uA=float(network.get("stim_gain_mv_per_uA", 5.5)),
    )
    task_cfg = TaskConfig(
        input_channels=tuple(int(v) for v in task.get("input_channels", [task.get("input_channel", 8)])),
        target_channels=tuple(int(v) for v in task.get("target_channels", [task.get("target_channel", 9)])),
        criterion_response_probability=float(task.get("criterion", task.get("criterion_response_probability", 0.875))),
        max_trials=int(task.get("max_train_trials", task.get("max_trials", 100))),
        eval_trials=int(task.get("eval_trials", 6)),
        eval_interval_trials=int(task.get("eval_interval_trials", 5)),
        inter_trial_ms=float(task.get("inter_trial_ms", 70.0)),
        pair_delay_ms=float(task.get("pair_delay_ms", 12.0)),
        input_current_uA=float(task.get("input_current_uA", 2.2)),
        target_current_uA=float(task.get("target_current_uA", 2.0)),
        pulse_width_us=int(task.get("pulse_width_us", 160)),
    )
    return ExperimentConfig(
        culture=culture,
        task=task_cfg,
        readout_window_s=float(payload.get("state", {}).get("readout_window_s", 0.5)),
        warmup_s=float(payload.get("run", {}).get("warmup_s", 0.0)),
        seed=int(payload.get("run", {}).get("random_seed", 123)),
        keep_snapshots=False,
    )


def _projector(payload: dict[str, Any], cfg: ExperimentConfig) -> HybridStateProjector:
    state = payload.get("state", {})
    pca = state.get("pca_components", {})
    return HybridStateProjector(
        cfg.task,
        n_electrodes=cfg.culture.n_electrodes,
        weight_projection_dim=int(pca.get("weight_projection", state.get("weight_projection_dim", 32))),
        weight_hist_bins=int(state.get("weight_hist_bins", 16)),
        projector_seed=int(state.get("projector_seed", 1729)),
        include_observable=bool(state.get("include_observable", True)),
        include_privileged=bool(state.get("include_privileged", True)),
    )


def _build_dataset(
    payload: dict[str, Any],
    experiment_config: ExperimentConfig,
    constraints: StimConstraints,
    projector: HybridStateProjector,
) -> CausalDeltaDataset:
    seeds = tuple(int(seed) for seed in payload.get("seeds", {}).get("train", (1, 3, 4)))
    sampling = dict(payload.get("stim_sampling", {}))
    builder = CausalDeltaDatasetBuilder(
        projector=projector,
        experiment_config=experiment_config,
        constraints=constraints,
        seeds=seeds,
        programs_per_trained_state=int(sampling.get("programs_per_trained_state", 100)),
        stim_sampling=sampling,
        random_seed=int(payload.get("run", {}).get("random_seed", 123)),
    )
    return builder.build()


def _fit_model(payload: dict[str, Any], dataset: CausalDeltaDataset) -> RidgeDeltaModel:
    model_config = payload.get("model", {})
    alphas = tuple(float(v) for v in model_config.get("regularization", {}).get("ridge_alpha", (0.1, 1.0, 10.0)))
    model = RidgeDeltaModel(
        alphas=alphas,
        max_state_features=int(model_config.get("max_state_features", 64)),
        max_interaction_state_features=int(model_config.get("max_interaction_state_features", 24)),
        max_interaction_control_features=int(model_config.get("max_interaction_control_features", 24)),
        random_seed=int(payload.get("run", {}).get("random_seed", 123)),
    )
    model.fit(
        dataset.trained_states,
        dataset.stim_features,
        dataset.regime_features,
        dataset.causal_deltas,
    )
    return model


def _optimize(
    payload: dict[str, Any],
    dataset: CausalDeltaDataset,
    model: RidgeDeltaModel,
    constraints: StimConstraints,
) -> list[CandidateProtocol]:
    optimizer_config = payload.get("optimizer", {})
    objective = InverseResetObjective(
        dataset.state_spec,
        loss_weights=payload.get("loss_weights", {}),
        max_energy_cost=constraints.max_energy_cost,
    )
    example_index = int(optimizer_config.get("example_index", 0))
    baseline = dataset.baseline_states[example_index]
    trained = dataset.trained_states[example_index]
    no_reset = dataset.no_reset_states[example_index]
    target = build_target_state(
        dataset.state_spec,
        baseline,
        trained,
        no_reset,
        mode=payload.get("target", {}).get("mode", "trace_removed"),
    )
    optimizer_type = str(optimizer_config.get("type", "cma_es")).lower()
    klass = CMAESStimOptimizer if optimizer_type in {"cma_es", "cmaes", "cem"} else RandomSearchOptimizer
    optimizer = klass(
        max_evaluations=int(optimizer_config.get("max_evaluations", 1000)),
        random_seed=int(payload.get("run", {}).get("random_seed", 123)),
    )
    return optimizer.propose(
        model=model,
        x_current=trained,
        no_reset_state=no_reset,
        target_state=target,
        objective=objective,
        constraints=constraints,
        input_channel=int(dataset.examples[example_index].task_input_channel),
        target_channel=int(dataset.examples[example_index].task_target_channel),
        stim_sampling=payload.get("stim_sampling", {}),
        candidate_count=int(optimizer_config.get("candidates_per_state", 20)),
        protocol_prefix="optimized",
    )


def _controllability_by_seed(
    model: RidgeDeltaModel,
    dataset: CausalDeltaDataset,
    *,
    target_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, example in enumerate(dataset.examples):
        if example.seed in seen:
            continue
        seen.add(example.seed)
        report = analyze_controllability(
            model=model,
            dataset=dataset,
            state_spec=dataset.state_spec,
            target_mode=target_mode,
            example_index=index,
        )
        rows.append(
            {
                "seed": int(example.seed),
                "example_index": int(index),
                "controllable_fraction": report.controllable_fraction,
                "target_norm": report.target_norm,
                "reachable_norm": report.reachable_norm,
                "unreachable_residual_norm": report.unreachable_residual_norm,
            }
        )
    return rows


def _save_model(model: RidgeDeltaModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def _load_model(path: Path) -> RidgeDeltaModel:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_candidates(path: Path) -> list[CandidateProtocol]:
    candidates: list[CandidateProtocol] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        program = StimProgram.from_json(payload)
        candidates.append(
            CandidateProtocol(
                protocol_id=str(payload.get("protocol_id", f"loaded_{index:05d}")),
                stim_program=program,
                predicted_delta=np.array([], dtype=np.float64),
                predicted_post_state=np.array([], dtype=np.float64),
                predicted_loss=float(payload.get("predicted_loss", 0.0)),
                predicted_task_erasure=float(payload.get("predicted_task_erasure", 0.0)),
                predicted_health_penalty=float(payload.get("predicted_health_penalty", 0.0)),
                predicted_energy_cost=float(payload.get("estimated_energy_cost", 0.0)),
                model_uncertainty=float(payload.get("model_uncertainty", 0.0)),
            )
        )
    return candidates


def _metadata(
    run_id: str,
    mode: str,
    output_dir: Path,
    payload: dict[str, Any],
    experiment_config: ExperimentConfig,
    constraints: StimConstraints,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": payload,
        "experiment_config": to_dict(experiment_config),
        "constraints": asdict(constraints),
        "output_dir": str(output_dir),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
