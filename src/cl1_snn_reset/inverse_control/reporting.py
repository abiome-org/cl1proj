from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .controllability import ControllabilityReport
from .inverse_optimizer import CandidateProtocol
from .rollout_dataset import CausalDeltaDataset


def write_controllability_artifacts(
    *,
    report: ControllabilityReport,
    dataset: CausalDeltaDataset,
    diagnostics: dict[str, Any],
    output_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "controllability_arrays.npz",
        jacobian=report.jacobian,
        reachable_component=report.reachable_component,
        desired_delta=report.desired_delta,
    )
    (output_dir / "controllability_summary.json").write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return write_markdown_report(
        path=output_dir / "controllability_report.md",
        title="Learned Inverse Reset Controllability",
        lines=_controllability_lines(report, dataset, diagnostics, metadata),
    )


def write_candidate_csv(candidates: list[CandidateProtocol], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "optimized_protocols.csv"
    pd.DataFrame([candidate.to_row() for candidate in candidates]).to_csv(path, index=False)
    with (output_dir / "optimized_protocols.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = candidate.stim_program.to_json() | {
                "protocol_id": candidate.protocol_id,
                "predicted_loss": candidate.predicted_loss,
                "predicted_task_erasure": candidate.predicted_task_erasure,
                "predicted_health_penalty": candidate.predicted_health_penalty,
                "model_uncertainty": candidate.model_uncertainty,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def write_inverse_reset_report(
    *,
    candidates: list[CandidateProtocol],
    validation: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_table = pd.DataFrame([candidate.to_row() for candidate in candidates])
    if not validation.empty:
        validation_summary = (
            validation.groupby(["protocol_id", "program_family"], as_index=False)
            .agg(
                validated_causal_task_erasure=("validated_causal_task_erasure", "mean"),
                validated_weight_erasure=("validated_weight_erasure", "mean"),
                validated_path_erasure=("validated_path_erasure", "mean"),
                validated_residual_performance=("validated_residual_performance", "mean"),
                validated_savings=("validated_savings", "mean"),
                validated_trace_auc=("validated_trace_auc", "mean"),
                validated_health=("validated_health", "mean"),
                validated_orthogonal_damage=("validated_orthogonal_damage", "mean"),
                stimulus_effect_norm=("stimulus_effect_norm", "mean"),
                beats_no_reset=("beats_no_reset", "mean"),
                passes_health_criterion=("passes_health_criterion", "mean"),
                passes_generalization_criterion=("passes_generalization_criterion", "mean"),
            )
            .sort_values("validated_causal_task_erasure", ascending=False)
        )
    else:
        validation_summary = pd.DataFrame()
    lines = [
        "## Run Metadata",
        "",
        f"- Run ID: `{metadata.get('run_id', 'unknown')}`",
        f"- Mode: `{metadata.get('mode', 'unknown')}`",
        f"- Dataset examples: `{metadata.get('dataset_examples', 'unknown')}`",
        f"- Candidates: `{len(candidates)}`",
        "",
        "## Candidate Protocols",
        "",
        _markdown_table(
            candidate_table,
            [
                "protocol_id",
                "program_family",
                "duration_s",
                "energy_cost",
                "predicted_loss",
                "predicted_task_erasure",
                "predicted_health_penalty",
                "model_uncertainty",
            ],
            limit=20,
        ),
        "",
        "## Paired Validation",
        "",
        _markdown_table(
            validation_summary,
            [
                "protocol_id",
                "program_family",
                "validated_causal_task_erasure",
                "validated_weight_erasure",
                "validated_path_erasure",
                "validated_residual_performance",
                "validated_savings",
                "validated_trace_auc",
                "validated_health",
                "validated_orthogonal_damage",
                "stimulus_effect_norm",
                "beats_no_reset",
                "passes_health_criterion",
                "passes_generalization_criterion",
            ],
            limit=20,
        ),
        "",
        "## Failure Cases",
        "",
        _failure_cases(validation_summary),
        "",
        "## Next Grammar Changes",
        "",
        _next_steps(validation_summary),
    ]
    return write_markdown_report(
        path=output_dir / "inverse_reset_report.md",
        title="Learned Inverse Reset Report",
        lines=lines,
    )


def write_markdown_report(path: Path, title: str, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")
    return path


def _controllability_lines(
    report: ControllabilityReport,
    dataset: CausalDeltaDataset,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    aligned = pd.DataFrame(report.top_aligned_dimensions)
    anti = pd.DataFrame(report.top_anti_aligned_dimensions)
    groups = pd.DataFrame(
        [
            {"feature_group": key, **value}
            for key, value in report.group_reachability.items()
        ]
    ).sort_values("controllable_fraction", ascending=False)
    actuator = _actuator_causality_table(dataset)
    max_effect = float(actuator["mean_stimulus_effect_norm"].max()) if not actuator.empty else 0.0
    if max_effect <= 1e-12:
        criterion_status = (
            "Actuator causality criterion fails: no sampled stimulation family "
            "produced causal change beyond no-reset."
        )
    elif report.controllable_fraction < 0.30:
        criterion_status = (
            "Actuator causality criterion passes; reachability criterion fails "
            "because anti-trace reachability is weak."
        )
    else:
        criterion_status = (
            "Actuator causality and reachability criteria pass: proceed to "
            "candidate validation before claiming reset."
        )
    return [
        "## Run Metadata",
        "",
        f"- Run ID: `{metadata.get('run_id', 'unknown')}`",
        f"- State vector version: `{dataset.state_spec.version}`",
        f"- State vector hash: `{dataset.state_spec.spec_hash()}`",
        f"- Dataset size: `{len(dataset.examples)}`",
        f"- Model RMSE: `{diagnostics.get('rmse', float('nan')):.5g}`",
        f"- Mean cosine similarity: `{diagnostics.get('mean_cosine_similarity', float('nan')):.5g}`",
        "",
        "## Reachable Subspace",
        "",
        f"- Target anti-trace norm: `{report.target_norm:.6g}`",
        f"- Reachable norm: `{report.reachable_norm:.6g}`",
        f"- Unreachable residual norm: `{report.unreachable_residual_norm:.6g}`",
        f"- Controllable fraction: `{report.controllable_fraction:.6g}`",
        f"- Best linear predicted reset loss: `{report.best_linear_predicted_reset_loss:.6g}`",
        f"- Recommendation: {report.recommendation}",
        f"- Scientific criterion status: {criterion_status}",
        "",
        "## Controllability By Seed",
        "",
        _markdown_table(
            pd.DataFrame(diagnostics.get("controllability_by_seed", [])),
            [
                "seed",
                "example_index",
                "controllable_fraction",
                "target_norm",
                "reachable_norm",
                "unreachable_residual_norm",
            ],
            limit=20,
        ),
        "",
        "## Actuator Causality By Family",
        "",
        _markdown_table(
            actuator,
            [
                "program_family",
                "examples",
                "mean_stimulus_effect_norm",
                "max_stimulus_effect_norm",
                "mean_spike_delta",
                "mean_weight_delta_norm",
                "mean_energy_cost",
            ],
            limit=20,
        ),
        "",
        "## Top Aligned Stimulation Dimensions",
        "",
        _markdown_table(aligned, ["dimension", "score"], limit=10),
        "",
        "## Top Anti-Aligned Stimulation Dimensions",
        "",
        _markdown_table(anti, ["dimension", "score"], limit=10),
        "",
        "## Feature Group Reachability",
        "",
        _markdown_table(
            groups,
            [
                "feature_group",
                "target_norm",
                "reachable_norm",
                "unreachable_residual_norm",
                "controllable_fraction",
            ],
            limit=20,
        ),
    ]


def _markdown_table(df: pd.DataFrame, columns: list[str], limit: int) -> str:
    if df.empty:
        return "_No rows._"
    present = [column for column in columns if column in df.columns]
    view = df.loc[:, present].head(limit).copy()
    for column in view.select_dtypes(include=["float", "float64"]).columns:
        view[column] = view[column].map(lambda value: f"{value:.5g}")
    view = view.fillna("")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy().tolist()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    return "\n".join(
        [
            format_row(headers),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(format_row(row) for row in rows),
        ]
    )


def _failure_cases(validation: pd.DataFrame) -> str:
    if validation.empty:
        return "No validation rows were produced."
    failures = validation[
        (validation["beats_no_reset"] < 0.5)
        | (validation["passes_health_criterion"] < 0.5)
    ]
    if failures.empty:
        return "No dominant validation failure among the selected candidates."
    return _markdown_table(
        failures,
        [
            "protocol_id",
            "program_family",
            "validated_causal_task_erasure",
            "validated_health",
            "stimulus_effect_norm",
            "beats_no_reset",
        ],
        limit=10,
    )


def _next_steps(validation: pd.DataFrame) -> str:
    if validation.empty:
        return "Run paired validation before interpreting optimized candidates."
    if bool((validation["beats_no_reset"] >= 0.5).any()):
        return (
            "Outcome C candidate family detected: increase held-out seeds and task pairs, "
            "then run closed-loop block-level validation."
        )
    if bool((validation["stimulus_effect_norm"] > 1e-9).any()):
        return (
            "Outcome B: stimulation is causal but not yet anti-trace aligned. Expand "
            "anti-causal timing, probe-triggered, and stronger positive-control families."
        )
    return (
        "Outcome A: actuator appears inert under this grammar. Verify pulse-driven "
        "spikes enter STDP and increase positive-control dose before trusting the model."
    )


def _actuator_causality_table(dataset: CausalDeltaDataset) -> pd.DataFrame:
    rows = []
    for example in dataset.examples:
        rows.append(
            {
                "program_family": example.metadata.get("program_family", "unknown"),
                "stimulus_effect_norm": float(np.linalg.norm(example.causal_delta)),
                "energy_cost": float(example.energy_cost),
                "spike_delta": float(example.metadata.get("spike_delta", np.nan)),
                "weight_delta_norm": float(example.metadata.get("weight_delta_norm", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby("program_family", as_index=False)
        .agg(
            examples=("program_family", "count"),
            mean_stimulus_effect_norm=("stimulus_effect_norm", "mean"),
            max_stimulus_effect_norm=("stimulus_effect_norm", "max"),
            mean_spike_delta=("spike_delta", "mean"),
            mean_weight_delta_norm=("weight_delta_norm", "mean"),
            mean_energy_cost=("energy_cost", "mean"),
        )
        .sort_values("mean_stimulus_effect_norm", ascending=False)
    )
