from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rollout_dataset import CausalDeltaDataset
from .state_projectors import StateVectorSpec, build_target_state


@dataclass(frozen=True)
class ControllabilityReport:
    target_norm: float
    reachable_norm: float
    unreachable_residual_norm: float
    controllable_fraction: float
    best_linear_predicted_reset_loss: float
    top_aligned_dimensions: tuple[dict[str, Any], ...]
    top_anti_aligned_dimensions: tuple[dict[str, Any], ...]
    group_reachability: dict[str, dict[str, float]]
    recommendation: str
    jacobian: np.ndarray
    reachable_component: np.ndarray
    desired_delta: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target_norm": self.target_norm,
            "reachable_norm": self.reachable_norm,
            "unreachable_residual_norm": self.unreachable_residual_norm,
            "controllable_fraction": self.controllable_fraction,
            "best_linear_predicted_reset_loss": self.best_linear_predicted_reset_loss,
            "top_aligned_dimensions": list(self.top_aligned_dimensions),
            "top_anti_aligned_dimensions": list(self.top_anti_aligned_dimensions),
            "group_reachability": self.group_reachability,
            "recommendation": self.recommendation,
        }


def analyze_controllability(
    *,
    model,
    dataset: CausalDeltaDataset,
    state_spec: StateVectorSpec,
    target_mode: str = "trace_removed",
    example_index: int = 0,
) -> ControllabilityReport:
    baseline = dataset.baseline_states[example_index]
    trained = dataset.trained_states[example_index]
    no_reset = dataset.no_reset_states[example_index]
    target = build_target_state(
        state_spec,
        baseline,
        trained,
        no_reset,
        mode=target_mode,
    )
    desired = target - no_reset
    stim = np.mean(dataset.stim_features, axis=0, keepdims=True)
    regime = np.mean(dataset.regime_features, axis=0, keepdims=True)
    x_current = trained[None, :]
    jacobian = fit_local_jacobian(model, x_current, stim, regime)
    reachable, coeffs = project_target_onto_reachable_subspace(jacobian, desired)
    residual = desired - reachable
    target_norm = float(np.linalg.norm(desired))
    reachable_norm = float(np.linalg.norm(reachable))
    residual_norm = float(np.linalg.norm(residual))
    fraction = controllable_fraction(desired, reachable)
    score = jacobian.T @ desired
    top_aligned = _top_control_dimensions(score, dataset, descending=True)
    top_anti = _top_control_dimensions(score, dataset, descending=False)
    group_reachability = _group_reachability(state_spec, desired, reachable, residual)
    loss = float(np.mean(np.square(residual)))
    return ControllabilityReport(
        target_norm=target_norm,
        reachable_norm=reachable_norm,
        unreachable_residual_norm=residual_norm,
        controllable_fraction=fraction,
        best_linear_predicted_reset_loss=loss,
        top_aligned_dimensions=tuple(top_aligned),
        top_anti_aligned_dimensions=tuple(top_anti),
        group_reachability=group_reachability,
        recommendation=_recommendation(fraction),
        jacobian=jacobian,
        reachable_component=reachable,
        desired_delta=desired,
    )


def fit_local_jacobian(
    model,
    x_current: np.ndarray,
    stim_features: np.ndarray,
    regime_features: np.ndarray,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    if hasattr(model, "control_jacobian"):
        return np.asarray(
            model.control_jacobian(
                x_current,
                stim_features,
                regime_features,
                epsilon=epsilon,
            ),
            dtype=np.float64,
        )
    base = model.predict_delta(x_current, stim_features, regime_features)[0]
    control = np.hstack([stim_features, regime_features])
    columns: list[np.ndarray] = []
    for index in range(control.shape[1]):
        shifted = control.copy()
        step = epsilon * max(1.0, abs(float(control[0, index])))
        shifted[0, index] += step
        pred = model.predict_delta(
            x_current,
            shifted[:, : stim_features.shape[1]],
            shifted[:, stim_features.shape[1] :],
        )[0]
        columns.append((pred - base) / step)
    return np.column_stack(columns)


def project_target_onto_reachable_subspace(
    jacobian: np.ndarray,
    desired_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if jacobian.size == 0:
        return np.zeros_like(desired_delta), np.array([], dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(jacobian, desired_delta, rcond=None)
    return jacobian @ coeffs, coeffs


def controllable_fraction(desired_delta: np.ndarray, reachable_component: np.ndarray) -> float:
    denom = float(np.linalg.norm(desired_delta))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.linalg.norm(reachable_component) / denom, 0.0, 1.0))


def _top_control_dimensions(
    scores: np.ndarray,
    dataset: CausalDeltaDataset,
    *,
    descending: bool,
    limit: int = 10,
) -> list[dict[str, Any]]:
    names = tuple(dataset.stim_feature_names) + tuple(dataset.regime_feature_names)
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    result = []
    for index in order[:limit]:
        result.append(
            {
                "dimension": names[int(index)] if int(index) < len(names) else f"control:{index}",
                "score": float(scores[int(index)]),
            }
        )
    return result


def _group_reachability(
    spec: StateVectorSpec,
    desired: np.ndarray,
    reachable: np.ndarray,
    residual: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for group, indices in spec.feature_groups.items():
        idx = list(indices)
        desired_norm = float(np.linalg.norm(desired[idx]))
        reachable_norm = float(np.linalg.norm(reachable[idx]))
        fraction = 0.0 if desired_norm <= 1e-12 else reachable_norm / desired_norm
        result[group] = {
            "target_norm": desired_norm,
            "reachable_norm": reachable_norm,
            "unreachable_residual_norm": float(np.linalg.norm(residual[idx])),
            "controllable_fraction": float(fraction),
        }
    return result


def _recommendation(fraction: float) -> str:
    if fraction < 0.10:
        return "Outcome B risk: actuator grammar appears unable to reach the anti-trace direction."
    if fraction < 0.30:
        return "Weak reachability: expand grammar or prioritize closed-loop blocks."
    if fraction < 0.60:
        return "Proceed to nonlinear inverse optimization with held-out validation."
    return "Strong reachable-subspace signal; prioritize validation and generalization."
