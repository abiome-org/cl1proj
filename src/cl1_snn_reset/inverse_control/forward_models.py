from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .rollout_dataset import CausalDeltaDataset
from .state_projectors import StateVectorSpec


class ForwardDeltaModel(Protocol):
    def fit(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
        causal_delta: np.ndarray,
    ) -> None:
        ...

    def predict_delta(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        ...

    def predict_uncertainty(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModelUncertainty:
    rms: np.ndarray
    per_feature: np.ndarray | None = None


class MeanZeroDeltaModel:
    def __init__(self):
        self.output_dim = 0
        self.residual_rms = 0.0

    def fit(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
        causal_delta: np.ndarray,
    ) -> None:
        del x_current, stim_features, regime_features
        y = np.asarray(causal_delta, dtype=np.float64)
        self.output_dim = int(y.shape[1])
        self.residual_rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0

    def predict_delta(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        rows = int(np.asarray(stim_features).shape[0])
        return np.zeros((rows, self.output_dim), dtype=np.float64)

    def predict_uncertainty(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        del x_current, regime_features
        return np.full(int(np.asarray(stim_features).shape[0]), self.residual_rms, dtype=np.float64)


class RidgeDeltaModel:
    """
    State-conditioned ridge model with compact interaction features.

    The model keeps the large state vector as the output but limits input-side
    interactions to high-variance state coordinates, which makes the pure
    inverse-control search much cheaper than a blind protocol grid.
    """

    def __init__(
        self,
        *,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
        max_state_features: int = 64,
        max_interaction_state_features: int = 24,
        max_interaction_control_features: int = 24,
        random_seed: int = 123,
    ):
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.max_state_features = int(max_state_features)
        self.max_interaction_state_features = int(max_interaction_state_features)
        self.max_interaction_control_features = int(max_interaction_control_features)
        self.random_seed = int(random_seed)
        self.model: Any | None = None
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.state_indices: np.ndarray | None = None
        self.output_dim = 0
        self.best_alpha: float | None = None
        self.residual_rms = 0.0

    def fit(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
        causal_delta: np.ndarray,
    ) -> None:
        from sklearn.linear_model import Ridge

        x_current = np.asarray(x_current, dtype=np.float64)
        y = np.asarray(causal_delta, dtype=np.float64)
        self.output_dim = int(y.shape[1])
        self.state_indices = _select_state_indices(x_current, self.max_state_features)
        design = self._design_matrix(x_current, stim_features, regime_features, fit=True)
        rng = np.random.default_rng(self.random_seed)
        order = rng.permutation(design.shape[0])
        split = max(1, int(0.75 * design.shape[0]))
        train_idx = order[:split]
        val_idx = order[split:] if split < design.shape[0] else order[:split]
        best_score = float("inf")
        best_model = None
        best_alpha = self.alphas[0]
        for alpha in self.alphas:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(design[train_idx], y[train_idx])
            pred = model.predict(design[val_idx])
            score = float(np.mean(np.square(pred - y[val_idx])))
            if score < best_score:
                best_score = score
                best_model = model
                best_alpha = float(alpha)
        self.model = best_model
        self.best_alpha = best_alpha
        pred_all = self.model.predict(design)
        self.residual_rms = float(np.sqrt(np.mean(np.square(pred_all - y)))) if y.size else 0.0

    def predict_delta(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RidgeDeltaModel is not fitted.")
        design = self._design_matrix(x_current, stim_features, regime_features, fit=False)
        return np.asarray(self.model.predict(design), dtype=np.float64)

    def predict_uncertainty(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
    ) -> np.ndarray:
        del x_current, regime_features
        return np.full(int(np.asarray(stim_features).shape[0]), self.residual_rms, dtype=np.float64)

    def control_jacobian(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
        *,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        x = _as_2d(x_current)
        stim = _as_2d(stim_features)
        regime = _as_2d(regime_features)
        control = np.hstack([stim, regime])
        base = self.predict_delta(x, stim, regime)[0]
        columns: list[np.ndarray] = []
        for index in range(control.shape[1]):
            perturbed = control.copy()
            step = epsilon * max(1.0, abs(float(control[0, index])))
            perturbed[0, index] += step
            stim_part = perturbed[:, : stim.shape[1]]
            regime_part = perturbed[:, stim.shape[1] :]
            shifted = self.predict_delta(x, stim_part, regime_part)[0]
            columns.append((shifted - base) / step)
        return np.column_stack(columns)

    def _design_matrix(
        self,
        x_current: np.ndarray,
        stim_features: np.ndarray,
        regime_features: np.ndarray,
        *,
        fit: bool,
    ) -> np.ndarray:
        x = _as_2d(x_current)
        stim = _as_2d(stim_features)
        regime = _as_2d(regime_features)
        if self.state_indices is None:
            self.state_indices = _select_state_indices(x, self.max_state_features)
        control = np.hstack([stim, regime])
        selected = x[:, self.state_indices]
        interaction_state = selected[:, : self.max_interaction_state_features]
        interaction_control = control[:, : self.max_interaction_control_features]
        interactions = (
            interaction_state[:, :, None] * interaction_control[:, None, :]
        ).reshape(x.shape[0], -1)
        raw = np.hstack([control, selected, interactions])
        if fit:
            self.x_mean = raw.mean(axis=0)
            self.x_scale = raw.std(axis=0)
            self.x_scale[self.x_scale < 1e-9] = 1.0
        if self.x_mean is None or self.x_scale is None:
            raise RuntimeError("Feature scaler is not fitted.")
        return (raw - self.x_mean) / self.x_scale


def evaluate_forward_model(
    model: ForwardDeltaModel,
    dataset: CausalDeltaDataset,
    *,
    state_spec: StateVectorSpec | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import r2_score

    y_true = dataset.causal_deltas
    y_pred = model.predict_delta(
        dataset.trained_states,
        dataset.stim_features,
        dataset.regime_features,
    )
    residual = y_pred - y_true
    cosine = _row_cosine(y_pred, y_true)
    diagnostics: dict[str, Any] = {
        "mse": float(np.mean(np.square(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "target_direction_accuracy": float(np.mean(cosine > 0.0)),
        "null_effect_norm_mean": float(np.mean(np.linalg.norm(y_true, axis=1))),
    }
    try:
        raw_r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    except Exception:
        raw_r2 = np.full(y_true.shape[1], np.nan)
    diagnostics["mean_r2"] = float(np.nanmean(raw_r2))
    if state_spec is not None:
        diagnostics["r2_by_feature_group"] = {
            group: float(np.nanmean(raw_r2[list(indices)]))
            for group, indices in state_spec.feature_groups.items()
            if len(indices)
        }
    return diagnostics


def _select_state_indices(x: np.ndarray, limit: int) -> np.ndarray:
    if x.shape[1] <= limit:
        return np.arange(x.shape[1], dtype=int)
    variance = np.var(x, axis=0)
    return np.argsort(variance)[-int(limit) :].astype(int)


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return np.sum(a * b, axis=1) / denom
