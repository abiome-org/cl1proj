from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cl1_snn_reset.config import TaskConfig
from cl1_snn_reset.electrodes import ChannelActivity
from cl1_snn_reset.metrics import activity_features, criticality, health_metrics
from cl1_snn_reset.trace_probe import trace_auc_proxy


def masked_norm(values: np.ndarray, mask: np.ndarray) -> float:
    """L2 norm of ``values`` restricted to ``mask`` (0.0 when the mask is empty)."""
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(values[mask]))


@dataclass(frozen=True)
class StateVectorSpec:
    feature_names: tuple[str, ...]
    feature_groups: dict[str, tuple[int, ...]]
    normalization: dict[str, Any]
    target_weights: dict[str, float]
    wetware_observable_mask: np.ndarray
    privileged_mask: np.ndarray
    version: str = "hybrid_v1"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "feature_groups": {
                key: list(value) for key, value in self.feature_groups.items()
            },
            "normalization": self.normalization,
            "target_weights": self.target_weights,
            "wetware_observable_mask": self.wetware_observable_mask.astype(bool).tolist(),
            "privileged_mask": self.privileged_mask.astype(bool).tolist(),
            "version": self.version,
            "spec_hash": self.spec_hash(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "StateVectorSpec":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            feature_groups={
                key: tuple(int(index) for index in value)
                for key, value in payload["feature_groups"].items()
            },
            normalization=dict(payload.get("normalization", {})),
            target_weights=dict(payload.get("target_weights", {})),
            wetware_observable_mask=np.asarray(
                payload["wetware_observable_mask"],
                dtype=bool,
            ),
            privileged_mask=np.asarray(payload["privileged_mask"], dtype=bool),
            version=str(payload.get("version", "hybrid_v1")),
        )

    def spec_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.version.encode("utf-8"))
        for name in self.feature_names:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def group_mask(self, groups: Iterable[str]) -> np.ndarray:
        mask = np.zeros(len(self.feature_names), dtype=bool)
        for group in groups:
            for index in self.feature_groups.get(group, ()):
                mask[int(index)] = True
        return mask


class StateProjector:
    def project(
        self,
        net,
        *,
        activity: ChannelActivity | None = None,
        baseline_activity: ChannelActivity | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def spec(self) -> StateVectorSpec:
        raise NotImplementedError


class HybridStateProjector(StateProjector):
    """
    Deterministic hybrid state vector.

    Observable activity and health features are separated from privileged
    hidden-weight diagnostics by masks in the spec.  The projector never edits
    weights and never requires privileged features for pulse compilation.
    """

    def __init__(
        self,
        task: TaskConfig,
        *,
        n_electrodes: int = 64,
        weight_projection_dim: int = 32,
        weight_hist_bins: int = 16,
        projector_seed: int = 1729,
        include_observable: bool = True,
        include_privileged: bool = True,
    ):
        self.task = task
        self.n_electrodes = int(n_electrodes)
        self.weight_projection_dim = int(weight_projection_dim)
        self.weight_hist_bins = int(weight_hist_bins)
        self.projector_seed = int(projector_seed)
        self.include_observable = bool(include_observable)
        self.include_privileged = bool(include_privileged)
        self._spec: StateVectorSpec | None = None
        self._projection_by_size: dict[int, np.ndarray] = {}

    @property
    def spec(self) -> StateVectorSpec:
        if self._spec is None:
            self._spec = self._build_spec()
        return self._spec

    def project(
        self,
        net,
        *,
        activity: ChannelActivity | None = None,
        baseline_activity: ChannelActivity | None = None,
    ) -> np.ndarray:
        activity = activity or _empty_activity(self.n_electrodes)
        values: list[np.ndarray] = []
        matrix = np.asarray(net.channel_connectivity_matrix(), dtype=np.float64)
        channel_scale = np.linalg.norm(matrix) + 1e-9
        values.append((matrix / channel_scale).ravel())

        task_path = np.asarray(
            [
                net.path_strength(self.task.input_channels, self.task.target_channels),
                net.path_strength(self.task.target_channels, self.task.input_channels),
                _channel_out_mean(matrix, self.task.input_channels),
                _channel_in_mean(matrix, self.task.target_channels),
            ],
            dtype=np.float64,
        )
        values.append(task_path)

        activity_vector = activity_features(
            activity,
            channel_count=self.n_electrodes,
        )
        values.append(_safe_scale(activity_vector, 50.0))

        rates = activity_vector[: self.n_electrodes]
        target_rate = float(np.mean(rates[list(self.task.target_channels)]))
        input_rate = float(np.mean(rates[list(self.task.input_channels)]))
        target_times = activity.spike_times_ms[
            np.isin(activity.channels, list(self.task.target_channels))
        ]
        latency = float(np.mean(target_times)) if target_times.size else activity.duration_ms
        trace_auc = (
            trace_auc_proxy(baseline_activity, activity)
            if baseline_activity is not None
            else 0.5
        )
        values.append(
            np.asarray(
                [
                    target_rate / 50.0,
                    input_rate / 50.0,
                    target_rate / (input_rate + 1e-6),
                    latency / max(activity.duration_ms, 1e-9),
                    trace_auc,
                ],
                dtype=np.float64,
            )
        )

        weights = np.asarray(net.weights_vector(), dtype=np.float64)
        hist, _ = np.histogram(
            weights,
            bins=self.weight_hist_bins,
            range=(float(net.cfg.w_min), float(net.cfg.w_max)),
        )
        hist = hist.astype(np.float64) / max(float(np.sum(hist)), 1.0)
        values.append(hist)
        values.append(self._weight_projection(weights))

        health = health_metrics(
            activity,
            duration_s=max(activity.duration_ms / 1000.0, 1e-9),
        )
        silent_fraction = float(np.mean(rates < 0.01))
        hyperactive_fraction = float(np.mean(rates > 80.0))
        values.append(
            np.asarray(
                [
                    health.firing_rate_hz / 50.0,
                    health.active_channel_fraction,
                    silent_fraction,
                    hyperactive_fraction,
                    health.score,
                ],
                dtype=np.float64,
            )
        )

        crit = criticality(activity)
        values.append(
            np.asarray(
                [
                    crit.avalanche_alpha,
                    crit.branching_ratio,
                    crit.mean_avalanche_size / 50.0,
                    abs(crit.branching_ratio - 1.0),
                ],
                dtype=np.float64,
            )
        )
        values.append(np.asarray([float(getattr(net, "elapsed_ms", 0.0)) / 1000.0], dtype=np.float64))

        vector = np.concatenate(values).astype(np.float64)
        if vector.size != len(self.spec.feature_names):
            raise ValueError(
                f"State vector length {vector.size} does not match spec length "
                f"{len(self.spec.feature_names)}."
            )
        return np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

    def project_with_readout(
        self,
        net,
        *,
        duration_s: float,
        baseline_activity: ChannelActivity | None = None,
    ) -> np.ndarray:
        probe = copy.deepcopy(net)
        activity = probe.advance(duration_s * 1000.0, [], plasticity=False, record=True)
        return self.project(net, activity=activity, baseline_activity=baseline_activity)

    def _weight_projection(self, weights: np.ndarray) -> np.ndarray:
        size = int(weights.size)
        if size not in self._projection_by_size:
            rng = np.random.default_rng(self.projector_seed + size)
            matrix = rng.normal(
                0.0,
                1.0 / np.sqrt(max(size, 1)),
                size=(size, self.weight_projection_dim),
            )
            self._projection_by_size[size] = matrix.astype(np.float64)
        return weights @ self._projection_by_size[size]

    def _build_spec(self) -> StateVectorSpec:
        names: list[str] = []
        groups: dict[str, tuple[int, ...]] = {}
        wetware: list[bool] = []
        privileged: list[bool] = []

        def add_group(group: str, group_names: list[str], *, observable: bool) -> None:
            start = len(names)
            names.extend(group_names)
            groups[group] = tuple(range(start, len(names)))
            wetware.extend([observable] * len(group_names))
            privileged.extend([not observable] * len(group_names))

        channels = [
            f"channel_path:{src}->{dst}"
            for src in range(self.n_electrodes)
            for dst in range(self.n_electrodes)
        ]
        add_group("channel_path", channels, observable=False)
        add_group(
            "task_path",
            [
                "task_path:input_to_target",
                "task_path:target_to_input",
                "task_path:input_out_mean",
                "task_path:target_in_mean",
            ],
            observable=False,
        )
        add_group(
            "evoked_activity",
            [f"activity:rate_ch{index}" for index in range(self.n_electrodes)]
            + [
                "activity:active_fraction",
                "activity:total_rate",
                "activity:isi_mean",
                "activity:isi_cv",
            ],
            observable=True,
        )
        add_group(
            "readout",
            [
                "readout:target_rate",
                "readout:input_rate",
                "readout:target_input_ratio",
                "readout:target_latency",
                "readout:trace_auc_proxy",
            ],
            observable=True,
        )
        add_group(
            "weight_histogram",
            [f"weight_histogram:bin{index}" for index in range(self.weight_hist_bins)],
            observable=False,
        )
        add_group(
            "privileged_weight_projection",
            [
                f"privileged_weight_projection:rp{index}"
                for index in range(self.weight_projection_dim)
            ],
            observable=False,
        )
        add_group(
            "health",
            [
                "health:firing_rate",
                "health:active_channel_fraction",
                "health:silent_channel_fraction",
                "health:hyperactive_channel_fraction",
                "health:score",
            ],
            observable=True,
        )
        add_group(
            "criticality",
            [
                "criticality:avalanche_alpha",
                "criticality:branching_ratio",
                "criticality:mean_avalanche_size",
                "criticality:distance_proxy",
            ],
            observable=True,
        )
        add_group("cost_context", ["cost_context:elapsed_s"], observable=True)

        if not self.include_observable:
            wetware = [False] * len(wetware)
        if not self.include_privileged:
            privileged = [False] * len(privileged)

        return StateVectorSpec(
            feature_names=tuple(names),
            feature_groups=groups,
            normalization={"mode": "fixed_scales", "projector_seed": self.projector_seed},
            target_weights={
                "task_path": 3.0,
                "readout": 3.0,
                "privileged_weight_projection": 2.0,
                "health": 2.0,
                "criticality": 1.0,
            },
            wetware_observable_mask=np.asarray(wetware, dtype=bool),
            privileged_mask=np.asarray(privileged, dtype=bool),
            version="hybrid_v1",
        )


class ObservableStateProjector(HybridStateProjector):
    def __init__(self, task: TaskConfig, **kwargs: Any):
        super().__init__(
            task,
            include_observable=True,
            include_privileged=False,
            **kwargs,
        )


class PrivilegedStateProjector(HybridStateProjector):
    def __init__(self, task: TaskConfig, **kwargs: Any):
        super().__init__(
            task,
            include_observable=False,
            include_privileged=True,
            **kwargs,
        )


def build_target_state(
    spec: StateVectorSpec,
    baseline_state: np.ndarray,
    trained_state: np.ndarray,
    no_reset_state: np.ndarray,
    *,
    mode: str = "trace_removed",
) -> np.ndarray:
    mode = mode.lower()
    if mode == "historical_baseline":
        return np.asarray(baseline_state, dtype=np.float64).copy()
    if mode == "age_matched_baseline":
        return np.asarray(baseline_state, dtype=np.float64).copy()
    if mode not in {"trace_removed", "hybrid"}:
        raise ValueError(f"Unknown target mode: {mode}")

    target = np.asarray(no_reset_state, dtype=np.float64).copy()
    trace_groups = ("task_path", "readout", "privileged_weight_projection")
    trace_mask = spec.group_mask(trace_groups)
    target[trace_mask] = np.asarray(baseline_state, dtype=np.float64)[trace_mask]
    health_mask = spec.group_mask(("health", "criticality"))
    target[health_mask] = np.asarray(no_reset_state, dtype=np.float64)[health_mask]
    return target


def _empty_activity(channel_count: int) -> ChannelActivity:
    return ChannelActivity(
        spike_times_ms=np.array([], dtype=np.float64),
        channels=np.array([], dtype=np.int64),
        counts=np.array([], dtype=np.int64),
        duration_ms=1000.0,
        total_neuron_spikes=0,
    )


def _channel_out_mean(matrix: np.ndarray, channels: Iterable[int]) -> float:
    selected = list(int(channel) for channel in channels)
    if not selected:
        return 0.0
    return float(np.mean(matrix[selected, :]))


def _channel_in_mean(matrix: np.ndarray, channels: Iterable[int]) -> float:
    selected = list(int(channel) for channel in channels)
    if not selected:
        return 0.0
    return float(np.mean(matrix[:, selected]))


def _safe_scale(values: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / max(float(scale), 1e-9)
