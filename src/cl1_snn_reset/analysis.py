"""Protocol screening: multi-objective Pareto front, weighted ranking, and screening plots."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _trace_metric(df: pd.DataFrame) -> str:
    return "trace_auc_proxy" if "trace_auc_proxy" in df.columns else "trace_auc"


def pareto_mask(
    df: pd.DataFrame,
    *,
    maximize: Sequence[str] = (),
    minimize: Sequence[str] = (),
) -> np.ndarray:
    """Boolean mask of nondominated rows over the given objective columns.

    ``maximize``/``minimize`` name the objective columns; a row is kept (True)
    when no other row is at least as good on every objective and strictly better
    on one. The generic core shared by ``pareto_front`` and the experiment
    forgetting/savings fronts.
    """
    if df.empty:
        return np.zeros(0, dtype=bool)
    values = df[list(maximize) + list(minimize)].to_numpy(dtype=np.float64)
    signs = np.array([1.0] * len(maximize) + [-1.0] * len(minimize))
    score = values * signs
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        better_or_equal = np.all(score >= score[i], axis=1)
        strictly_better = np.any(score > score[i], axis=1)
        keep[i] = not bool(np.any(better_or_equal & strictly_better))
    return keep


def pareto_front(
    df: pd.DataFrame,
    *,
    maximize: tuple[str, ...] = ("weight_erasure", "health", "path_erasure"),
    minimize: tuple[str, ...] = (
        "residual_performance",
        "savings",
        "trace_auc_proxy",
        "criticality_distance",
        "energy_cost",
    ),
) -> pd.DataFrame:
    """Return nondominated protocol rows."""
    if df.empty:
        return df.copy()
    minimize = tuple(_trace_metric(df) if metric == "trace_auc_proxy" else metric for metric in minimize)
    return df.loc[pareto_mask(df, maximize=maximize, minimize=minimize)].copy()


def rank_protocols(df: pd.DataFrame) -> pd.DataFrame:
    """Scalar screen for quick inspection; Pareto front remains authoritative."""
    if df.empty:
        return df.copy()
    ranked = df.copy()
    ranked["reset_score"] = (
        1.8 * ranked["weight_erasure"]
        + 1.2 * ranked["path_erasure"]
        + 1.0 * ranked["health"]
        - 1.2 * ranked["residual_performance"]
        - 1.0 * ranked["savings"]
        - 0.8 * (ranked[_trace_metric(ranked)] - 0.5)
        - 0.4 * ranked["criticality_distance"]
        - 0.05 * ranked["energy_cost"]
    )
    return ranked.sort_values("reset_score", ascending=False).reset_index(drop=True)
