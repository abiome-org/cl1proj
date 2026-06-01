from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(
    df: pd.DataFrame,
    *,
    maximize: tuple[str, ...] = ("weight_erasure", "health", "path_erasure"),
    minimize: tuple[str, ...] = (
        "residual_performance",
        "savings",
        "trace_auc",
        "criticality_distance",
        "energy_cost",
    ),
) -> pd.DataFrame:
    """Return nondominated protocol rows."""
    if df.empty:
        return df.copy()
    values = df[list(maximize + minimize)].to_numpy(dtype=np.float64)
    signs = np.array([1.0] * len(maximize) + [-1.0] * len(minimize))
    score = values * signs
    dominated = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if dominated[i]:
            continue
        better_or_equal = np.all(score >= score[i], axis=1)
        strictly_better = np.any(score > score[i], axis=1)
        dominated[i] = bool(np.any(better_or_equal & strictly_better))
    return df.loc[~dominated].copy()


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
        - 0.8 * (ranked["trace_auc"] - 0.5)
        - 0.4 * ranked["criticality_distance"]
        - 0.05 * ranked["energy_cost"]
    )
    return ranked.sort_values("reset_score", ascending=False).reset_index(drop=True)


def plot_protocol_scatter(
    df: pd.DataFrame,
    *,
    x: str = "weight_erasure",
    y: str = "residual_performance",
    hue: str = "beta",
):
    """Create a quick protocol metric scatter plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style="schedule", ax=ax)
    ax.set_title("SNN Reset Protocol Screen")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_pareto_summary(df: pd.DataFrame):
    """Plot the nondominated front over weight erasure and trace detectability."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    front = pareto_front(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="weight_erasure", y="trace_auc", color="0.7", ax=ax, label="screened")
    if not front.empty:
        sns.scatterplot(
            data=front,
            x="weight_erasure",
            y="trace_auc",
            hue="health",
            size="energy_cost",
            ax=ax,
            label="pareto",
        )
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.4)
    ax.set_title("Pareto Reset Candidates")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax
