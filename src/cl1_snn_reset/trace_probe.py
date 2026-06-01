from __future__ import annotations

import numpy as np

from .electrodes import ChannelActivity
from .metrics import activity_features


def trace_auc_proxy(naive: ChannelActivity, post: ChannelActivity) -> float:
    """
    Single-trial trace detectability proxy on channel-level readouts.

    Full held-out classifier AUC needs several seeds.  This maps standardized
    feature distance into [0.5, 1.0] so individual rows still expose a trace
    detectability signal.
    """
    a = activity_features(naive)
    b = activity_features(post)
    distance = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-9))
    return float(0.5 + 0.5 * (1.0 - np.exp(-3.0 * distance)))


def trace_probe_auc(
    naive_activities: list[ChannelActivity],
    post_activities: list[ChannelActivity],
    *,
    random_state: int = 1,
) -> float:
    """Train a channel-readout classifier for naive vs post-reset traces."""
    if len(naive_activities) < 2 or len(post_activities) < 2:
        if naive_activities and post_activities:
            return trace_auc_proxy(naive_activities[0], post_activities[0])
        return 0.5
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        proxies = [
            trace_auc_proxy(naive, post)
            for naive, post in zip(naive_activities, post_activities)
        ]
        return float(np.mean(proxies)) if proxies else 0.5

    X = np.vstack(
        [activity_features(item) for item in naive_activities]
        + [activity_features(item) for item in post_activities]
    )
    y = np.array([0] * len(naive_activities) + [1] * len(post_activities), dtype=np.int64)
    splits = min(5, np.bincount(y).min())
    if splits < 2:
        return 0.5
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=random_state),
    )
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, scores))
