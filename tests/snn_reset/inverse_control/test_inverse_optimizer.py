import numpy as np

from cl1_snn_reset.inverse_control import (
    InverseResetObjective,
    RandomSearchOptimizer,
    StateVectorSpec,
    StimConstraints,
)


class SyntheticModel:
    def predict_delta(self, x_current, stim_features, regime_features):
        del x_current, regime_features
        y = np.zeros((stim_features.shape[0], 4))
        y[:, 0] = -stim_features[:, 13] / 10.0
        return y

    def predict_uncertainty(self, x_current, stim_features, regime_features):
        return np.zeros(stim_features.shape[0])


def test_random_optimizer_returns_valid_improving_candidates():
    spec = StateVectorSpec(
        feature_names=("task_path:a", "readout:b", "health:c", "other:d"),
        feature_groups={"task_path": (0,), "readout": (1,), "health": (2,), "criticality": (), "privileged_weight_projection": ()},
        normalization={},
        target_weights={},
        wetware_observable_mask=np.array([False, True, True, True]),
        privileged_mask=np.array([True, False, False, False]),
    )
    objective = InverseResetObjective(spec, max_energy_cost=1.0)
    optimizer = RandomSearchOptimizer(max_evaluations=20, random_seed=4)

    candidates = optimizer.propose(
        model=SyntheticModel(),
        x_current=np.zeros(4),
        no_reset_state=np.array([1.0, 0.0, 0.0, 0.0]),
        target_state=np.zeros(4),
        objective=objective,
        constraints=StimConstraints(max_energy_cost=1.0, max_total_duration_s=0.4),
        input_channel=8,
        target_channel=9,
        stim_sampling={
            "include_blocks": ["anti_stdp_pairing"],
            "duration_s": [0.2],
            "amplitude_uA": [0.8],
            "delays_ms": [2, 10],
        },
        candidate_count=3,
    )

    assert candidates
    assert all(candidate.predicted_loss >= 0.0 for candidate in candidates)
