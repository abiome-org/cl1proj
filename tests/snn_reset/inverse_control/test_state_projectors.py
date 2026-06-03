import copy

import numpy as np

from cl1_snn_reset import CultureConfig, TaskConfig, build_network
from cl1_snn_reset.inverse_control import HybridStateProjector, build_target_state


def small_culture():
    return CultureConfig(
        n_neurons=96,
        mean_out_degree=8,
        max_out_degree=12,
        local_candidate_multiplier=2,
        build_workers=1,
        spontaneous_rate_hz=0.0,
        backend="numpy",
    )


def test_hybrid_state_projector_is_deterministic_and_masked():
    task = TaskConfig(input_channels=(8,), target_channels=(9,))
    net = build_network(small_culture(), seed=3)
    projector = HybridStateProjector(task, weight_projection_dim=6, weight_hist_bins=6)
    activity = copy.deepcopy(net).advance(50.0, [], plasticity=False, record=True)

    first = projector.project(net, activity=activity)
    second = projector.project(net, activity=activity)

    assert np.allclose(first, second)
    assert len(projector.spec.feature_names) == first.size
    assert projector.spec.wetware_observable_mask.shape == (first.size,)
    assert projector.spec.privileged_mask.shape == (first.size,)
    assert np.any(projector.spec.wetware_observable_mask)
    assert np.any(projector.spec.privileged_mask)


def test_trace_removed_target_preserves_no_reset_health():
    task = TaskConfig(input_channels=(8,), target_channels=(9,))
    net = build_network(small_culture(), seed=4)
    projector = HybridStateProjector(task, weight_projection_dim=4, weight_hist_bins=4)
    state = projector.project(net)
    baseline = state.copy()
    trained = state + 1.0
    no_reset = state + 2.0

    target = build_target_state(projector.spec, baseline, trained, no_reset)
    task_mask = projector.spec.group_mask(("task_path", "readout", "privileged_weight_projection"))
    health_mask = projector.spec.group_mask(("health", "criticality"))

    assert np.allclose(target[task_mask], baseline[task_mask])
    assert np.allclose(target[health_mask], no_reset[health_mask])
