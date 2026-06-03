import numpy as np

from cl1_snn_reset.inverse_control.controllability import (
    controllable_fraction,
    project_target_onto_reachable_subspace,
)


def test_projection_recovers_known_reachable_subspace():
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    desired = np.array([2.0, -1.0, 3.0])

    reachable, coeffs = project_target_onto_reachable_subspace(jacobian, desired)

    assert np.allclose(coeffs, [2.0, -1.0])
    assert np.allclose(reachable, [2.0, -1.0, 0.0])
    assert 0.0 <= controllable_fraction(desired, reachable) <= 1.0
