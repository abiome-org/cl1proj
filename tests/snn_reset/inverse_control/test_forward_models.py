import numpy as np

from cl1_snn_reset.inverse_control.forward_models import MeanZeroDeltaModel, RidgeDeltaModel


def test_mean_zero_model_shapes():
    y = np.ones((5, 3))
    model = MeanZeroDeltaModel()
    model.fit(np.zeros((5, 2)), np.zeros((5, 4)), np.zeros((5, 1)), y)

    assert model.predict_delta(np.zeros((2, 2)), np.zeros((2, 4)), np.zeros((2, 1))).shape == (2, 3)
    assert model.predict_uncertainty(np.zeros((2, 2)), np.zeros((2, 4)), np.zeros((2, 1))).shape == (2,)


def test_ridge_model_fits_synthetic_linear_dynamics():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 5))
    stim = rng.normal(size=(80, 4))
    regime = rng.normal(size=(80, 2))
    control = np.hstack([stim, regime])
    weights = rng.normal(size=(control.shape[1], 3))
    y = control @ weights

    model = RidgeDeltaModel(alphas=(1e-6, 1e-3), max_state_features=3)
    model.fit(x, stim, regime, y)
    pred = model.predict_delta(x, stim, regime)

    assert np.mean((pred - y) ** 2) < 1e-4
    assert model.control_jacobian(x[:1], stim[:1], regime[:1]).shape == (3, 6)
