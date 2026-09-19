import numpy as np

from probability_calibration_audit_lab.simulation import simulate_binary_outcomes


def test_simulation_is_reproducible_and_probabilities_are_valid() -> None:
    first = simulate_binary_outcomes(200, seed=10)
    second = simulate_binary_outcomes(200, seed=10)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.outcomes, second.outcomes)
    assert np.array_equal(first.true_probabilities, second.true_probabilities)
    assert np.all((first.true_probabilities > 0.0) & (first.true_probabilities < 1.0))


def test_shift_changes_covariates_but_keeps_shapes() -> None:
    baseline = simulate_binary_outcomes(500, seed=12)
    shifted = simulate_binary_outcomes(500, seed=12, shift=1.0)
    assert baseline.features.shape == shifted.features.shape == (500, 5)
    assert shifted.features[:, 0].mean() > baseline.features[:, 0].mean() + 0.8
