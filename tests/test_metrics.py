import numpy as np

from probability_calibration_audit_lab.metrics import (
    calibration_table,
    expected_calibration_error,
    select_cost_threshold,
)


def test_equal_frequency_table_preserves_all_observations() -> None:
    outcomes = np.array([0, 1, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.7, 0.2, 0.9, 0.8, 0.3, 0.6])
    table = calibration_table(outcomes, probabilities, n_bins=3)
    assert table["count"].sum() == len(outcomes)
    assert table["mean_probability"].is_monotonic_increasing
    assert np.isclose(
        expected_calibration_error(outcomes, probabilities, n_bins=3),
        np.average(table["absolute_gap"], weights=table["count"]),
    )


def test_cost_threshold_selection_matches_simple_separable_case() -> None:
    outcomes = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, cost = select_cost_threshold(
        outcomes,
        probabilities,
        thresholds=np.array([0.15, 0.5, 0.85]),
    )
    assert threshold == 0.5
    assert cost == 0.0
