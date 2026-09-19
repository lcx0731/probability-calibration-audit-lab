import numpy as np

from probability_calibration_audit_lab.calibration import IsotonicCalibrator, PlattCalibrator


def test_calibrators_produce_bounded_monotone_mappings() -> None:
    probabilities = np.linspace(0.02, 0.98, 60)
    outcomes = (probabilities + 0.15 * np.sin(np.arange(60)) > 0.55).astype(int)
    grid = np.linspace(0.01, 0.99, 101)
    for calibrator in (
        PlattCalibrator.fit(probabilities, outcomes),
        IsotonicCalibrator.fit(probabilities, outcomes),
    ):
        transformed = calibrator.predict(grid)
        assert np.all((transformed > 0.0) & (transformed < 1.0))
        assert np.all(np.diff(transformed) >= -1e-12)
