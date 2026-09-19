import numpy as np

from probability_calibration_audit_lab.evaluation import run_audit


def test_end_to_end_audit_has_complete_finite_outputs() -> None:
    result = run_audit(
        n_train=800,
        n_calibration=500,
        n_test=600,
        n_bootstrap=20,
        seed=77,
    )
    expected_methods = {"raw", "platt", "isotonic", "oracle"}
    assert set(result.metrics["method"]) == expected_methods
    assert set(result.metrics["split"]) == {"iid_test", "shifted_test"}
    assert len(result.metrics) == 8
    numeric = result.metrics.select_dtypes(include="number")
    assert np.isfinite(numeric).all().all()
    assert len(result.bootstrap) == 2 * 20 * len(expected_methods)
    assert all(0.0 < value < 1.0 for value in result.summary["selected_thresholds"].values())
