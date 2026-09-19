"""End-to-end calibration audit on independent synthetic data splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .calibration import IsotonicCalibrator, PlattCalibrator, clip_probabilities
from .metrics import (
    calibration_table,
    decision_cost,
    probability_metrics,
    select_cost_threshold,
)
from .simulation import SimulatedSample, simulate_binary_outcomes


@dataclass(frozen=True)
class AuditResult:
    metrics: pd.DataFrame
    calibration_curves: pd.DataFrame
    decisions: pd.DataFrame
    bootstrap: pd.DataFrame
    summary: dict[str, object]


def _fit_predictions(
    train: SimulatedSample,
    calibration: SimulatedSample,
    evaluation_samples: dict[str, SimulatedSample],
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=1.0,
        random_state=seed,
    )
    classifier.fit(train.features, train.outcomes)
    raw_calibration = clip_probabilities(
        classifier.predict_proba(calibration.features)[:, 1]
    )
    platt = PlattCalibrator.fit(raw_calibration, calibration.outcomes)
    isotonic = IsotonicCalibrator.fit(raw_calibration, calibration.outcomes)

    calibration_predictions = {
        "raw": raw_calibration,
        "platt": platt.predict(raw_calibration),
        "isotonic": isotonic.predict(raw_calibration),
        "oracle": calibration.true_probabilities,
    }
    evaluation_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split, sample in evaluation_samples.items():
        raw = clip_probabilities(classifier.predict_proba(sample.features)[:, 1])
        evaluation_predictions[split] = {
            "raw": raw,
            "platt": platt.predict(raw),
            "isotonic": isotonic.predict(raw),
            "oracle": sample.true_probabilities,
        }
    return evaluation_predictions, calibration_predictions


def _bootstrap_metrics(
    samples: dict[str, SimulatedSample],
    predictions: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for split, sample in samples.items():
        for repetition in range(n_bootstrap):
            indices = rng.integers(0, len(sample.outcomes), len(sample.outcomes))
            outcomes = sample.outcomes[indices]
            for method, values in predictions[split].items():
                scores = probability_metrics(outcomes, values[indices])
                rows.append(
                    {
                        "split": split,
                        "repetition": repetition,
                        "method": method,
                        "brier_score": scores["brier_score"],
                        "ece": scores["ece"],
                    }
                )
    return pd.DataFrame(rows)


def run_audit(
    n_train: int = 4000,
    n_calibration: int = 2200,
    n_test: int = 3000,
    n_bootstrap: int = 200,
    shift: float = 1.0,
    seed: int = 20260919,
) -> AuditResult:
    """Fit once, calibrate on held-out data, then audit IID and shifted tests."""
    train = simulate_binary_outcomes(n_train, seed)
    calibration = simulate_binary_outcomes(n_calibration, seed + 1)
    samples = {
        "iid_test": simulate_binary_outcomes(n_test, seed + 2),
        "shifted_test": simulate_binary_outcomes(n_test, seed + 3, shift=shift),
    }
    predictions, calibration_predictions = _fit_predictions(
        train, calibration, samples, seed
    )

    selected_thresholds = {
        method: select_cost_threshold(calibration.outcomes, values)[0]
        for method, values in calibration_predictions.items()
    }
    metric_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    for split, sample in samples.items():
        for method, values in predictions[split].items():
            scores = probability_metrics(sample.outcomes, values)
            metric_rows.append(
                {
                    "split": split,
                    "method": method,
                    **scores,
                    "roc_auc": float(roc_auc_score(sample.outcomes, values)),
                    "mean_probability": float(values.mean()),
                    "event_rate": float(sample.outcomes.mean()),
                }
            )
            table = calibration_table(sample.outcomes, values)
            table.insert(0, "method", method)
            table.insert(0, "split", split)
            curve_frames.append(table)
            threshold = selected_thresholds[method]
            decision_rows.append(
                {
                    "split": split,
                    "method": method,
                    "selected_threshold": threshold,
                    "test_cost_per_observation": decision_cost(
                        sample.outcomes, values, threshold
                    ),
                    "positive_decision_rate": float(np.mean(values >= threshold)),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    bootstrap = _bootstrap_metrics(samples, predictions, n_bootstrap, seed + 4)
    intervals = (
        bootstrap.groupby(["split", "method"])[["brier_score", "ece"]]
        .quantile([0.025, 0.975])
        .unstack()
    )
    interval_rows: list[dict[str, object]] = []
    for (split, method), row in intervals.iterrows():
        interval_rows.append(
            {
                "split": split,
                "method": method,
                "brier_ci_95": [
                    float(row[("brier_score", 0.025)]),
                    float(row[("brier_score", 0.975)]),
                ],
                "ece_ci_95": [
                    float(row[("ece", 0.025)]),
                    float(row[("ece", 0.975)]),
                ],
            }
        )
    summary: dict[str, object] = {
        "configuration": {
            "n_train": n_train,
            "n_calibration": n_calibration,
            "n_test_per_split": n_test,
            "n_bootstrap": n_bootstrap,
            "covariate_shift_strength": shift,
            "false_positive_cost": 1.0,
            "false_negative_cost": 4.0,
            "seed": seed,
        },
        "selected_thresholds": selected_thresholds,
        "bootstrap_intervals": interval_rows,
    }
    return AuditResult(metrics, curves, decisions, bootstrap, summary)
