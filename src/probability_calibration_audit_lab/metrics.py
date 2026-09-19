"""Calibration and cost-sensitive decision metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def calibration_table(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency bins; tied scores are deterministically rank-split."""
    outcomes = np.asarray(outcomes, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if outcomes.shape != probabilities.shape:
        raise ValueError("outcomes and probabilities must have matching shapes")
    if not 2 <= n_bins <= len(outcomes):
        raise ValueError("n_bins must be between 2 and the number of observations")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    order = np.argsort(probabilities, kind="mergesort")
    rows = []
    for bin_number, indices in enumerate(np.array_split(order, n_bins), start=1):
        rows.append(
            {
                "bin": bin_number,
                "count": len(indices),
                "mean_probability": float(probabilities[indices].mean()),
                "event_rate": float(outcomes[indices].mean()),
                "absolute_gap": float(
                    abs(probabilities[indices].mean() - outcomes[indices].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    table = calibration_table(outcomes, probabilities, n_bins)
    return float(np.average(table["absolute_gap"], weights=table["count"]))


def probability_metrics(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    outcomes = np.asarray(outcomes, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(outcomes, probabilities, n_bins),
    }


def decision_cost(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 4.0,
) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    predictions = np.asarray(probabilities) >= threshold
    outcomes = np.asarray(outcomes, dtype=int)
    false_positives = np.sum(predictions & (outcomes == 0))
    false_negatives = np.sum(~predictions & (outcomes == 1))
    return float(
        (false_positive_cost * false_positives + false_negative_cost * false_negatives)
        / len(outcomes)
    )


def select_cost_threshold(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 4.0,
) -> tuple[float, float]:
    candidates = (
        np.linspace(0.05, 0.95, 181) if thresholds is None else np.asarray(thresholds)
    )
    costs = np.array(
        [
            decision_cost(
                outcomes,
                probabilities,
                threshold,
                false_positive_cost,
                false_negative_cost,
            )
            for threshold in candidates
        ]
    )
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])
