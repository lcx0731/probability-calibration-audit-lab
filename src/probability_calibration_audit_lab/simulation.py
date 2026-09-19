"""Deterministic synthetic binary-outcome data with a known probability model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulatedSample:
    features: np.ndarray
    outcomes: np.ndarray
    true_probabilities: np.ndarray


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def simulate_binary_outcomes(
    n_samples: int,
    seed: int,
    shift: float = 0.0,
) -> SimulatedSample:
    """Generate nonlinear outcomes; ``shift`` changes covariates, not P(Y|X)."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    if shift < 0.0:
        raise ValueError("shift must be non-negative")

    rng = np.random.default_rng(seed)
    correlation = np.array(
        [
            [1.00, 0.35, 0.10, 0.00, 0.15],
            [0.35, 1.00, 0.20, 0.05, 0.00],
            [0.10, 0.20, 1.00, 0.25, 0.10],
            [0.00, 0.05, 0.25, 1.00, 0.30],
            [0.15, 0.00, 0.10, 0.30, 1.00],
        ]
    )
    features = rng.multivariate_normal(np.zeros(5), correlation, size=n_samples)
    if shift:
        features[:, 0] += 0.85 * shift
        features[:, 2] = (1.0 + 0.30 * shift) * features[:, 2] - 0.50 * shift
        features[:, 4] += 0.35 * shift

    linear_predictor = (
        -1.15
        + 1.10 * features[:, 0]
        - 0.85 * features[:, 1]
        + 0.65 * np.sin(features[:, 2])
        + 0.45 * features[:, 0] * features[:, 1]
        - 0.30 * features[:, 3] ** 2
        + 0.35 * features[:, 4]
    )
    true_probabilities = sigmoid(linear_predictor)
    outcomes = rng.binomial(1, true_probabilities).astype(int)
    return SimulatedSample(features, outcomes, true_probabilities)
