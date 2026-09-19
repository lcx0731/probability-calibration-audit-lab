"""Post-hoc probability calibrators fitted on a held-out calibration set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def clip_probabilities(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    return np.clip(values, epsilon, 1.0 - epsilon)


def logit(probabilities: np.ndarray) -> np.ndarray:
    values = clip_probabilities(probabilities)
    return np.log(values / (1.0 - values))


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    @classmethod
    def fit(cls, probabilities: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logit(probabilities).reshape(-1, 1), outcomes)
        return cls(model)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]
        return clip_probabilities(calibrated)


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression

    @classmethod
    def fit(cls, probabilities: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(np.asarray(probabilities, dtype=float), outcomes)
        return cls(model)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict(np.asarray(probabilities, dtype=float))
        return clip_probabilities(calibrated)
