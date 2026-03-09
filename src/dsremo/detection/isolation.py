"""Isolation Forest detector — multivariate anomaly detection.

This is the most powerful detector in Phase 1. It operates on the full
feature vector across multiple parameters simultaneously, catching anomalies
that only show up in parameter *relationships*.

Example: battery_voltage and solar_current normally correlate. If voltage
drops but current stays high, a univariate detector sees nothing wrong.
Isolation Forest catches the broken correlation.

The model is periodically re-fit on recent normal data (unsupervised).
No labeled anomaly data needed.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from dsremo.core.models import DetectorResult, Severity

logger = structlog.get_logger()


class IsolationForestDetector:
    """Multivariate anomaly detection using Isolation Forest."""

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 100,
        min_training_samples: int = 200,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.min_training_samples = min_training_samples

        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._fit_count = 0

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    def fit(self, training_data: np.ndarray, feature_names: list[str]) -> None:
        """Fit the model on recent normal telemetry data.

        training_data: shape (n_samples, n_features)
        feature_names: list of parameter names matching the columns
        """
        if len(training_data) < self.min_training_samples:
            logger.warning(
                "isolation_forest_skip_fit",
                reason="insufficient_data",
                samples=len(training_data),
                required=self.min_training_samples,
            )
            return

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=1,  # single-threaded to avoid fork issues in async
        )
        self._model.fit(training_data)
        self._is_fitted = True
        self._feature_names = feature_names
        self._fit_count += 1

        logger.info(
            "isolation_forest_fitted",
            samples=len(training_data),
            features=len(feature_names),
            fit_count=self._fit_count,
        )

    def detect(self, feature_vector: np.ndarray) -> DetectorResult:
        """Score a single multivariate observation.

        feature_vector: shape (n_features,) — one value per parameter
        """
        if not self._is_fitted or self._model is None:
            return DetectorResult(
                detector_name="isolation_forest",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "model_not_fitted"},
            )

        sample = feature_vector.reshape(1, -1)

        # Isolation Forest anomaly score: negative = more anomalous
        raw_score = self._model.decision_function(sample)[0]
        prediction = self._model.predict(sample)[0]  # 1 = normal, -1 = anomaly

        # Normalize to [0, 1] where 1 = highly anomalous
        # decision_function returns ~[-0.5, 0.5], with negative = anomaly
        normalized_score = float(np.clip(-raw_score + 0.5, 0, 1))

        is_anomaly = prediction == -1

        # Identify which features contributed most to the anomaly
        contributions = self._compute_feature_contributions(feature_vector)

        severity = Severity.NOMINAL
        if is_anomaly:
            if normalized_score > 0.85:
                severity = Severity.CRITICAL
            elif normalized_score > 0.65:
                severity = Severity.WARNING
            else:
                severity = Severity.WATCH

        return DetectorResult(
            detector_name="isolation_forest",
            is_anomaly=is_anomaly,
            score=normalized_score,
            severity=severity,
            details={
                "raw_score": float(raw_score),
                "feature_contributions": contributions,
                "fit_count": self._fit_count,
            },
        )

    def _compute_feature_contributions(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Estimate which features contributed most to the anomaly score.

        Uses a simple perturbation-based approach: mask each feature to its
        mean value and measure score change. Not as rigorous as SHAP but
        fast enough for real-time use.
        """
        if not self._is_fitted or self._model is None:
            return {}

        base_score = self._model.decision_function(feature_vector.reshape(1, -1))[0]
        contributions = {}

        for i, name in enumerate(self._feature_names):
            perturbed = feature_vector.copy()
            perturbed[i] = 0.0  # set to zero (approximate mean for standardized data)
            perturbed_score = self._model.decision_function(perturbed.reshape(1, -1))[0]
            # Positive delta = this feature made it more anomalous
            contributions[name] = float(base_score - perturbed_score)

        return contributions

    def needs_refit(self, samples_since_fit: int, refit_interval: int = 1000) -> bool:
        """Check if the model should be retrained on newer data."""
        if not self._is_fitted:
            return True
        return samples_since_fit >= refit_interval
