"""Statistical anomaly detector — rolling Z-score + seasonal decomposition.

The simplest and most reliable detector in the ensemble. Works on day one
with zero training data. Catches the majority of telemetry anomalies:
  - Thermal drift
  - Voltage sag
  - Unexpected oscillation
  - Out-of-range values

The Z-score tells you "how unusual is this value relative to recent history."
Seasonal decomposition handles predictable orbital patterns (eclipse cycles)
so they don't trigger false positives.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.models import DetectorResult, Severity
from sentinel.features.engine import FeatureVector


class StatisticalDetector:
    """Z-score based anomaly detection with seasonal awareness."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        severe_z_threshold: float = 5.0,
        min_window: int = 30,
    ):
        self.z_threshold = z_threshold
        self.severe_z_threshold = severe_z_threshold
        self.min_window = min_window

    def detect(self, features: FeatureVector, window_values: np.ndarray | None = None) -> DetectorResult:
        """Evaluate a single parameter's feature vector for anomalies."""
        z = abs(features.z_score)

        # Not enough data to make a call
        if window_values is not None and len(window_values) < self.min_window:
            return DetectorResult(
                detector_name="statistical",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "insufficient_data", "window_size": len(window_values) if window_values is not None else 0},
            )

        # Compute anomaly score: normalized Z to [0, 1] range
        # score = 0 at z_threshold, 1 at 2x z_threshold
        if z < self.z_threshold:
            score = z / self.z_threshold * 0.3  # below threshold = low score
            return DetectorResult(
                detector_name="statistical",
                is_anomaly=False,
                score=float(score),
                severity=Severity.NOMINAL,
                details={"z_score": float(features.z_score), "threshold": self.z_threshold},
            )

        # Above threshold — it's anomalous
        score = min(1.0, (z - self.z_threshold) / self.z_threshold + 0.5)
        severity = self._classify_severity(z, features)

        # Additional checks for rate-of-change anomalies
        roc_flag = abs(features.rate_of_change) > abs(features.rolling_std) * 3 if features.rolling_std > 0 else False

        return DetectorResult(
            detector_name="statistical",
            is_anomaly=True,
            score=float(score),
            severity=severity,
            details={
                "z_score": float(features.z_score),
                "threshold": self.z_threshold,
                "rate_of_change_anomaly": roc_flag,
                "rolling_mean": float(features.rolling_mean),
                "rolling_std": float(features.rolling_std),
                "deviation": float(features.deviation_from_trend),
            },
        )

    def _classify_severity(self, z: float, features: FeatureVector) -> Severity:
        """Map Z-score magnitude to operational severity."""
        if z >= self.severe_z_threshold:
            return Severity.CRITICAL
        if z >= self.z_threshold * 1.5:
            return Severity.WARNING
        return Severity.WATCH

    def detect_batch(
        self, feature_vectors: list[FeatureVector], window_arrays: dict[str, np.ndarray]
    ) -> list[DetectorResult]:
        """Run detection on multiple parameters at once."""
        return [
            self.detect(fv, window_arrays.get(fv.parameter))
            for fv in feature_vectors
        ]
