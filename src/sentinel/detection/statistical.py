"""Statistical detector — rolling Z-score on STL residuals.

Role in the ensemble (post-STL architecture):
    CUSUM / EWMA handle drift and level shifts.
    This detector handles SPIKES — sudden single-point outliers that CUSUM
    misses because it requires sustained accumulation.

Input: STL residual array (oldest → newest), NOT raw telemetry values.
    Residuals are centred near 0 (mean subtracted by calibration baseline).
    Z-score of a residual = "how many σ away from zero is this deviation?"

The detector also flags rapid rate-of-change anomalies (dv/dt) using the
same residual window.  A spike looks like a large residual; a ramp looks
like a large rate-of-change.

Constant-value guard: if residuals have std < 1e-4, the channel is
perfectly stable and the score is forced to NOMINAL.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.models import DetectorResult, Severity
from sentinel.features.engine import FeatureVector


class StatisticalDetector:
    """Z-score spike detector.  Stateless — all context comes from the window."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        severe_z_threshold: float = 5.0,
        min_window: int = 30,
    ):
        self.z_threshold = z_threshold
        self.severe_z_threshold = severe_z_threshold
        self.min_window = min_window

    def detect(
        self,
        features: FeatureVector,
        window_values: np.ndarray | None = None,
    ) -> DetectorResult:
        """Evaluate a single parameter's feature vector for spikes.

        Args:
            features:     FeatureVector computed on the STL residual window.
                          features.raw_value  = current residual
                          features.z_score    = residual / rolling_std(residuals)
            window_values: Raw residual array (optional).  Used for minimum-
                          window enforcement and rate-of-change check.

        Returns:
            DetectorResult from the "statistical" detector.
        """
        z = abs(features.z_score)

        # Constant-signal guard: perfectly flat residuals mean the STL
        # decomposition captured everything — this channel is nominal.
        if features.rolling_std < 1e-4:
            return DetectorResult(
                detector_name="statistical",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={
                    "reason": "constant_residual",
                    "rolling_std": float(features.rolling_std),
                },
            )

        # Insufficient history for a meaningful z-score.
        n = len(window_values) if window_values is not None else 0
        if n < self.min_window:
            return DetectorResult(
                detector_name="statistical",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "insufficient_data", "window_size": n},
            )

        # Score: maps z → [0, 1]
        #   z <  z_threshold : sub-threshold, low score (0 → 0.3)
        #   z >= z_threshold : anomalous, score starts at 0.5 and grows
        if z < self.z_threshold:
            score = z / self.z_threshold * 0.3
            return DetectorResult(
                detector_name="statistical",
                is_anomaly=False,
                score=float(score),
                severity=Severity.NOMINAL,
                details={
                    "z_score":   float(features.z_score),
                    "threshold": self.z_threshold,
                },
            )

        score    = min(1.0, (z - self.z_threshold) / self.z_threshold + 0.5)
        severity = self._classify_severity(z)

        # Rate-of-change anomaly: residual is jumping too fast.
        roc_flag = (
            abs(features.rate_of_change) > abs(features.rolling_std) * 3
            if features.rolling_std > 0
            else False
        )

        return DetectorResult(
            detector_name="statistical",
            is_anomaly=True,
            score=float(score),
            severity=severity,
            details={
                "z_score":              float(features.z_score),
                "threshold":            self.z_threshold,
                "rate_of_change_anomaly": roc_flag,
                "rolling_mean":         float(features.rolling_mean),
                "rolling_std":          float(features.rolling_std),
                "deviation":            float(features.deviation_from_trend),
            },
        )

    def _classify_severity(self, z: float) -> Severity:
        if z >= self.severe_z_threshold:
            return Severity.CRITICAL
        if z >= self.z_threshold * 1.5:
            return Severity.WARNING
        return Severity.WATCH

    def detect_batch(
        self,
        feature_vectors: list[FeatureVector],
        window_arrays: dict[str, np.ndarray],
    ) -> list[DetectorResult]:
        return [
            self.detect(fv, window_arrays.get(fv.parameter))
            for fv in feature_vectors
        ]
