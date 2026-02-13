"""Detection pipeline orchestrator — the brain of Sentinel.

Coordinates all detectors, combines their verdicts via weighted ensemble,
triggers explanations, and stores confirmed anomalies.

This is the module that turns Sentinel from "a collection of ML models"
into "an intelligent anomaly detection engine."
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import structlog

from sentinel.core.models import Anomaly, DetectorResult, Severity
from sentinel.db import queries
from sentinel.detection.changepoint import ChangePointDetector
from sentinel.detection.isolation import IsolationForestDetector
from sentinel.detection.statistical import StatisticalDetector
from sentinel.features.engine import FeatureEngine

logger = structlog.get_logger()

# Singleton instances — initialized once, reused across detection cycles
_feature_engine = FeatureEngine(window_size=600)
_stat_detector = StatisticalDetector(z_threshold=3.0, severe_z_threshold=5.0)
_iso_detector = IsolationForestDetector(contamination=0.01)
_cp_detector = ChangePointDetector(penalty=3.0, min_segment_size=30)

# Track samples since last Isolation Forest refit
_samples_since_fit: dict[str, int] = {}

# All parameters we track for multivariate detection
ALL_PARAMETERS = [
    "battery_voltage", "battery_current", "solar_array_current", "bus_voltage",
    "wheel_speed_x", "wheel_speed_y", "wheel_speed_z", "pointing_error",
    "panel_temp_sun", "panel_temp_shade", "battery_temp", "electronics_temp",
    "signal_strength", "bit_error_rate", "link_margin",
]

# Ensemble weights (must sum to 1.0)
WEIGHTS = {
    "statistical": 0.35,
    "isolation_forest": 0.40,
    "changepoint": 0.25,
}


async def run_detection_cycle(satellite_id: str) -> list[Anomaly]:
    """Run a full detection cycle for a satellite.

    Called after new telemetry is ingested. Fetches recent data,
    computes features, runs all detectors, and stores anomalies.
    """
    start = time.monotonic()
    anomalies: list[Anomaly] = []

    # Fetch recent telemetry for all parameters
    latest = await queries.get_latest_values(satellite_id)
    if not latest:
        return anomalies

    now_epoch = datetime.now(timezone.utc).timestamp()

    # Process each parameter through the feature engine
    for row in latest:
        param = row["parameter"]
        value = row["value"]
        ts = row["timestamp"].timestamp() if hasattr(row["timestamp"], "timestamp") else float(row["timestamp"])

        features = _feature_engine.compute(param, value, ts)

        # --- Statistical Detection ---
        window_rows = await queries.get_recent_telemetry_window(satellite_id, param, 300)
        window_values = np.array([r["value"] for r in window_rows], dtype=np.float64) if window_rows else None
        stat_result = _stat_detector.detect(features, window_values)

        # --- Change-Point Detection ---
        cp_result = _cp_detector.detect(window_values, param) if window_values is not None and len(window_values) >= 60 else DetectorResult(
            detector_name="changepoint", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "insufficient_data"},
        )

        # --- Isolation Forest (multivariate) ---
        iso_result = _detect_isolation_forest(satellite_id)

        # --- Ensemble Verdict ---
        results = [stat_result, iso_result, cp_result]
        is_anomaly, confidence, severity = _ensemble_vote(results)

        if is_anomaly:
            explanation = _build_explanation(param, features, results, row)
            contributing = _extract_contributions(results)

            anomaly = Anomaly(
                satellite_id=satellite_id,
                timestamp=row["timestamp"],
                subsystem=row.get("subsystem", ""),
                parameter=param,
                value=value,
                severity=severity,
                confidence=confidence,
                detectors_triggered=tuple(r.detector_name for r in results if r.is_anomaly),
                explanation=explanation,
                contributing_params=contributing,
            )

            try:
                await queries.insert_anomaly(anomaly)
                anomalies.append(anomaly)

                # Broadcast to WebSocket clients
                from sentinel.api.websocket import broadcast_anomaly
                await broadcast_anomaly({
                    "id": anomaly.id,
                    "satellite_id": anomaly.satellite_id,
                    "parameter": anomaly.parameter,
                    "value": anomaly.value,
                    "severity": anomaly.severity.value,
                    "confidence": anomaly.confidence,
                    "explanation": anomaly.explanation,
                    "timestamp": anomaly.timestamp.isoformat() if isinstance(anomaly.timestamp, datetime) else str(anomaly.timestamp),
                })
            except Exception as e:
                logger.error("anomaly_store_failed", error=str(e), parameter=param)

    # Periodically refit Isolation Forest
    sat_count = _samples_since_fit.get(satellite_id, 0) + len(latest)
    _samples_since_fit[satellite_id] = sat_count
    if _iso_detector.needs_refit(sat_count):
        await _refit_isolation_forest(satellite_id)
        _samples_since_fit[satellite_id] = 0

    elapsed_ms = (time.monotonic() - start) * 1000
    if anomalies:
        logger.info(
            "detection_cycle_complete",
            satellite=satellite_id,
            anomalies_found=len(anomalies),
            elapsed_ms=round(elapsed_ms, 1),
        )

    return anomalies


def _detect_isolation_forest(satellite_id: str) -> DetectorResult:
    """Run Isolation Forest on the latest multivariate snapshot."""
    if not _iso_detector.is_ready:
        return DetectorResult(
            detector_name="isolation_forest",
            is_anomaly=False,
            score=0.0,
            severity=Severity.NOMINAL,
            details={"reason": "model_not_fitted"},
        )

    snapshot = _feature_engine.get_multivariate_snapshot(ALL_PARAMETERS)
    if snapshot is None:
        return DetectorResult(
            detector_name="isolation_forest",
            is_anomaly=False,
            score=0.0,
            severity=Severity.NOMINAL,
            details={"reason": "incomplete_data"},
        )

    return _iso_detector.detect(snapshot)


async def _refit_isolation_forest(satellite_id: str) -> None:
    """Retrain Isolation Forest on recent normal data."""
    matrix = _feature_engine.get_window_matrix(ALL_PARAMETERS, length=200)
    if matrix is not None:
        _iso_detector.fit(matrix, ALL_PARAMETERS)


def _ensemble_vote(
    results: list[DetectorResult],
) -> tuple[bool, float, Severity]:
    """Combine detector outputs into a single verdict.

    Uses weighted voting for confidence, takes max severity
    from any detector that flagged anomaly.
    """
    any_anomaly = any(r.is_anomaly for r in results)

    if not any_anomaly:
        avg_score = sum(r.score * WEIGHTS.get(r.detector_name, 0.33) for r in results)
        return False, float(avg_score), Severity.NOMINAL

    # Weighted confidence score
    weighted_score = sum(
        r.score * WEIGHTS.get(r.detector_name, 0.33)
        for r in results
    )
    confidence = min(1.0, weighted_score)

    # Severity: take the max from triggered detectors
    triggered = [r for r in results if r.is_anomaly]
    severity_order = {Severity.NOMINAL: 0, Severity.WATCH: 1, Severity.WARNING: 2, Severity.CRITICAL: 3}
    max_severity = max(triggered, key=lambda r: severity_order.get(r.severity, 0)).severity

    # Boost severity if multiple detectors agree
    if len(triggered) >= 2 and max_severity == Severity.WATCH:
        max_severity = Severity.WARNING
    if len(triggered) == 3 and max_severity == Severity.WARNING:
        max_severity = Severity.CRITICAL

    return True, confidence, max_severity


def _build_explanation(
    parameter: str,
    features,
    results: list[DetectorResult],
    row: dict,
) -> str:
    """Generate a human-readable explanation of why this is anomalous.

    This is what operators see. It must be clear, specific, and actionable.
    """
    triggered = [r for r in results if r.is_anomaly]
    parts = []

    parts.append(
        f"{parameter} = {row['value']:.4f} {row.get('unit', '')} "
        f"(rolling avg: {features.rolling_mean:.4f}, std: {features.rolling_std:.4f})"
    )

    for r in triggered:
        match r.detector_name:
            case "statistical":
                z = r.details.get("z_score", 0)
                parts.append(f"Z-score: {z:.2f} (threshold: {r.details.get('threshold', 3.0)})")
                if r.details.get("rate_of_change_anomaly"):
                    parts.append("Rapid rate of change detected")
            case "isolation_forest":
                contribs = r.details.get("feature_contributions", {})
                if contribs:
                    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    top_str = ", ".join(f"{k}: {v:+.3f}" for k, v in top)
                    parts.append(f"Cross-parameter anomaly. Top contributors: {top_str}")
            case "changepoint":
                cps = r.details.get("change_points", [])
                if cps:
                    recent = cps[-1]
                    parts.append(
                        f"Behavioral change detected "
                        f"(mean shift: {recent.get('mean_shift', 0):.4f})"
                    )

    agreement = f"{len(triggered)}/{len(results)} detectors triggered"
    parts.append(agreement)

    return " | ".join(parts)


def _extract_contributions(results: list[DetectorResult]) -> dict[str, float]:
    """Extract feature contributions from Isolation Forest for the anomaly record."""
    for r in results:
        if r.detector_name == "isolation_forest" and r.details.get("feature_contributions"):
            return r.details["feature_contributions"]
    return {}
