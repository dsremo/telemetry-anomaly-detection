"""Telemetry adapter — the boundary between outside world and our domain.

Validates, sanitizes, normalizes, and converts raw JSON payloads into
domain TelemetryPoint objects. This is the ONLY entry point for telemetry.
Anything that gets past here is trusted internal data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from sentinel.core.models import TelemetryPoint
from sentinel.core.security import sanitize_identifier

logger = structlog.get_logger()

# Allowed subsystem values — reject anything else
_VALID_SUBSYSTEMS = frozenset({"eps", "adcs", "thermal", "comms"})

# Maximum batch size to prevent memory abuse
MAX_BATCH_SIZE = 500


class AdapterError(Exception):
    """Raised when telemetry fails validation."""


def adapt_single(raw: dict) -> TelemetryPoint:
    """Convert a single raw JSON dict into a validated TelemetryPoint.

    Raises AdapterError if the input is malformed or suspicious.
    """
    _require_fields(raw, ("satellite_id", "timestamp", "subsystem", "parameter", "value"))

    satellite_id = sanitize_identifier(str(raw["satellite_id"]))
    if not satellite_id:
        raise AdapterError("satellite_id is empty after sanitization")

    subsystem = str(raw["subsystem"]).lower().strip()
    if subsystem not in _VALID_SUBSYSTEMS:
        raise AdapterError(f"invalid subsystem: {subsystem!r} — must be one of {_VALID_SUBSYSTEMS}")

    parameter = sanitize_identifier(str(raw["parameter"]))
    if not parameter:
        raise AdapterError("parameter is empty after sanitization")

    try:
        value = float(raw["value"])
    except (TypeError, ValueError) as e:
        raise AdapterError(f"value must be numeric, got {raw['value']!r}") from e

    if not _is_finite(value):
        raise AdapterError(f"value must be finite, got {value}")

    timestamp = _parse_timestamp(raw["timestamp"])
    unit = str(raw.get("unit", "")).strip()[:16]
    quality = _clamp(float(raw.get("quality", 1.0)), 0.0, 1.0)

    return TelemetryPoint(
        satellite_id=satellite_id,
        timestamp=timestamp,
        subsystem=subsystem,
        parameter=parameter,
        value=value,
        unit=unit,
        quality=quality,
    )


def adapt_batch(raw_points: list[dict]) -> tuple[list[TelemetryPoint], list[dict]]:
    """Convert a batch of raw dicts. Returns (valid_points, errors).

    Partial success: valid points are accepted, bad ones returned as errors.
    This prevents a single malformed point from killing an entire batch.
    """
    if len(raw_points) > MAX_BATCH_SIZE:
        raise AdapterError(f"batch too large: {len(raw_points)} points (max {MAX_BATCH_SIZE})")

    valid: list[TelemetryPoint] = []
    errors: list[dict] = []

    for i, raw in enumerate(raw_points):
        try:
            point = adapt_single(raw)
            valid.append(point)
        except AdapterError as e:
            errors.append({"index": i, "error": str(e), "input": _safe_repr(raw)})
            logger.warning("telemetry_rejected", index=i, reason=str(e))

    return valid, errors


def _require_fields(data: dict, fields: tuple[str, ...]) -> None:
    """Check that all required fields are present."""
    missing = [f for f in fields if f not in data]
    if missing:
        raise AdapterError(f"missing required fields: {missing}")


def _parse_timestamp(value) -> datetime:
    """Parse a timestamp from ISO format string or unix epoch."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError as e:
            raise AdapterError(f"invalid timestamp format: {value!r}") from e

    raise AdapterError(f"timestamp must be ISO string or unix epoch, got {type(value).__name__}")


def _is_finite(value: float) -> bool:
    """Reject inf and nan — these break every downstream computation."""
    import math
    return math.isfinite(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_repr(obj: dict) -> dict:
    """Truncate field values for safe logging — no sensitive data leaks."""
    return {k: str(v)[:100] for k, v in obj.items()}
