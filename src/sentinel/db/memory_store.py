"""In-memory store — drop-in replacement for DB queries when running without PostgreSQL.

Used in demo mode (`sentinel serve --demo`). All data lives in memory and is
lost when the server stops. This lets investors see the full pipeline working
on any laptop without Docker or PostgreSQL.

Implements the same async interface as sentinel.db.queries so the rest of the
codebase doesn't need to know the difference.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from threading import Lock

import structlog

from sentinel.core.models import Anomaly, TelemetryPoint

logger = structlog.get_logger()

# --- In-memory storage ---
_lock = Lock()
_telemetry: list[dict] = []
_anomalies: list[dict] = []
_api_keys: list[dict] = []
_MAX_TELEMETRY = 100_000  # cap to prevent OOM in long demos


def _point_to_dict(p: TelemetryPoint) -> dict:
    return {
        "satellite_id": p.satellite_id,
        "timestamp": p.timestamp,
        "subsystem": p.subsystem,
        "parameter": p.parameter,
        "value": p.value,
        "unit": p.unit,
        "quality": p.quality,
    }


def _anomaly_to_dict(a: Anomaly) -> dict:
    return {
        "id": a.id,
        "satellite_id": a.satellite_id,
        "timestamp": a.timestamp,
        "subsystem": a.subsystem,
        "parameter": a.parameter,
        "value": a.value,
        "severity": a.severity.value,
        "confidence": a.confidence,
        "detectors_triggered": list(a.detectors_triggered),
        "explanation": a.explanation,
        "root_cause_group": a.root_cause_group,
        "contributing_params": json.dumps(a.contributing_params),
    }


# ---------------------------------------------------------------------------
# Telemetry (same signatures as queries.py)
# ---------------------------------------------------------------------------

async def insert_telemetry(points: list[TelemetryPoint]) -> int:
    if not points:
        return 0
    with _lock:
        for p in points:
            _telemetry.append(_point_to_dict(p))
        # Trim oldest if over cap
        if len(_telemetry) > _MAX_TELEMETRY:
            del _telemetry[: len(_telemetry) - _MAX_TELEMETRY]
    return len(points)


async def get_telemetry(
    satellite_id: str,
    parameter: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 1000,
) -> list[dict]:
    with _lock:
        results = [r for r in _telemetry if r["satellite_id"] == satellite_id]
    if parameter:
        results = [r for r in results if r["parameter"] == parameter]
    if since:
        results = [r for r in results if r["timestamp"] >= since]
    if until:
        results = [r for r in results if r["timestamp"] <= until]
    results.sort(key=lambda r: r["timestamp"], reverse=True)
    return results[:min(limit, 10_000)]


async def get_recent_telemetry_window(
    satellite_id: str,
    parameter: str,
    window_size: int = 300,
) -> list[dict]:
    with _lock:
        results = [
            {"timestamp": r["timestamp"], "value": r["value"], "quality": r["quality"]}
            for r in _telemetry
            if r["satellite_id"] == satellite_id and r["parameter"] == parameter
        ]
    results.sort(key=lambda r: r["timestamp"], reverse=True)
    return results[:window_size]


async def get_latest_values(satellite_id: str) -> list[dict]:
    with _lock:
        by_param: dict[str, dict] = {}
        for r in _telemetry:
            if r["satellite_id"] != satellite_id:
                continue
            param = r["parameter"]
            if param not in by_param or r["timestamp"] > by_param[param]["timestamp"]:
                by_param[param] = r
    return list(by_param.values())


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

async def insert_anomaly(anomaly: Anomaly) -> str:
    with _lock:
        _anomalies.append(_anomaly_to_dict(anomaly))
    return anomaly.id


async def get_anomalies(
    satellite_id: str | None = None,
    severity: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    with _lock:
        results = list(_anomalies)
    if satellite_id:
        results = [r for r in results if r["satellite_id"] == satellite_id]
    if severity:
        results = [r for r in results if r["severity"] == severity]
    if since:
        results = [r for r in results if r["timestamp"] >= since]
    results.sort(key=lambda r: r["timestamp"], reverse=True)
    return results[:min(limit, 1000)]


async def get_anomaly_by_id(anomaly_id: str) -> dict | None:
    with _lock:
        for r in _anomalies:
            if r["id"] == anomaly_id:
                return r
    return None


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

async def store_api_key(key_hash: str, label: str) -> None:
    with _lock:
        _api_keys.append({"key_hash": key_hash, "label": label, "active": True})


async def verify_api_key_exists(key_hash: str) -> bool:
    with _lock:
        return any(k["key_hash"] == key_hash and k["active"] for k in _api_keys)


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

async def get_known_satellites() -> list[str]:
    with _lock:
        sats = sorted({r["satellite_id"] for r in _telemetry})
    return sats
