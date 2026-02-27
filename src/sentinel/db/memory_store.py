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
_alerts: list[dict] = []
_api_keys: list[dict] = []
_channel_configs: dict[tuple[str, str], dict] = {}  # (satellite_id, parameter) → config
_alert_configs: dict[str, dict] = {}               # tenant_id → config
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


# ---------------------------------------------------------------------------
# Channel registry + per-channel config (demo stubs)
# ---------------------------------------------------------------------------

async def get_channel_stats(satellite_id: str | None = None) -> list[dict]:
    """Demo stub — returns empty list (no channels until telemetry is ingested)."""
    return []


async def get_channel_config(satellite_id: str, parameter: str) -> dict | None:
    """Return in-memory config override for the channel, or None."""
    with _lock:
        return dict(_channel_configs.get((satellite_id, parameter), {})) or None


async def upsert_channel_config(
    satellite_id: str,
    parameter: str,
    *,
    z_threshold: float | None = None,
    cusum_h: float | None = None,
    cusum_k: float | None = None,
    ewma_lambda: float | None = None,
    ewma_sigma_mult: float | None = None,
    min_confidence: float | None = None,
    alert_cooldown_s: int | None = None,
) -> dict:
    """Insert or partial-update in-memory channel config. Returns the full row."""
    from datetime import timezone
    key = (satellite_id, parameter)
    with _lock:
        existing = _channel_configs.get(key, {})
        _FIELDS = ("z_threshold", "cusum_h", "cusum_k", "ewma_lambda",
                   "ewma_sigma_mult", "min_confidence", "alert_cooldown_s")
        new_vals = dict(zip(
            _FIELDS,
            (z_threshold, cusum_h, cusum_k, ewma_lambda, ewma_sigma_mult, min_confidence, alert_cooldown_s),
        ))
        merged = {
            f: (new_vals[f] if new_vals[f] is not None else existing.get(f))
            for f in _FIELDS
        }
        merged["updated_at"] = datetime.now(timezone.utc)
        merged["satellite_id"] = satellite_id
        merged["parameter"] = parameter
        _channel_configs[key] = merged
        return dict(merged)


async def delete_channel_config(satellite_id: str, parameter: str) -> bool:
    """Remove in-memory config override. Returns True if row existed."""
    key = (satellite_id, parameter)
    with _lock:
        existed = key in _channel_configs
        _channel_configs.pop(key, None)
    return existed


async def load_all_channel_configs(satellite_id: str | None = None) -> list[dict]:
    """Return all in-memory channel config rows (no RLS needed in demo mode)."""
    with _lock:
        configs = list(_channel_configs.values())
    if satellite_id is not None:
        configs = [c for c in configs if c.get("satellite_id") == satellite_id]
    return configs


# ---------------------------------------------------------------------------
# Alert dispatch records (demo stubs)
# ---------------------------------------------------------------------------

async def insert_alert(anomaly: "Anomaly") -> str:  # type: ignore[name-defined]
    """Stub: store alert in-memory."""
    from datetime import timezone
    import uuid as _uuid
    alert_id = _uuid.uuid4().hex[:12]
    title = f"[{anomaly.severity.value.upper()}] {anomaly.satellite_id} — {anomaly.parameter}"
    with _lock:
        _alerts.append({
            "id": alert_id,
            "satellite_id": anomaly.satellite_id,
            "severity": anomaly.severity.value,
            "title": title,
            "message": anomaly.explanation,
            "dispatched_at": datetime.now(timezone.utc),
            "acknowledged": False,
            "subsystem": anomaly.subsystem,
            "parameter": anomaly.parameter,
            "value": anomaly.value,
            "confidence": anomaly.confidence,
            "anomaly_timestamp": anomaly.timestamp,
            "explanation": anomaly.explanation,
        })
    return alert_id


async def get_alerts(
    satellite_id: str | None = None,
    severity: str | None = None,
    since: "datetime | None" = None,
    acknowledged: bool | None = None,
    limit: int = 100,
) -> list[dict]:
    """Demo stub: return in-memory alert records."""
    with _lock:
        results = list(_alerts)
    if satellite_id:
        results = [r for r in results if r["satellite_id"] == satellite_id]
    if severity:
        results = [r for r in results if r["severity"] == severity]
    if since:
        results = [r for r in results if r["dispatched_at"] >= since]
    if acknowledged is not None:
        results = [r for r in results if r["acknowledged"] == acknowledged]
    results.sort(key=lambda r: r["dispatched_at"], reverse=True)
    return results[:min(limit, 1000)]


async def acknowledge_alert(alert_id: str) -> bool:
    """Demo stub: mark alert acknowledged."""
    with _lock:
        for alert in _alerts:
            if alert["id"] == alert_id and not alert["acknowledged"]:
                alert["acknowledged"] = True
                return True
    return False


# ---------------------------------------------------------------------------
# Alert configs (per-tenant delivery settings — demo stubs)
# ---------------------------------------------------------------------------

async def upsert_alert_config(
    tenant_id: str,
    *,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
    email_to: list[str] | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    min_severity: str | None = None,
    dedup_window_s: int | None = None,
    escalation_delay_s: int | None = None,
    enabled: bool | None = None,
) -> dict:
    """Demo stub: insert or partial-update alert config in memory."""
    from datetime import timezone
    with _lock:
        existing = _alert_configs.get(tenant_id, {
            "tenant_id": tenant_id,
            "webhook_url": None,
            "webhook_secret": None,
            "email_to": None,
            "smtp_host": None,
            "smtp_port": 587,
            "smtp_user": None,
            "smtp_password": None,
            "min_severity": "warning",
            "dedup_window_s": 300,
            "escalation_delay_s": 600,
            "enabled": True,
            "updated_at": None,
        })
        # Partial update: only overwrite non-None values
        updates = {
            "webhook_url": webhook_url,
            "webhook_secret": webhook_secret,
            "email_to": email_to,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
            "min_severity": min_severity,
            "dedup_window_s": dedup_window_s,
            "escalation_delay_s": escalation_delay_s,
            "enabled": enabled,
        }
        merged = dict(existing)
        for k, v in updates.items():
            if v is not None:
                merged[k] = v
        merged["updated_at"] = datetime.now(timezone.utc)
        _alert_configs[tenant_id] = merged
    return dict(merged)


async def get_alert_config(tenant_id: str | None = None) -> dict | None:
    """Demo stub: return alert config for the given (or default) tenant."""
    key = tenant_id or "default"
    with _lock:
        row = _alert_configs.get(key)
    return dict(row) if row else None


async def delete_alert_config(tenant_id: str) -> bool:
    """Demo stub: remove alert config."""
    with _lock:
        existed = tenant_id in _alert_configs
        _alert_configs.pop(tenant_id, None)
    return existed


async def load_all_alert_configs() -> list[dict]:
    """Demo stub: return all alert configs."""
    with _lock:
        return [dict(v) for v in _alert_configs.values() if v.get("enabled", True)]
