"""Parameterized queries — the ONLY way to talk to the database.

Every query uses $1, $2, ... placeholders. No string interpolation. Ever.
This file is the single source of truth for all SQL operations.
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from sentinel.core.models import Anomaly, TelemetryPoint
from sentinel.db.connection import acquire

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

async def insert_telemetry(points: list[TelemetryPoint]) -> int:
    """Batch insert telemetry points. Returns number of rows inserted."""
    if not points:
        return 0

    async with acquire() as conn:
        records = [
            (
                p.satellite_id,
                p.timestamp,
                p.subsystem,
                p.parameter,
                p.value,
                p.unit,
                p.quality,
            )
            for p in points
        ]
        await conn.executemany(
            """
            INSERT INTO telemetry (satellite_id, timestamp, subsystem, parameter, value, unit, quality)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            records,
        )
        return len(records)


async def get_telemetry(
    satellite_id: str,
    parameter: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Query telemetry with optional filters. Always bounded by limit."""
    async with acquire() as conn:
        conditions = ["satellite_id = $1"]
        params: list = [satellite_id]
        idx = 2

        if parameter:
            conditions.append(f"parameter = ${idx}")
            params.append(parameter)
            idx += 1

        if since:
            conditions.append(f"timestamp >= ${idx}")
            params.append(since)
            idx += 1

        if until:
            conditions.append(f"timestamp <= ${idx}")
            params.append(until)
            idx += 1

        where = " AND ".join(conditions)
        params.append(min(limit, 10_000))  # hard cap

        rows = await conn.fetch(
            f"SELECT * FROM telemetry WHERE {where} ORDER BY timestamp DESC LIMIT ${idx}",
            *params,
        )
        return [dict(r) for r in rows]


async def get_recent_telemetry_window(
    satellite_id: str,
    parameter: str,
    window_size: int = 300,
) -> list[dict]:
    """Get the N most recent points for a specific parameter. Used by detectors."""
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT timestamp, value, quality
            FROM telemetry
            WHERE satellite_id = $1 AND parameter = $2
            ORDER BY timestamp DESC
            LIMIT $3
            """,
            satellite_id, parameter, window_size,
        )
        return [dict(r) for r in rows]


async def get_latest_values(satellite_id: str) -> list[dict]:
    """Get the most recent value for every parameter of a satellite.

    Uses DISTINCT ON — PostgreSQL-specific but very efficient.
    """
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (parameter)
                satellite_id, timestamp, subsystem, parameter, value, unit, quality
            FROM telemetry
            WHERE satellite_id = $1
            ORDER BY parameter, timestamp DESC
            """,
            satellite_id,
        )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

async def insert_anomaly(anomaly: Anomaly) -> str:
    """Insert an anomaly record. Returns the anomaly ID."""
    async with acquire() as conn:
        await conn.execute(
            """
            INSERT INTO anomalies
                (id, satellite_id, timestamp, subsystem, parameter, value,
                 severity, confidence, detectors_triggered, explanation,
                 root_cause_group, contributing_params)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            anomaly.id,
            anomaly.satellite_id,
            anomaly.timestamp,
            anomaly.subsystem,
            anomaly.parameter,
            anomaly.value,
            anomaly.severity.value,
            anomaly.confidence,
            list(anomaly.detectors_triggered),
            anomaly.explanation,
            anomaly.root_cause_group,
            json.dumps(anomaly.contributing_params),
        )
        return anomaly.id


async def get_anomalies(
    satellite_id: str | None = None,
    severity: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    """Query anomalies with optional filters."""
    async with acquire() as conn:
        conditions: list[str] = []
        params: list = []
        idx = 1

        if satellite_id:
            conditions.append(f"satellite_id = ${idx}")
            params.append(satellite_id)
            idx += 1

        if severity:
            conditions.append(f"severity = ${idx}")
            params.append(severity)
            idx += 1

        if since:
            conditions.append(f"timestamp >= ${idx}")
            params.append(since)
            idx += 1

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(min(limit, 1000))

        rows = await conn.fetch(
            f"SELECT * FROM anomalies{where} ORDER BY timestamp DESC LIMIT ${idx}",
            *params,
        )
        return [dict(r) for r in rows]


async def get_anomaly_by_id(anomaly_id: str) -> dict | None:
    """Get a single anomaly by ID."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM anomalies WHERE id = $1",
            anomaly_id,
        )
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

async def store_api_key(key_hash: str, label: str) -> None:
    """Store a hashed API key."""
    async with acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys (key_hash, label) VALUES ($1, $2)",
            key_hash, label,
        )


async def verify_api_key_exists(key_hash: str) -> bool:
    """Check if a hashed API key exists and is active."""
    async with acquire() as conn:
        result = await conn.fetchval(
            "SELECT active FROM api_keys WHERE key_hash = $1",
            key_hash,
        )
        return result is True


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

async def get_known_satellites() -> list[str]:
    """Get all satellite IDs that have sent telemetry."""
    async with acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT satellite_id FROM telemetry ORDER BY satellite_id"
        )
        return [r["satellite_id"] for r in rows]
