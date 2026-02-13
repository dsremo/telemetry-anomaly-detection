"""Schema migrations — idempotent, forward-only.

Every migration is a plain SQL string executed in order.
All CREATE statements use IF NOT EXISTS for safe re-runs.
No migration framework needed at this stage.
"""

from __future__ import annotations

import structlog

from sentinel.db.connection import acquire

logger = structlog.get_logger()

SCHEMA_VERSION = 3

_MIGRATIONS: list[str] = [
    # v1: Core telemetry storage
    """
    CREATE TABLE IF NOT EXISTS telemetry (
        id              BIGSERIAL PRIMARY KEY,
        satellite_id    TEXT NOT NULL,
        timestamp       TIMESTAMPTZ NOT NULL,
        subsystem       TEXT NOT NULL,
        parameter       TEXT NOT NULL,
        value           DOUBLE PRECISION NOT NULL,
        unit            TEXT NOT NULL DEFAULT '',
        quality         REAL NOT NULL DEFAULT 1.0,
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_telemetry_sat_time
        ON telemetry (satellite_id, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_telemetry_param_time
        ON telemetry (satellite_id, parameter, timestamp DESC);
    """,

    # v2: Anomaly records
    """
    CREATE TABLE IF NOT EXISTS anomalies (
        id                  TEXT PRIMARY KEY,
        satellite_id        TEXT NOT NULL,
        timestamp           TIMESTAMPTZ NOT NULL,
        subsystem           TEXT NOT NULL,
        parameter           TEXT NOT NULL,
        value               DOUBLE PRECISION NOT NULL,
        severity            TEXT NOT NULL DEFAULT 'watch',
        confidence          REAL NOT NULL DEFAULT 0.0,
        detectors_triggered TEXT[] NOT NULL DEFAULT '{}',
        explanation         TEXT NOT NULL DEFAULT '',
        root_cause_group    TEXT,
        contributing_params JSONB NOT NULL DEFAULT '{}',
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_anomalies_sat_time
        ON anomalies (satellite_id, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_anomalies_severity
        ON anomalies (severity, timestamp DESC);
    """,

    # v3: Alerts + API keys
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id              TEXT PRIMARY KEY,
        anomaly_id      TEXT NOT NULL REFERENCES anomalies(id),
        satellite_id    TEXT NOT NULL,
        severity        TEXT NOT NULL,
        title           TEXT NOT NULL,
        message         TEXT NOT NULL DEFAULT '',
        dispatched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        acknowledged    BOOLEAN NOT NULL DEFAULT FALSE
    );

    CREATE TABLE IF NOT EXISTS api_keys (
        key_hash        TEXT PRIMARY KEY,
        label           TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        active          BOOLEAN NOT NULL DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
]


async def run_migrations() -> None:
    """Apply all pending migrations. Idempotent — safe to run on every startup."""
    async with acquire() as conn:
        # Ensure version tracking table exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        current = await conn.fetchval(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        )

        for i, sql in enumerate(_MIGRATIONS, start=1):
            if i <= current:
                continue

            logger.info("migration_applying", version=i)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES ($1) ON CONFLICT DO NOTHING",
                    i,
                )
            logger.info("migration_applied", version=i)

        logger.info("migrations_complete", current_version=max(current, len(_MIGRATIONS)))
