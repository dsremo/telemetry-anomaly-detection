"""Schema migrations — idempotent, forward-only.

Design principles:
- Every migration is a plain SQL block, applied in a transaction.
- All CREATE statements use IF NOT EXISTS for safe re-runs.
- ADD COLUMN uses IF NOT EXISTS — safe on existing databases.
- TimescaleDB features (hypertable, compression, continuous aggregates,
  retention) are applied conditionally: works on plain PostgreSQL too.
- Continuous aggregates run outside a transaction (TimescaleDB requirement).

Tables:
  telemetry          — raw time-series, TimescaleDB hypertable
  satellites         — satellite registry (auto-populated on ingest)
  channel_registry   — per (satellite, parameter) metadata
  channel_calibration — per-channel CUSUM/EWMA reference distribution
  detector_state     — CUSUM/EWMA accumulator persistence across restarts
  anomalies          — confirmed anomalies with ensemble metadata
  incidents          — root-cause groups of related anomalies
  alerts             — dispatched notifications
  api_keys           — hashed API credentials
  schema_version     — migration tracking
"""

from __future__ import annotations

import structlog

from sentinel.db.connection import acquire, get_pool

logger = structlog.get_logger()

SCHEMA_VERSION = 7


# ---------------------------------------------------------------------------
# Migrations v1-v3: existing schema (keep untouched for safe upgrades)
# ---------------------------------------------------------------------------
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

    # v3: Alerts + API keys + schema version
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
        version     INTEGER PRIMARY KEY,
        applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # v4: Satellite registry + channel registry + dedup + better indexes
    """
    -- Deduplicate telemetry before adding unique constraint.
    -- Keeps the row with the lowest id when duplicates exist.
    DELETE FROM telemetry a
    USING telemetry b
    WHERE a.id > b.id
      AND a.satellite_id = b.satellite_id
      AND a.parameter    = b.parameter
      AND a.timestamp    = b.timestamp;

    -- Unique constraint enables ON CONFLICT DO NOTHING for idempotent ingestion.
    CREATE UNIQUE INDEX IF NOT EXISTS idx_telemetry_unique
        ON telemetry (satellite_id, parameter, timestamp);

    -- Satellite registry — auto-populated on first telemetry from each sat.
    CREATE TABLE IF NOT EXISTS satellites (
        satellite_id        TEXT PRIMARY KEY,
        display_name        TEXT NOT NULL DEFAULT '',
        operator            TEXT NOT NULL DEFAULT '',
        orbit_type          TEXT NOT NULL DEFAULT 'LEO',
        orbital_period_s    INTEGER NOT NULL DEFAULT 5400,
        first_telemetry_at  TIMESTAMPTZ,
        last_telemetry_at   TIMESTAMPTZ,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    -- Channel registry — one row per (satellite, parameter) pair.
    CREATE TABLE IF NOT EXISTS channel_registry (
        satellite_id    TEXT NOT NULL,
        parameter       TEXT NOT NULL,
        subsystem       TEXT NOT NULL DEFAULT '',
        unit            TEXT NOT NULL DEFAULT '',
        first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        total_points    BIGINT NOT NULL DEFAULT 0,
        PRIMARY KEY (satellite_id, parameter)
    );

    CREATE INDEX IF NOT EXISTS idx_channel_registry_sat
        ON channel_registry (satellite_id);

    -- Anomaly query patterns: by confidence ranking, by channel, and by root cause.
    CREATE INDEX IF NOT EXISTS idx_anomalies_confidence
        ON anomalies (confidence DESC);

    CREATE INDEX IF NOT EXISTS idx_anomalies_param_time
        ON anomalies (satellite_id, parameter, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_anomalies_root_cause
        ON anomalies (root_cause_group)
        WHERE root_cause_group IS NOT NULL;

    -- GIN index so JSONB containment queries on contributing_params are fast.
    CREATE INDEX IF NOT EXISTS idx_anomalies_contributing
        ON anomalies USING GIN (contributing_params);

    -- Partial index: unacknowledged alert lookup is the most common alert query.
    CREATE INDEX IF NOT EXISTS idx_alerts_unacked
        ON alerts (satellite_id, dispatched_at DESC)
        WHERE acknowledged = FALSE;

    -- api_keys: track last use for auditing and auto-expiry logic.
    ALTER TABLE api_keys
        ADD COLUMN IF NOT EXISTS last_used_at        TIMESTAMPTZ,
        ADD COLUMN IF NOT EXISTS rate_limit_override INTEGER;
    """,

    # v5: Per-channel calibration state + detector accumulator persistence
    """
    -- Per-channel calibration: reference μ and σ computed from the first
    -- CALIBRATION_WINDOW samples. Used to set CUSUM k/H and EWMA UCL/LCL.
    -- States: warming_up → calibrated → recalibrating (after regime shift)
    CREATE TABLE IF NOT EXISTS channel_calibration (
        satellite_id        TEXT NOT NULL,
        parameter           TEXT NOT NULL,
        state               TEXT NOT NULL DEFAULT 'warming_up',
        ref_mean            DOUBLE PRECISION,
        ref_std             DOUBLE PRECISION,
        ref_sample_count    INTEGER NOT NULL DEFAULT 0,
        calibrated_at       TIMESTAMPTZ,
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (satellite_id, parameter)
    );

    -- Detector accumulator state — survives server restarts.
    -- detector_name: 'cusum' | 'ewma' | 'stl'
    -- state_data: JSONB containing the detector's internal variables.
    --   cusum: { s_pos, s_neg, alarm_count, last_alarm_at }
    --   ewma:  { z_ewma, alarm_count, last_alarm_at }
    --   stl:   { last_decomposed_at, period_estimate }
    CREATE TABLE IF NOT EXISTS detector_state (
        satellite_id    TEXT NOT NULL,
        parameter       TEXT NOT NULL,
        detector_name   TEXT NOT NULL,
        state_data      JSONB NOT NULL DEFAULT '{}',
        last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (satellite_id, parameter, detector_name)
    );

    CREATE INDEX IF NOT EXISTS idx_detector_state_sat
        ON detector_state (satellite_id);
    """,

    # v6: Incidents + anomaly augmentation + TimescaleDB hypertable
    """
    -- Incidents: root-cause groups of related anomalies.
    -- Multiple anomalies firing in the same subsystem within a short window
    -- are grouped into one incident for operator clarity.
    CREATE TABLE IF NOT EXISTS incidents (
        id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        satellite_id        TEXT NOT NULL,
        subsystem           TEXT NOT NULL DEFAULT '',
        severity            TEXT NOT NULL DEFAULT 'watch',
        status              TEXT NOT NULL DEFAULT 'open',
        title               TEXT NOT NULL DEFAULT '',
        root_cause_summary  TEXT NOT NULL DEFAULT '',
        anomaly_count       INTEGER NOT NULL DEFAULT 1,
        first_anomaly_at    TIMESTAMPTZ NOT NULL,
        last_anomaly_at     TIMESTAMPTZ NOT NULL,
        acknowledged_at     TIMESTAMPTZ,
        closed_at           TIMESTAMPTZ,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_incidents_sat_status
        ON incidents (satellite_id, status, last_anomaly_at DESC);

    -- Partial index: open incidents are the hot path (dashboards + alert routing).
    CREATE INDEX IF NOT EXISTS idx_incidents_open
        ON incidents (satellite_id, last_anomaly_at DESC)
        WHERE status = 'open';

    -- Augment anomalies table for review workflow + STL residual storage.
    ALTER TABLE anomalies
        ADD COLUMN IF NOT EXISTS incident_id    UUID REFERENCES incidents(id),
        ADD COLUMN IF NOT EXISTS reviewed       BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS false_positive BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS stl_residual   DOUBLE PRECISION;

    CREATE INDEX IF NOT EXISTS idx_anomalies_incident
        ON anomalies (incident_id)
        WHERE incident_id IS NOT NULL;

    -- Partial index: dashboard shows only unreviewed, non-false-positive anomalies.
    CREATE INDEX IF NOT EXISTS idx_anomalies_unreviewed
        ON anomalies (satellite_id, timestamp DESC)
        WHERE reviewed = FALSE AND false_positive = FALSE;

    -- TimescaleDB: convert telemetry to hypertable for time-partitioned storage.
    -- Safe no-op if extension is not installed — plain PostgreSQL still works.
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
            PERFORM create_hypertable(
                'telemetry',
                'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists       => TRUE,
                migrate_data        => TRUE
            );

            -- Compress chunks older than 7 days.
            -- compress_segmentby: one segment per (satellite, parameter) — optimal
            -- for the most common query pattern: WHERE sat AND param.
            EXECUTE $cfg$
                ALTER TABLE telemetry SET (
                    timescaledb.compress,
                    timescaledb.compress_orderby    = 'timestamp DESC',
                    timescaledb.compress_segmentby  = 'satellite_id, parameter'
                )
            $cfg$;

            PERFORM add_compression_policy(
                'telemetry',
                INTERVAL '7 days',
                if_not_exists => TRUE
            );

            RAISE NOTICE 'TimescaleDB: hypertable + compression policy applied';
        ELSE
            RAISE NOTICE 'TimescaleDB not installed — running plain PostgreSQL';
        END IF;
    END;
    $$;
    """,

    # v7: Continuous aggregates + data retention (TimescaleDB only, no-op otherwise)
    #
    # Continuous aggregates are materialized incrementally by the TimescaleDB
    # background worker.  They cannot be created inside a transaction, so this
    # migration block is intentionally a series of conditional DO blocks that
    # EXCEPTION-catch failures gracefully.
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
            RAISE NOTICE 'v7: TimescaleDB absent — skipping continuous aggregates';
            RETURN;
        END IF;

        -- Hourly rollup: powers the dashboard trend charts without scanning
        -- raw telemetry. The cagg is kept 1 hour behind real-time to avoid
        -- partial-bucket reads (end_offset => '1 hour').
        BEGIN
            EXECUTE $sql$
                CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_hourly
                WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
                SELECT
                    satellite_id,
                    parameter,
                    subsystem,
                    time_bucket(INTERVAL '1 hour', timestamp)  AS hour,
                    AVG(value)     AS avg_value,
                    MIN(value)     AS min_value,
                    MAX(value)     AS max_value,
                    STDDEV(value)  AS stddev_value,
                    COUNT(*)::INT  AS sample_count
                FROM telemetry
                GROUP BY satellite_id, parameter, subsystem, hour
                WITH NO DATA
            $sql$;

            PERFORM add_continuous_aggregate_policy(
                'telemetry_hourly',
                start_offset      => INTERVAL '3 hours',
                end_offset        => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists     => TRUE
            );

            RAISE NOTICE 'v7: telemetry_hourly cagg created';
        EXCEPTION WHEN others THEN
            RAISE NOTICE 'v7: telemetry_hourly skipped: %', SQLERRM;
        END;

        -- Retention: keep raw telemetry for 90 days.
        -- The hourly cagg is never dropped — it becomes the long-term record.
        BEGIN
            PERFORM add_retention_policy(
                'telemetry',
                INTERVAL '90 days',
                if_not_exists => TRUE
            );
            RAISE NOTICE 'v7: 90-day retention policy applied to telemetry';
        EXCEPTION WHEN others THEN
            RAISE NOTICE 'v7: retention policy skipped: %', SQLERRM;
        END;
    END;
    $$;
    """,
]


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

async def run_migrations() -> None:
    """Apply all pending migrations. Idempotent — safe on every startup.

    Each migration block runs in its own transaction so a partial failure
    leaves the schema at the last successfully applied version.
    """
    async with acquire() as conn:
        # Bootstrap: ensure version tracking exists before anything else.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        current = await conn.fetchval(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        )

    logger.info("migrations_start", current_version=current, target_version=SCHEMA_VERSION)

    for version, sql in enumerate(_MIGRATIONS, start=1):
        if version <= current:
            continue

        logger.info("migration_applying", version=version)
        async with acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_version (version) VALUES ($1) "
                    "ON CONFLICT DO NOTHING",
                    version,
                )
        logger.info("migration_applied", version=version)

    final = max(current, len(_MIGRATIONS))
    logger.info("migrations_complete", schema_version=final)
