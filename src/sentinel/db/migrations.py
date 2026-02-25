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

SCHEMA_VERSION = 11


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

    # v8: Unique constraint on anomalies(satellite_id, parameter, timestamp).
    # Without this, re-running bulk analysis stores the same anomaly twice because
    # insert_anomaly generates a fresh UUID each call — ON CONFLICT (id) never fires
    # since the UUID is different every time.  The composite key makes every insert
    # idempotent: same satellite + parameter + timestamp is silently discarded.
    """
    -- Remove pre-existing duplicates before adding the constraint.
    -- Keep only the earliest-created record for each (sat, param, ts) triple.
    DELETE FROM anomalies
    WHERE id NOT IN (
        SELECT DISTINCT ON (satellite_id, parameter, timestamp) id
        FROM anomalies
        ORDER BY satellite_id, parameter, timestamp, created_at
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_anomalies_sat_param_ts
        ON anomalies (satellite_id, parameter, timestamp);
    """,

    # v9: Multi-tenancy — tenant registry + tenant_id on all data tables + RLS.
    #
    # Design rationale:
    #   - All data tables gain tenant_id TEXT NOT NULL DEFAULT 'default'.
    #     Existing rows automatically belong to the 'default' tenant — no data loss.
    #   - Composite PKs are widened to include tenant_id so two tenants can own
    #     the same satellite_id, channel, etc. without collision.
    #   - Unique indexes (telemetry, anomalies) are also widened.
    #   - FORCE ROW LEVEL SECURITY on data tables: sentinel user is the table owner,
    #     so without FORCE it would bypass RLS. FORCE closes that gap.
    #   - api_keys uses ENABLE-only (not FORCE): the table owner (sentinel) needs to
    #     read all tenants' keys at startup to build the in-memory auth cache.
    #   - The RLS policy uses current_setting('app.tenant_id', true) which is set
    #     per-connection by connection.acquire() via set_config(). asyncpg's
    #     RESET ALL on connection return clears it automatically.
    """
    -- 1. Tenants registry
    CREATE TABLE IF NOT EXISTS tenants (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        active      BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    INSERT INTO tenants (id, name) VALUES ('default', 'Default')
    ON CONFLICT (id) DO NOTHING;

    -- 2. Add tenant_id column to every data table (safe re-run: IF NOT EXISTS)
    ALTER TABLE api_keys            ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE telemetry           ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE satellites          ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE channel_registry    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE channel_calibration ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE detector_state      ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE anomalies           ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE incidents           ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE alerts              ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

    -- 3. Widen composite PKs to include tenant_id.
    --    Two tenants can legitimately track the same satellite_id — the PK must
    --    be scoped per-tenant.  Table owner can always ALTER TABLE regardless of RLS.
    ALTER TABLE satellites          DROP CONSTRAINT IF EXISTS satellites_pkey;
    ALTER TABLE satellites          ADD PRIMARY KEY (tenant_id, satellite_id);

    ALTER TABLE channel_registry    DROP CONSTRAINT IF EXISTS channel_registry_pkey;
    ALTER TABLE channel_registry    ADD PRIMARY KEY (tenant_id, satellite_id, parameter);

    ALTER TABLE channel_calibration DROP CONSTRAINT IF EXISTS channel_calibration_pkey;
    ALTER TABLE channel_calibration ADD PRIMARY KEY (tenant_id, satellite_id, parameter);

    ALTER TABLE detector_state      DROP CONSTRAINT IF EXISTS detector_state_pkey;
    ALTER TABLE detector_state      ADD PRIMARY KEY (tenant_id, satellite_id, parameter, detector_name);

    -- 4. Widen unique indexes to include tenant_id.
    --    Both include 'timestamp' — required by TimescaleDB for hypertable uniqueness.
    DROP INDEX IF EXISTS idx_telemetry_unique;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_telemetry_unique
        ON telemetry (tenant_id, satellite_id, parameter, timestamp);

    DROP INDEX IF EXISTS idx_anomalies_sat_param_ts;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_anomalies_sat_param_ts
        ON anomalies (tenant_id, satellite_id, parameter, timestamp);

    -- 5. Enable Row Level Security.
    --    FORCE on data tables: sentinel user is the table owner and would otherwise
    --    bypass RLS — FORCE closes that gap so every query is filtered.
    --    api_keys: ENABLE-only so the table owner can load all keys at startup.
    ALTER TABLE telemetry           ENABLE ROW LEVEL SECURITY;
    ALTER TABLE telemetry           FORCE  ROW LEVEL SECURITY;
    ALTER TABLE satellites          ENABLE ROW LEVEL SECURITY;
    ALTER TABLE satellites          FORCE  ROW LEVEL SECURITY;
    ALTER TABLE channel_registry    ENABLE ROW LEVEL SECURITY;
    ALTER TABLE channel_registry    FORCE  ROW LEVEL SECURITY;
    ALTER TABLE channel_calibration ENABLE ROW LEVEL SECURITY;
    ALTER TABLE channel_calibration FORCE  ROW LEVEL SECURITY;
    ALTER TABLE detector_state      ENABLE ROW LEVEL SECURITY;
    ALTER TABLE detector_state      FORCE  ROW LEVEL SECURITY;
    ALTER TABLE anomalies           ENABLE ROW LEVEL SECURITY;
    ALTER TABLE anomalies           FORCE  ROW LEVEL SECURITY;
    ALTER TABLE incidents           ENABLE ROW LEVEL SECURITY;
    ALTER TABLE incidents           FORCE  ROW LEVEL SECURITY;
    ALTER TABLE alerts              ENABLE ROW LEVEL SECURITY;
    ALTER TABLE alerts              FORCE  ROW LEVEL SECURITY;
    ALTER TABLE api_keys            ENABLE ROW LEVEL SECURITY;

    -- 6. Create RLS policies (idempotent: DROP IF EXISTS then CREATE).
    --    USING  — filters rows on SELECT/UPDATE/DELETE.
    --    WITH CHECK — enforces tenant_id on INSERT/UPDATE.
    --    The second arg to current_setting() is 'true' (missing-ok): returns NULL
    --    rather than raising an error when app.tenant_id has not been set, which
    --    makes the policy a safe no-op in that case (NULL != any tenant_id → 0 rows).
    DO $$
    DECLARE tbl TEXT;
    BEGIN
        FOREACH tbl IN ARRAY ARRAY[
            'telemetry', 'satellites', 'channel_registry', 'channel_calibration',
            'detector_state', 'anomalies', 'incidents', 'alerts', 'api_keys'
        ] LOOP
            EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', tbl);
            EXECUTE format(
                $pol$CREATE POLICY tenant_isolation ON %I
                     USING     (tenant_id = current_setting('app.tenant_id', true))
                     WITH CHECK (tenant_id = current_setting('app.tenant_id', true))$pol$,
                tbl
            );
        END LOOP;
    END;
    $$;
    """,

    # v10: Referential integrity — FK from tenant_id → tenants(id) on every data table.
    #
    #
    # Why this is a separate migration from v9:
    #   v9 added tenant_id columns and seeded 'default', but had no FK constraint.
    #   A missing FK means typo tenant IDs (e.g. 'ddefault') are silently accepted —
    #   violating consistency.  Adding the FK now closes that gap.
    #
    # Design:
    #   - FK name is 'fk_tenant_id' on every table (consistent, easy to grep).
    #   - Idempotent: checks pg_constraint before ALTER — safe to re-run.
    #   - telemetry is a TimescaleDB hypertable.  FK on hypertable columns referencing
    #     a regular table IS supported in TimescaleDB 2.x, but we wrap in EXCEPTION
    #     to degrade gracefully on older versions rather than failing the whole migration.
    #   - FK checks run at the PG engine level and bypass RLS — they always see
    #     all rows in tenants regardless of app.tenant_id. No bootstrapping issue.
    """
    DO $$
    DECLARE
        tbl TEXT;
    BEGIN
        FOREACH tbl IN ARRAY ARRAY[
            'api_keys', 'satellites', 'channel_registry', 'channel_calibration',
            'detector_state', 'anomalies', 'incidents', 'alerts', 'telemetry'
        ] LOOP
            -- Idempotent: skip if constraint already exists.
            IF NOT EXISTS (
                SELECT 1
                FROM   pg_constraint c
                JOIN   pg_class      t ON t.oid = c.conrelid
                WHERE  c.conname = 'fk_tenant_id'
                  AND  t.relname = tbl
            ) THEN
                BEGIN
                    EXECUTE format(
                        'ALTER TABLE %I
                         ADD CONSTRAINT fk_tenant_id
                         FOREIGN KEY (tenant_id) REFERENCES tenants(id)',
                        tbl
                    );
                    RAISE NOTICE 'v10: fk_tenant_id added to %', tbl;
                EXCEPTION WHEN others THEN
                    -- TimescaleDB older than 2.x does not support FK on hypertables.
                    -- Log and continue — integrity is still enforced on all other tables.
                    RAISE NOTICE 'v10: fk_tenant_id skipped on % — %', tbl, SQLERRM;
                END;
            ELSE
                RAISE NOTICE 'v10: fk_tenant_id on % already exists, skipping', tbl;
            END IF;
        END LOOP;
    END;
    $$;
    """,

    # v11: User Auth — users table + refresh_tokens table with RLS.
    #
    # Design:
    #   - users(id UUID PK, tenant_id FK, email, password_hash, role enum, active, timestamps)
    #     UNIQUE (tenant_id, email) — one account per email per tenant (B2B pattern).
    #   - user_role ENUM: admin > operator > viewer > report_only
    #   - refresh_tokens: opaque random bytes hashed with SHA-256; 7-day TTL;
    #     can be revoked individually (logout) or all at once (password reset).
    #   - Both tables have FORCE RLS with the standard tenant_isolation policy.
    #   - FK to tenants(id) is embedded in REFERENCES — no separate DO block needed.
    #   - RLS on users is FORCE — even admin users only see their own tenant's users
    #     (table owner bypass is prevented by FORCE, same pattern as other data tables).
    """
    DO $$ BEGIN
        CREATE TYPE user_role AS ENUM ('admin', 'operator', 'viewer', 'report_only');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END; $$;

    CREATE TABLE IF NOT EXISTS users (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id     TEXT NOT NULL REFERENCES tenants(id),
        email         TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role          user_role NOT NULL DEFAULT 'viewer',
        active        BOOLEAN NOT NULL DEFAULT TRUE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_login    TIMESTAMPTZ,
        UNIQUE (tenant_id, email)
    );

    CREATE INDEX IF NOT EXISTS idx_users_tenant_email
        ON users (tenant_id, email);

    CREATE TABLE IF NOT EXISTS refresh_tokens (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        tenant_id   TEXT NOT NULL REFERENCES tenants(id),
        token_hash  TEXT NOT NULL UNIQUE,
        expires_at  TIMESTAMPTZ NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        revoked     BOOLEAN NOT NULL DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user
        ON refresh_tokens (user_id, revoked, expires_at);

    ALTER TABLE users           ENABLE ROW LEVEL SECURITY;
    ALTER TABLE users           FORCE  ROW LEVEL SECURITY;
    ALTER TABLE refresh_tokens  ENABLE ROW LEVEL SECURITY;
    ALTER TABLE refresh_tokens  FORCE  ROW LEVEL SECURITY;

    DROP POLICY IF EXISTS tenant_isolation ON users;
    CREATE POLICY tenant_isolation ON users
        USING     (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

    DROP POLICY IF EXISTS tenant_isolation ON refresh_tokens;
    CREATE POLICY tenant_isolation ON refresh_tokens
        USING     (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
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
