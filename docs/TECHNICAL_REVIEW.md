# Dsremo — Technical Review Document

**Prepared for:** Senior Engineer Review
**Product:** Dsremo AI Telemetry Anomaly Detection Engine
**Version:** 0.1.0
**Date:** March 2026
**Author:** Ashutosh (Founder)

---

## Table of Contents

1. [What We Built — Executive Summary](#1-what-we-built)
2. [System Architecture](#2-system-architecture)
3. [Detection Pipeline — The Core Algorithm](#3-detection-pipeline)
4. [Data Ingestion Layer](#4-data-ingestion-layer)
5. [Database Design](#5-database-design)
6. [API Design](#6-api-design)
7. [Security Model](#7-security-model)
8. [Alert & Incident Management](#8-alert--incident-management)
9. [ML Detectors — Design Decisions](#9-ml-detectors)
10. [Code Quality & Engineering Standards](#10-code-quality)
11. [Test Coverage](#11-test-coverage)
12. [Benchmark Results](#12-benchmark-results)
13. [Known Gaps & Next Steps](#13-known-gaps--next-steps)
14. [Pricing & Deployment Plan](#14-pricing--deployment-plan)

---

## 1. What We Built

Dsremo is a **ground-based satellite telemetry anomaly detection engine** built for small satellite operators, aerospace companies, and any team ingesting time-series sensor data. It provides an automated pipeline from raw telemetry to explainable anomaly alerts.

### Problem We Solve

Satellite operators receive hundreds to thousands of telemetry channels per second. Traditional threshold-based monitoring:
- Generates massive false-positive floods on normal orbital variations
- Misses gradual drift anomalies that develop over hours
- Cannot correlate anomalies across channels to identify root causes
- Requires domain experts to set thresholds for every channel manually

Dsremo replaces manual threshold tuning with a self-calibrating 12-detector ensemble that learns each channel's normal behavior and flags genuine deviations.

### What It Does (User-Facing)

1. **Ingest** — Upload CSV telemetry or connect live data sources (YAMCS, InfluxDB, SatNOGS, ESA)
2. **Detect** — 12 detectors run per channel: statistical + ML methods, STL decomposition removes orbital seasonality first
3. **Explain** — Every anomaly comes with a plain-English explanation: "Gradual drift accumulation on battery_voltage (CUSUM+EWMA, z=4.2)"
4. **Alert** — Webhook or email notifications with dedup and cooldown to prevent alert fatigue
5. **Investigate** — Dashboard with real-time WebSocket updates, anomaly timeline, and operator feedback

---

## 2. System Architecture

```
                    Internet Clients
                         |
              ┌──────────▼──────────────┐
              │  FastAPI HTTP Server     │
              │  + WebSocket broadcast   │
              │  + Static Dashboard UI   │
              └──────────┬──────────────┘
                         |
          ┌──────────────┼──────────────────┐
          │              │                  │
   ┌──────▼──────┐ ┌─────▼──────┐  ┌───────▼──────┐
   │  Ingest     │ │  Detection  │  │  Alert       │
   │  Layer      │ │  Pipeline   │  │  Service     │
   │             │ │  (12 det.)  │  │              │
   │  CSV        │ │             │  │  Webhook     │
   │  YAMCS      │ │  STL Decomp │  │  Email       │
   │  InfluxDB   │ │  CUSUM/EWMA │  │  Dedup       │
   │  SatNOGS    │ │  GRU/TCN    │  │  Escalation  │
   │  ESA        │ │  Ensemble   │  │              │
   └──────┬──────┘ └─────┬──────┘  └───────┬──────┘
          │              │                  │
          └──────────────▼──────────────────┘
                         │
              ┌──────────▼──────────────┐
              │  PostgreSQL + TimescaleDB│
              │  Row-Level Security      │
              │  Multi-tenant isolation  │
              └─────────────────────────┘
```

### Technology Stack

| Layer | Choice | Reason |
|---|---|---|
| Runtime | Python 3.10+ | Type hints, asyncio, strong ML ecosystem |
| HTTP Framework | FastAPI | Async, auto-OpenAPI, strong typing via Pydantic |
| Database | PostgreSQL + TimescaleDB | Time-series queries, hypertables, compression |
| DB Driver | asyncpg | Non-blocking, typed, no ORM overhead |
| ML | PyTorch (GRU, TCN) + NumPy | CPU-only, no GPU required |
| STL | statsmodels | Seasonal-Trend decomposition |
| Changepoint | ruptures (PELT) | Fast binary-search changepoint detection |
| Logging | structlog | JSON structured logs, production-ready |
| Config | dynaconf | Multi-env YAML + env var override |
| HTTP Client | httpx | Async, retries, timeout |

### Key Design Principles

- **No ORM** — All SQL is hand-written with parameterized queries. Zero SQL injection risk, full control over query plans
- **No Redis, no Kafka** — Single-process asyncio. Sufficient for hundreds of channels. Eliminates operational complexity
- **Frozen dataclasses** — Domain models (`Anomaly`, `Incident`, `DetectorResult`) are immutable. No accidental mutation bugs
- **Multi-tenant from day one** — PostgreSQL Row-Level Security isolates every customer's data. A compromised API key cannot read other tenants' data

---

## 3. Detection Pipeline

This is the core intellectual property of the product. Every telemetry channel runs through the following pipeline per detection cycle:

### Step 1: STL Decomposition (Seasonal-Trend Decomposition)

```
raw_values → STL → { trend, seasonal, residuals }
```

STL removes the orbital period from the signal before detection. Without this, every satellite passage over the ground station looks like a spike.

- **FFT-based auto-period detection** — Finds the dominant oscillation frequency in the data using `np.fft.rfft`. Falls back to `orbital_period_s / sample_interval` if FFT finds no clear periodicity.
- **Savitzky-Golay fallback** — When data is insufficient for STL, uses SG filter as a smooth trend estimator.

### Step 2: Self-Calibration

Each channel maintains a `CalibrationState` with:
- `ref_mean`, `ref_std` — rolling reference distribution of residuals
- `CUSUM` H, K thresholds — dynamically set from ref_std
- `EWMA` UCL/LCL — control limits from rolling statistics
- Calibration fires after `min_samples` (configurable, default: 50) and re-fires when distribution shifts significantly

This means **zero manual threshold configuration** is required for new channels.

### Step 3: The 12-Detector Ensemble

Each detector receives the STL residuals (or raw values for multivariate detectors) and returns a `DetectorResult`:

```python
@dataclass(frozen=True)
class DetectorResult:
    detector_name: str
    is_anomaly:    bool
    score:         float          # [0.0, 1.0]
    severity:      Severity       # NOMINAL / WATCH / WARNING / CRITICAL
    details:       dict           # detector-specific diagnostics
```

| # | Detector | Method | What It Catches |
|---|---|---|---|
| 1 | **CUSUM** | Cumulative sum of residuals | Gradual drift — accumulates over time |
| 2 | **EWMA** | Exponentially weighted moving avg | Sudden level shifts |
| 3 | **Z-Score** | Statistical (σ) | Single-point spikes |
| 4 | **PELT** | Pruned exact linear time (ruptures) | Abrupt structural breaks |
| 5 | **Isolation Forest** | sklearn ensemble, multivariate | Cross-parameter covariance anomalies |
| 6 | **Variance** | Rolling std / ref_std ratio | Variance spikes (increased noise floor) |
| 7 | **GRU Autoencoder** | PyTorch, reconstruction MSE | Nonlinear temporal patterns |
| 8 | **TCN** | Dilated causal convolutions | Long-range temporal dependencies |
| 9 | **Trend Velocity** | STL trend acceleration | Onset detection — early drift warning |
| 10 | **Discord (Matrix Profile)** | Pure NumPy FFT distance | Novel subsequence patterns (never-seen-before) |
| 11 | **Correlation Graph** | Rolling Pearson across channels | Cross-channel decoupling anomalies |
| 12 | **BOCPD** | Bayesian Online Changepoint (Adams & MacKay 2007) | Structural regime changes with calibrated probability |

### Step 4: Ensemble Vote

```python
WEIGHTS = {
    "cusum": 0.17, "ewma": 0.14, "statistical": 0.10,
    "changepoint": 0.00, "isolation_forest": 0.05, "variance": 0.07,
    "lstm": 0.10, "tcn": 0.09, "trend_velocity": 0.08,
    "matrix_profile": 0.06, "correlation_graph": 0.06, "bocpd": 0.08
}  # sum = 1.0 (verified by test)
```

- **Weighted confidence**: `confidence = Σ(weight_i × score_i)` for flagging detectors
- **Agreement factor**: `agreement = n_flagging / n_total` — more detectors agreeing → higher confidence
- **Final confidence**: `min(1.0, weighted_conf × (1 + 0.5 × agreement))`

Severity gates (all configurable in `dsremo.yaml`):
- WATCH ≥ 0.50
- WARNING ≥ 0.65
- CRITICAL ≥ 0.85

### Step 5: Incident Grouping

Individual per-channel anomalies are grouped into `Incidents` by `IncidentGrouper`:
- Anomalies within 300 seconds on the same satellite → same incident
- Incident auto-closes after 3600 seconds of silence
- Severity = max of member anomalies; root cause derived from detector pattern

This follows the NASA GSFC event correlation pattern (cf. NPR 7150.2D §5.3): "No single raw sensor alert reaches an operator without correlation."

### Step 6: Explanation Generation

Every anomaly gets a plain-English explanation built by `_build_explanation()`:

```
"Autoencoder: reconstruction MSE=0.0823 (threshold=0.0412, z=3.7)"
"CUSUM: accumulated drift H=12.3 (limit=6.0), direction=up"
"Variance spike: rolling_std/ref_std = 3.2× (threshold: 2.5)"
```

---

## 4. Data Ingestion Layer

### Supported Sources

| Connector | Protocol | Auth | Notes |
|---|---|---|---|
| `CSVConnector` | File/Upload | None | Wide-format CSV, auto-resample |
| `YAMCSConnector` | HTTP REST | Bearer | Mission planning & ops standard |
| `InfluxDBConnector` | HTTP + Flux | Token | 1-line Flux query |
| `SatNOGSFetcher` | HTTP REST | API Token | Open-source ground station network |
| `ESADataLoader` | HTTP REST | API Key | ESA mission telemetry |
| Direct JSON | POST /telemetry | JWT/API Key | Real-time push |

### Common Base: `HTTPConnector`

All HTTP connectors extend `HTTPConnector` which provides:
- Shared `_retry(client, method, url, **kwargs)` — handles both GET and POST
- 429 rate-limit backoff with Retry-After header respect
- Exponential backoff on TransportError (3 attempts: 1s, 2s, 4s)
- Structured logging on every retry (includes url, method, attempt, wait)

### Bulk Loading Pattern

`run_bulk_detection()` in `bulk_loader.py` is the offline analysis path:
1. Per-channel row count check (`check_channel_row_count()` — no inline SQL)
2. Full detection history run via `analyze_channel_history()`
3. Global state save/restore in `try/finally` — **always** restores even on exception
4. tqdm progress bar for CLI usage

### CSV Upload

- Max 10 MB per upload
- Wide-format: `timestamp, param1, param2, ...`
- Auto-resampling to configurable interval (1–1440 minutes)
- Idempotent: channels with ≥ 50,000 rows are skipped (safe re-upload)
- Returns per-channel row counts in response

---

## 5. Database Design

### Tables

```sql
telemetry          -- Raw time-series (TimescaleDB hypertable, partitioned by month)
satellites         -- Registry of all seen satellites
channel_registry   -- (satellite, parameter) metadata
channel_calibration -- CUSUM/EWMA reference state per channel
detector_state     -- Accumulator persistence (CUSUM H, EWMA S) across restarts
anomalies          -- Detected anomalies with ensemble metadata (JSONB)
incidents          -- Root-cause groups (many anomalies → one incident)
alerts             -- Dispatched notifications with dedup key
alert_configs      -- Per-tenant webhook/email config
channel_config     -- Per-channel threshold overrides
api_keys           -- SHA-256 hashed API credentials
users              -- Tenant-scoped users (RBAC: viewer / operator / admin)
refresh_tokens     -- JWT refresh tokens
schema_version     -- Migration state tracker
```

### Key Design Decisions

**TimescaleDB Hypertable on `telemetry`**
Partitioned by month. Automatic compression after 7 days. Continuous aggregates for 1-minute OHLC queries. Retention policy configurable per tenant.

**Row-Level Security (FORCE RLS) on `telemetry` and `anomalies`**
Every query is isolated by `app.tenant_id` session variable. A compromised connection cannot see other tenants' data. The `api_keys` table explicitly excludes RLS to allow the startup `load_api_key_map()` to see all keys without setting a tenant.

**UNNEST bulk inserts**
All batch inserts use `UNNEST($1::text[], $2::timestamptz[], ...)` — one SQL statement for any batch size. No N+1 insert loops.

**Nullable parameter SQL pattern**
Optional filter parameters use `$N::type IS NULL OR column = $N` — eliminates SQL branching and reduces query surface:
```sql
WHERE ($3::timestamptz IS NULL OR timestamp > $3)
  AND ($4::text IS NULL OR satellite_id = $4)
```

**COALESCE partial update**
Channel config updates preserve existing values for unset fields:
```sql
UPDATE channel_config SET
    z_threshold = COALESCE($3, z_threshold),
    cusum_h     = COALESCE($4, cusum_h)
WHERE tenant_id = $1 AND satellite_id = $2
```

**Schema migrations**
Forward-only, versioned, applied in transactions. All `CREATE` statements use `IF NOT EXISTS`. Safe to run on startup every time. Current version: 18.

---

## 6. API Design

### Route Structure

```
GET  /api/v1/health                         -- No auth, always accessible
GET  /api/v1/stats                          -- Telemetry + anomaly counts

POST /api/v1/telemetry                      -- Ingest batch (operator)
POST /api/v1/telemetry/single              -- Ingest single point (operator)
POST /api/v1/telemetry/upload              -- CSV upload (operator)
GET  /api/v1/telemetry/{satellite_id}      -- Query history (viewer)
POST /api/v1/telemetry/{sat}/analyze       -- On-demand full detection (operator)

GET  /api/v1/anomalies                      -- List with pagination/filters
GET  /api/v1/anomalies/{id}                -- Single with full explanation
PATCH /api/v1/anomalies/{id}/feedback      -- Mark TP/FP (operator feedback loop)

GET  /api/v1/satellites                    -- List known satellites

GET  /api/v1/incidents                     -- Active incidents
GET  /api/v1/incidents/{id}               -- Incident detail

PUT  /api/v1/channels/{sat}/{param}/config -- Per-channel threshold override
GET  /api/v1/channels/{sat}/{param}/config -- Read config
DELETE /api/v1/channels/{sat}/{param}/config -- Reset to defaults

POST /api/v1/satellites/{sat}/suppress     -- Alert suppression window
DELETE /api/v1/satellites/{sat}/suppress   -- Lift suppression
GET  /api/v1/satellites/{sat}/suppress     -- List suppressions

POST /api/v1/connectors/yamcs/pull        -- Pull from YAMCS
POST /api/v1/connectors/influxdb/pull     -- Pull from InfluxDB
POST /api/v1/connectors/satnogs/pull      -- Pull from SatNOGS

POST /api/v1/auth/login                   -- JWT login
POST /api/v1/auth/refresh                 -- Token refresh
POST /api/v1/auth/logout                  -- Revoke token

GET  /api/v1/users                        -- List users (admin)
POST /api/v1/users                        -- Create user (admin)
PUT  /api/v1/users/{id}                   -- Update user (admin)

POST /api/v1/keys                         -- Generate API key (admin)
DELETE /api/v1/keys/{id}                  -- Revoke key (admin)

WS   /api/v1/ws                           -- Real-time anomaly stream
```

### RBAC (Role-Based Access Control)

| Role | Permissions |
|---|---|
| viewer | GET anomalies, telemetry, incidents, satellites |
| operator | viewer + POST/PATCH telemetry, analyze, feedback, channel config |
| admin | operator + users, API keys, alert config, suppressions |

### API Design Principles

- **Thin routes** — Route handlers validate, delegate to domain modules, respond. No business logic in routes
- **Partial success** — Batch ingest: valid points stored, invalid reported as errors in response
- **Cursor pagination** — Anomaly listing uses `before=<timestamp>` for infinite scroll, `since=<timestamp>` for polling
- **Idempotent endpoints** — CSV upload, channel config PUT, and suppression POST are safe to retry

---

## 7. Security Model

### Authentication

- **JWT (HS256)** with access tokens (15-minute TTL) and refresh tokens (7-day TTL) stored in `refresh_tokens` table
- **API Keys** for machine-to-machine: SHA-256 hashed with salt `dsremo::` before storage. Plaintext never stored or logged
- JWT secret loaded from `DSREMO_JWT_SECRET` environment variable only — never in YAML or code

### Middleware Stack (Defense in Depth)

1. **PayloadLimitMiddleware** — Rejects bodies > 1 MB (upload paths have separate per-handler limits)
2. **ApiKeyMiddleware** — Validates hashed API keys; sets tenant context via `ContextVar`
3. **RateLimitMiddleware** — Per-key sliding window, 300 req/min default, returns 429 with `Retry-After`
4. **AuditLogMiddleware** — Logs every request: method, path, status, latency, key prefix. Never logs bodies or full keys
5. **CORSMiddleware** — Origins allowlist from `dsremo.yaml`

### Multi-Tenant Data Isolation

PostgreSQL FORCE RLS ensures every DB query is filtered by `app.tenant_id` session variable:
- Set by `ApiKeyMiddleware` on authenticated requests via `set_tenant(tenant_id)` ContextVar
- Even a direct DB connection with the `dsremo` user cannot bypass it
- `api_keys` table is explicitly excluded from RLS (needed for startup key loading)

### Input Validation

- All API inputs validated by Pydantic at the boundary (before any processing)
- No string interpolation in SQL queries — all parameterized via asyncpg typed parameters
- CSV upload: max 10 MB enforced before parsing
- XTCE XML parser: file size and schema validation before processing

---

## 8. Alert & Incident Management

### Alert Service

`AlertService` dispatches notifications through configurable routers:

- **WebhookRouter** — HMAC-SHA256 signed JSON payload to customer URL; 429-aware retry (3 attempts)
- **EmailRouter** — SMTP via smtplib (optional; requires SMTP config)

**Alert deduplication**: SHA-256 fingerprint of `(satellite_id, parameter, severity, detector_pattern)` — same event within cooldown window is not re-dispatched.

**Escalation**: Background task runs every 60 seconds. If a WARNING anomaly is unacknowledged for > 2 hours, it escalates to CRITICAL and re-dispatches.

### Alert Suppression Windows

Operators can suppress alerts for a satellite during planned maneuvers or maintenance:
```
POST /api/v1/satellites/{sat}/suppress
{"duration_minutes": 120, "reason": "Station-keeping burn"}
```
Alerts are silently dropped while suppression is active. No anomaly data is lost — only notifications are suppressed.

### Incident Grouping

Raw anomalies → `IncidentGrouper` → `Incident` objects:
- 5-minute correlation window (configurable)
- 1-hour auto-close window
- Severity = max of members
- Root cause summary from detector pattern matching

---

## 9. ML Detectors

### GRU Autoencoder (`autoencoder_detector.py`)

- Architecture: GRU encoder → bottleneck → feed-forward decoder (no second GRU — faster CPU inference)
- Parameters: seq_length=30, hidden=32, bottleneck=8, epochs=30
- `< 10K parameters` — trains in under 1 second on CPU per channel
- **Lazy `import torch`** inside `fit()` and `detect()` — module loads even without PyTorch installed; existing tests pass without it
- Per-channel models keyed `"satellite_id:parameter"` in `_lstm_models` dict
- Warm-start from disk checkpoint on server restart (`_model_dir` configurable)

### TCN Detector (`tcn_detector.py`)

- Architecture: Dilated causal convolutions (3 blocks, kernel_size=3, dilation doubling)
- Shared `AbstractMLDetector` ABC with GRU — both implement `fit()`, `detect()`, `save()`, `load()`
- `detector_name = "tcn"` — distinguishable in ensemble weights and explanations
- Same lazy-import, per-channel registry, and warm-start pattern as GRU

### Shared `_get_ml_model()` Factory

Both `_get_lstm_model()` and `_get_tcn_model()` delegate to a shared factory:
```python
def _get_ml_model(satellite_id, parameter, registry, factory, ext):
    key = f"{satellite_id}:{parameter}"
    if key not in registry:
        det = factory()
        path = _model_path(satellite_id, parameter, ext)
        if path is not None and path.exists():
            det.load(path)   # warm-start from checkpoint
        registry[key] = det
    return registry[key]
```
Adding a 3rd ML detector requires implementing `AbstractMLDetector`, adding a registry dict, and a one-line factory call — no other changes.

### Matrix Profile Discord Detector (`discord_detector.py`)

- Pure NumPy implementation — no additional dependencies
- Detects novel subsequences (patterns never seen in training window)
- Uses FFT-based convolution for O(n log n) distance profile computation

### Correlation Graph Detector (`correlation_detector.py`)

- Rolling Pearson correlation across all peer channels on the same satellite
- Calibrates mean/std of correlations; flags when a channel decouples from its peers
- Returns NOMINAL if fewer than 2 channels (no peers to correlate with)
- Based on STGLR research (MDPI Sensors, Jan 2025): F1 > 0.97 on correlated sensor failures

---

## 10. Code Quality

### Standards Applied

- **Type hints everywhere** — `from __future__ import annotations` on every file
- **ruff linting** with bandit security plugin — enforced in CI
- **structlog** — all logging is structured JSON in production, not print statements
- **Frozen dataclasses** for all domain models — immutability enforced at runtime

### Key Patterns Used

**DRY — Shared retry loop** (`connector.py`)
Single `_retry()` method shared by `_get()` and `_post()`. Previously, `_post()` silently swallowed 429 responses — both methods now log rate-limits identically.

**DRY — Nullable SQL parameters** (`queries.py`)
Optional filters use `$N::type IS NULL OR column = $N` — no SQL branching per optional parameter.

**DRY — Cache refresh helper** (`routes_channels.py`)
`_refresh_channel_config_cache()` called by both PUT and DELETE handlers — no 2-line duplication.

**DRY — ISO-8601 parse helper** (`routes_connectors.py`)
`_parse_iso_dt(s, field)` reused for `start` and `stop` parameters — no 7-line try/except duplication.

**DRY — Generic ML factory** (`detector.py`)
`_get_ml_model(sat, param, registry, factory, ext)` shared by GRU + TCN — identical lazy-init + warm-start pattern extracted once.

**SOLID — `AbstractMLDetector` ABC** (`base_ml_detector.py`)
GRU and TCN both implement `fit()`, `detect()`, `save()`, `load()` interface. Swappable without touching the ensemble orchestrator.

**Safe global restore** (`bulk_loader.py`)
`try/finally` block guarantees module-level globals (cooldown, thresholds, ML model registries) are always restored even when bulk detection raises an exception.

---

## 11. Test Coverage

**1042+ tests, 100% passing** (as of Sprint 19, ~110 seconds, no database required).

Test suite runs entirely in-memory using `memory_store.py` — a dict-backed stub that implements the same query interface as `queries.py`. No PostgreSQL dependency in CI.

### Test Organization

| File | Sprint | Focus |
|---|---|---|
| `test_sprint3.py` | S3 | API routes, auth, telemetry ingest |
| `test_sprint4.py` | S4 | Alert service, webhooks, escalation |
| `test_sprint5.py` | S5 | HTTP connector retries, YAMCS, InfluxDB |
| `test_sprint6.py` | S6 | Channel config, per-channel overrides |
| `test_sprint7.py` | S7 | Incidents, RBAC, STL decomposer |
| `test_sprint8.py` | S8 | XTCE parser, simulation, eval scoring |
| `test_sprint9.py` | S9 | Variance detector, FFT period, cooldown utils |
| `test_sprint10.py` | S10 | Auto-scale context window, bulk loading |
| `test_sprint11.py` | S11 | GRU autoencoder, 7-detector ensemble |
| `test_sprint13.py` | S13 | TCN detector, 8-detector ensemble |
| `test_sprint14.py` | S14 | Trend velocity, dynamic isolation forest |
| `test_sprint15.py` | S15 | Discord / matrix profile detector |
| `test_sprint16.py` | S16 | Model persistence, warm-start |
| `test_sprint17.py` | S17 | Incident grouper, hierarchical routing |
| `test_sprint18.py` | S18 | Stale data detector, MAD z-score, subsystem health |
| `test_sprint19.py` | S19 | Correlation graph, hard limits, suppression windows |
| `test_sprint25.py` | - | Code quality / DRY refactor verification |
| `test_auth.py` | - | JWT security tests |
| `test_tenant.py` | - | Multi-tenant RLS tests |
| `test_statistical.py` | - | Statistical detector edge cases |

### Test Principles

- Weight tests use `issubset()` not `==` — new detectors don't break existing weight tests
- No `SCHEMA_VERSION` hardcoding — migration count tests use `>= N`
- No demo mode removed — tests use minimal FastAPI app + `patch.object` + `dependency_overrides`

---

## 12. Benchmark Results

Validated against 4 publicly available space telemetry datasets:

| Dataset | Description | P | R | F1 |
|---|---|---|---|---|
| **OPS-SAT-AD** | ESA real spacecraft (3 channels, Oct 2023) | 60.7% | 100% | 75.6% |
| **SKAB-Valve2** | Industrial valve sensor, 9 anomaly windows | 100% | 83.3% | **90.9%** |
| **CATS (1-min)** | ESA simulation, 5M rows at 1Hz | 11.0% | 10.0% | 10.5% |
| **GECCO Water** | Water quality time-series, 430 events | 3.7% | 31.4% | 6.7% |

**Notes:**
- OPS-SAT: R=100% means every real anomaly event is detected. P=60.7% means ~40% false positive rate — acceptable for ops monitoring where misses are costly
- SKAB: F1=90.9% is competitive with supervised ML (some supervised models reach F1=0.95 on this dataset)
- CATS/GECCO: Low F1 due to massive class imbalance (>95% normal data) and high event density — the system produces raw detections well; precision requires per-customer tuning
- All results are **unsupervised** — no labeled training data used

---

## 13. Known Gaps & Next Steps

### Technical Gaps

| Gap | Priority | Notes |
|---|---|---|
| No HTTPS / TLS termination in app | High | Needs nginx/ALB in front for production |
| JWT secret rotation | Medium | Adding key versioning would improve security |
| Model training is synchronous | Medium | Should be background task for large channels |
| No rate limit by tenant tier | High | Free tier needs stricter limits than pro |
| Dashboard is vanilla JS | Low | Functional but not polished for B2C |
| No email verification on signup | High | Required before production |
| No Stripe billing integration | High | Needed for paid tiers |

### Scalability

Current single-process asyncio design handles:
- ~100 channels at 1-minute intervals comfortably
- ~10–20 channels at 1Hz (CPU-bound by STL + ML inference)
- Beyond this: horizontal scaling via running multiple workers with shared PostgreSQL

For scale-out: the stateless HTTP layer scales horizontally. ML models and calibration state are in-DB — workers share state naturally.

---

## 14. Pricing & Deployment Plan

### Tier Design

| Tier | Price | Satellites | Channels | Data | Retention | Detectors |
|---|---|---|---|---|---|---|
| **Free** | $0 | 1 | 5 | 1 MB/upload | 7 days | 6 stats only |
| **Pro** | $99/mo | 5 | 50 | 100 MB/upload | 90 days | All 12 (+ ML) |
| **Team** | $299/mo | 20 | 500 | 500 MB/upload | 1 year | All 12 + webhooks |
| **Enterprise** | Custom | Unlimited | Unlimited | Custom | Custom | All + XTCE + SLA |

### AWS Deployment Architecture (Planned)

```
Route 53 (domain)
    → CloudFront (CDN + TLS)
        → ALB (load balancer)
            → EC2 / ECS (Dsremo app)
                → RDS PostgreSQL + TimescaleDB
                    → S3 (CSV uploads, model checkpoints)
```

- **EC2**: t3.medium (2 vCPU, 4 GB RAM) — sufficient for Free/Pro tiers
- **RDS**: db.t3.medium with Multi-AZ for Pro+
- **S3**: ML model checkpoint storage, uploaded CSV archival
- **CloudFront**: Dashboard static assets at edge
- **Secrets Manager**: DB password, JWT secret, API keys

---

## Appendix: File Structure

```
src/dsremo/
├── api/
│   ├── app.py              # FastAPI factory + lifespan + middleware
│   ├── middleware.py        # PayloadLimit, ApiKey, RateLimit, AuditLog
│   ├── dependencies.py      # JWT auth dependencies
│   ├── schemas.py           # Pydantic request/response models
│   ├── routes.py            # Core telemetry + anomaly routes
│   ├── routes_auth.py       # Login, refresh, logout
│   ├── routes_alerts.py     # Alert config + history
│   ├── routes_channels.py   # Per-channel threshold config
│   ├── routes_connectors.py # YAMCS, InfluxDB, SatNOGS pull
│   ├── routes_incidents.py  # Incident listing
│   ├── routes_suppress.py   # Alert suppression windows
│   ├── routes_users.py      # User management
│   ├── routes_keys.py       # API key management
│   ├── routes_tenants.py    # Tenant management
│   ├── routes_parameters.py # XTCE parameter import
│   ├── routes_health.py     # Health check
│   └── websocket.py         # Real-time WebSocket broadcast
├── core/
│   ├── models.py            # Frozen dataclasses: Anomaly, Incident, DetectorResult
│   ├── config.py            # dynaconf config loader
│   ├── security.py          # RateLimiter, HMAC signing
│   └── tenant.py            # ContextVar-based tenant propagation
├── db/
│   ├── connection.py        # asyncpg pool management
│   ├── migrations.py        # Forward-only schema migrations (v18)
│   ├── queries.py           # All SQL — parameterized, no interpolation
│   └── memory_store.py      # In-memory stub for unit tests
├── detection/
│   ├── detector.py          # Ensemble orchestrator — the core pipeline
│   ├── calibration.py       # Self-calibrating reference distribution
│   ├── stl_decomposer.py    # STL + FFT period detection + SG fallback
│   ├── cusum.py             # CUSUM detector
│   ├── ewma.py              # EWMA detector
│   ├── statistical.py       # Z-score detector
│   ├── changepoint.py       # PELT changepoint detector
│   ├── isolation.py         # Isolation Forest
│   ├── variance_detector.py # Variance ratio detector
│   ├── trend_velocity_detector.py # STL trend acceleration
│   ├── discord_detector.py  # Matrix Profile (pure NumPy)
│   ├── correlation_detector.py # Cross-channel Pearson
│   ├── base_ml_detector.py  # AbstractMLDetector ABC
│   ├── autoencoder_detector.py # GRU Autoencoder (PyTorch)
│   ├── tcn_detector.py      # TCN (PyTorch)
│   └── incident_grouper.py  # Anomaly → Incident correlation
├── ingest/
│   ├── adapter.py           # Raw dict → typed TelemetryPoint
│   ├── connector.py         # DataConnector ABC + HTTPConnector
│   ├── csv_connector.py     # CSV bulk loader
│   ├── yamcs_connector.py   # YAMCS REST connector
│   ├── influxdb_connector.py # InfluxDB Flux connector
│   ├── satnogs_fetcher.py   # SatNOGS API connector
│   ├── esa_loader.py        # ESA mission data connector
│   ├── xtce_parser.py       # XTCE XML parameter parser
│   ├── bulk_loader.py       # Offline bulk analysis runner
│   ├── pipeline.py          # Stream ingest pipeline
│   └── utils.py             # detect_data_frequency, adaptive_cooldown_hours
├── features/
│   └── engine.py            # Feature extraction (windowing, stats)
├── alerts/
│   └── service.py           # AlertService, WebhookRouter, EmailRouter
├── explain/
│   └── explainer.py         # Root-cause grouping + causal chain
├── eval/
│   └── scoring.py           # cluster_events, score() — benchmark utilities
└── simulate/
    ├── spacecraft.py        # SpacecraftSimulator — synthetic telemetry
    └── injector.py          # Fault injection for testing
```

---

*This document represents the state of the Dsremo codebase as of Sprint 19 (March 2026). 1042+ tests passing. No external service dependencies required for the test suite. Production deployment targets AWS with PostgreSQL + TimescaleDB.*
