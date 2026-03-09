# Dsremo — AI Telemetry Anomaly Detection Engine
## Product Overview

---

### The Problem

Satellite operators spend hundreds of engineer-hours per month manually reviewing telemetry dashboards. Anomalies are missed until they become failures. Alerts fire on normal orbital patterns. On-call engineers are paged at 3 AM for false positives while real degradations accumulate silently.

**The cost of a missed anomaly in orbit: $50K–$500M.**

For early-stage operators — a single missed battery degradation can end a demonstration mission that took 3 years to build.

---

### What Dsremo Does

Dsremo is a **multi-tenant SaaS platform** that ingests your satellite telemetry and automatically detects anomalies in real time — with confidence scores, root-cause explanations, and alert routing built in.

No model training required. No labeled data needed. Operational in under 30 minutes.

---

### How It Works

```
Your Telemetry → Dsremo API → Feature Engine → 6-Detector Ensemble → Root Cause → Alert
     JSON / CSV / YAMCS / InfluxDB                                      Email / Webhook / SMS
```

**Detection stack (6 algorithms in consensus):**
| Detector | Catches |
|---|---|
| Z-Score | Sudden spikes, sensor faults |
| Isolation Forest | Multivariate outliers |
| CUSUM | Slow drift, gradual degradation |
| PELT (changepoint) | Mode switches, orbital transitions |
| Rolling-STD | Variance anomalies, noise increases |
| Variance Detector | Variance spikes in STL residuals (sinusoidal/oscillating channels) |

Anomalies are flagged only when **2+ detectors agree** — cutting false positives by 60–80% vs single-method approaches.

**Adaptive signal processing:**
- FFT-based period auto-detection — no manual orbital period configuration required
- STL seasonal decomposition automatically removes orbital sinusoids before detection
- Per-channel threshold overrides (operators can tune per-parameter sensitivity)
- Auto-calibrating baseline: detectors warm up from your own data, no training labels needed

---

### Validated on Real Spacecraft

**ISS (NORAD 25544) — 5,000 frames, blind detection:**
| Event | Our Detection | Ground Truth |
|---|---|---|
| 2025-10-02 ARISS SSTV Sputnik broadcast | ✓ Detected | Confirmed by NASA/AMSAT |
| 2025-10-19 Post-EVA RF power cycling | ✓ Detected | Confirmed Russian EVA Oct 16 |
| 2025-11-08 ARISS 25th anniversary SSTV | ✓ Detected | 12-image broadcast confirmed |
| 2026-01-09 Medical emergency + Dragon undocking | ✓ Detected | All 3 detectors fired |

**ESA Mission 1 — 7.1M telemetry points, 58 channels:**
- 8,795 anomalies detected across power, thermal, ADCS, and comms subsystems
- Severity breakdown: 39 critical / 6,962 warning / 1,794 watch
- 72-hour deduplication prevents alert fatigue

**Numenta Anomaly Benchmark (out-of-domain stress test):**
- 5 industrial/IoT datasets, 62K points, 15 labeled events — **default config, no tuning**
- 100% recall on HVAC fault detection (ambient temperature)
- 75% recall on machine temperature failure (early-warning credit: fires 2–7 days before failure)
- 60% recall on NYC demand anomalies (holiday/event spikes)
- False positives reduced 60% vs single-detector baseline via consensus requirement
- See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full precision/recall/F1 breakdown

---

### Key Features

**Multi-Tenancy & Security**
- Full row-level security (PostgreSQL RLS) — zero data leakage between customers
- JWT + refresh token auth, API key support, per-tenant RBAC (viewer / operator / admin)
- All secrets via env vars, parameterized SQL only, bandit security scanning in CI

**Data Connectors (plug in your stack — all accessible from the dashboard UI)**
- REST API (JSON push) — streaming ingest, detection runs automatically
- CSV upload (bulk historical data) + one-click analysis trigger
- YAMCS (flight operations standard) — enter URL + credentials in the Import tab, click Connect
- InfluxDB (metrics infrastructure) — Flux query per field, no client library needed
- SatNOGS (open ground station network) — CLI ingestion for large archival pulls
- Custom: `DataConnector` ABC — any source in <50 lines

**Alert Routing**
- Webhook (HMAC-SHA256 signed) → PagerDuty, Opsgenie, Slack
- Email (SMTP) with severity thresholds
- Per-tenant config: min_severity, dedup window, escalation delay
- Alert history + acknowledge API

**Dashboard**
- 6-tab UI: Monitor / Analysis / Channels / Alerts / Import / Admin
- Infinite scroll timeline, subsystem health matrix, severity bars
- Light / dark / system theme
- Per-channel threshold overrides
- Live Data Integrations panel (REST Push, YAMCS, InfluxDB) — connect from the browser, no CLI needed

---

### Technical Stack

| Layer | Technology |
|---|---|
| API | FastAPI (async Python 3.10+) |
| Database | PostgreSQL 16 + TimescaleDB |
| Detection | scikit-learn, ruptures, scipy |
| Auth | JWT HS256 + bcrypt |
| Observability | structlog (JSON) |
| Deployment | Single process, Docker-ready |

**Footprint:** Runs on 2 vCPU / 4 GB RAM. Scales to millions of telemetry points per day on a $50/mo VPS.

---

### For Startups: First Anomaly in 5 Minutes

If you have one satellite and one engineer, here's the full onboarding flow:

1. **Create account** — email + password in the dashboard login
2. **Upload your CSV** — drag and drop your telemetry file (timestamp + parameter columns)
3. **Click "Run Analysis"** — full 6-detector ensemble runs on your data
4. **Review anomalies** — timeline view, severity breakdown, root-cause explanations
5. **Set up alerts** — paste your Slack webhook or email address in the Alerts tab

**Total time: under 15 minutes from zero to first anomaly report.**

No engineering integration required for CSV. For live streaming, add two lines to your ground station script:

```bash
curl -X POST https://your-dsremo-host/api/v1/telemetry \
  -H "X-API-Key: stl_your_key_here" \
  -d '{"satellite_id":"SAT-1","timestamp":"...","subsystem":"eps","parameter":"battery_voltage","value":27.3}'
```

Detection runs automatically on every POST. No separate analysis step needed for real-time data.

---

### Integration Example

```bash
# Push a telemetry reading
curl -X POST https://your-dsremo-host/api/v1/telemetry \
  -H "X-API-Key: stl_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "satellite_id": "SAT-1",
    "timestamp": "2026-03-01T12:00:00Z",
    "subsystem": "eps",
    "parameter": "battery_voltage",
    "value": 27.3,
    "unit": "V"
  }'

# Query anomalies
curl https://your-dsremo-host/api/v1/anomalies?severity=critical \
  -H "X-API-Key: stl_your_key_here"
```

**Response:**
```json
{
  "satellite_id": "SAT-1",
  "parameter": "battery_voltage",
  "severity": "critical",
  "confidence": 0.91,
  "detectors_triggered": ["zscore", "cusum", "isolation_forest"],
  "explanation": "battery_voltage dropped 2.8σ below baseline (28.4V→25.6V). CUSUM negative drift S=0.34. Possible: cell degradation, load spike, or thermal event."
}
```

---

### Why Now

- **NewSpace launches doubling every 18 months** — more satellites, same number of engineers
- **Constellation operators** (100+ sats) cannot manually monitor each vehicle
- **Seed-stage startups** launching first satellites have zero ops budget for dedicated FDIR engineers — they need automation from day one
- **In-orbit servicing missions** require predictive health monitoring to prioritize targets
- **Insurance underwriters** increasingly require anomaly logs for coverage

---

### About Dsremo

Built by a team with experience in ground segment operations and ML infrastructure. Dsremo is production-ready, tested against 7M+ real telemetry points across 5 satellites, and designed from day one for multi-tenant commercial deployment.

**Contact:** [your email]
**Demo:** Available on request (live dashboard with real ISS + ESA data)
**Repo:** Private — NDA available
