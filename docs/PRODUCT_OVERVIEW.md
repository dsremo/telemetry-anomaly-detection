# Sentinel — AI Telemetry Anomaly Detection Engine
## Product Overview

---

### The Problem

Satellite operators spend hundreds of engineer-hours per month manually reviewing telemetry dashboards. Anomalies are missed until they become failures. Alerts fire on normal orbital patterns. On-call engineers are paged at 3 AM for false positives while real degradations accumulate silently.

**The cost of a missed anomaly in orbit: $50K–$500M.**

---

### What Sentinel Does

Sentinel is a **multi-tenant SaaS platform** that ingests your satellite telemetry and automatically detects anomalies in real time — with confidence scores, root-cause explanations, and alert routing built in.

No model training required. No labeled data needed. Operational in under 30 minutes.

---

### How It Works

```
Your Telemetry → Sentinel API → Feature Engine → 5-Detector Ensemble → Root Cause → Alert
     JSON / CSV / YAMCS / InfluxDB                                      Email / Webhook / SMS
```

**Detection stack (5 algorithms in consensus):**
| Detector | Catches |
|---|---|
| Z-Score | Sudden spikes, sensor faults |
| Isolation Forest | Multivariate outliers |
| CUSUM | Slow drift, gradual degradation |
| PELT (changepoint) | Mode switches, orbital transitions |
| Rolling-STD | Variance anomalies, noise increases |

Anomalies are flagged only when **2+ detectors agree** — cutting false positives by 60–80% vs single-method approaches.

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

**Data Connectors (plug in your stack)**
- REST API (JSON push)
- CSV upload (bulk historical data)
- YAMCS (flight operations standard)
- InfluxDB (metrics infrastructure)
- SatNOGS (open ground station network)
- Custom: `DataConnector` ABC — any source in <50 lines

**Alert Routing**
- Webhook (HMAC-SHA256 signed) → PagerDuty, Opsgenie, Slack
- Email (SMTP) with severity thresholds
- Per-tenant config: min_severity, dedup window, escalation delay
- Alert history + acknowledge API

**Dashboard**
- 5-tab UI: Monitor / Analysis / Channels / Alerts / Admin
- Infinite scroll timeline, subsystem health matrix, severity bars
- Light / dark / system theme
- Per-channel threshold overrides

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

### Integration Example

```bash
# Push a telemetry reading
curl -X POST https://your-sentinel-host/api/v1/telemetry \
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
curl https://your-sentinel-host/api/v1/anomalies?severity=critical \
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
- **In-orbit servicing missions** require predictive health monitoring to prioritize targets
- **Insurance underwriters** increasingly require anomaly logs for coverage

---

### About Sentinel

Built by a team with experience in ground segment operations and ML infrastructure. Sentinel is production-ready, tested against 7M+ real telemetry points across 5 satellites, and designed from day one for multi-tenant commercial deployment.

**Contact:** [your email]
**Demo:** Available on request (live dashboard with real ISS + ESA data)
**Repo:** Private — NDA available
