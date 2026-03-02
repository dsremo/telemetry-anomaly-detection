# Sentinel — Pricing

## SaaS Tiers

| | **Free** | **Starter** | **Pro** | **Enterprise** |
|---|---|---|---|---|
| **Price** | $0 | $299/mo | $999/mo | Custom |
| **Satellites** | 1 | Up to 5 | Up to 25 | Unlimited |
| **Telemetry points/day** | 50K | 500K | 5M | Unlimited |
| **Data retention** | 14 days | 90 days | 1 year | Custom |
| **Users** | 1 | 3 | 15 | Unlimited |
| **API keys** | 1 | 5 | 25 | Unlimited |
| **Alert channels** | None | Email | Email + Webhook | Email + Webhook + Custom |
| **Connectors** | REST + CSV | REST + CSV + YAMCS + InfluxDB | All connectors | + Custom connectors |
| **Support** | Community | Community | Email (48h SLA) | Dedicated + SLA |
| **Deployment** | Cloud (shared) | Cloud (shared) | Cloud (shared) | Cloud (dedicated) or On-prem |
| **ITAR/Export control** | No | No | No | Available |
| **Custom ML models** | No | No | No | Available |

---

## Annual Pricing (2 months free)

| Tier | Monthly | Annual |
|---|---|---|
| Starter | $299/mo | $2,990/yr |
| Pro | $999/mo | $9,990/yr |
| Enterprise | Custom | Custom |

---

## Add-Ons

| Add-On | Price |
|---|---|
| Extra satellite slot | $49/mo per satellite |
| Extra 1M telemetry points/day | $99/mo |
| XTCE/CCSDS parameter import | $199/mo |
| Historical data analysis (one-time) | $499 per 100M points |
| On-premises deployment | $2,000 setup + $500/mo support |
| ITAR-compliant isolated environment | Contact us |
| Custom connector development | $150/hr, min 10 hours |

---

## For Startups

### Free Tier (no credit card required)
The Free tier is designed for pre-revenue operators and university missions proving out their first satellite. It covers:
- 1 satellite, 50K points/day — enough for a CubeSat with 1-minute telemetry cadence
- 14-day rolling retention — access today's anomalies, not archival history
- Full 6-detector ensemble — same algorithms as Enterprise
- Dashboard access: all 6 tabs (Monitor, Analysis, Channels, Alerts, Import, Admin)
- YAMCS and InfluxDB pull connectors — enter your credentials in the Import tab, no CLI needed

**Free tier is not time-limited.** Upgrade when you add a second satellite or need longer retention.

### Startup Pilot Program
First 10 paying customers (any tier) get:
- 90 days free on any paid tier
- Direct engineering access (Slack channel with the founding team)
- Anomaly validation report against your historical data — we run your archive and send you a PDF
- Input on roadmap priorities

Applies to operators with at least one operational or manifested satellite. No revenue requirement.

### Typical startup budget scenarios

| Stage | Typical Fit | Monthly Cost |
|---|---|---|
| University / pre-launch | Free tier | $0 |
| First satellite, tight budget | Free tier → Starter | $0–$299 |
| 3–5 satellites, Series A | Starter | $299 |
| 10–25 satellites, Series B+ | Pro | $999 |
| Constellation (100+ sats) | Enterprise | Custom |

---

## Frequently Asked Questions

**Do I need labeled training data?**
No. Sentinel uses unsupervised detection (z-score, Isolation Forest, CUSUM, PELT, rolling variance, variance spike detection). It learns your telemetry baseline automatically in the first 24–48 hours of operation. No labels, no model training.

**How long does onboarding take?**
- **CSV upload**: Under 15 minutes. Upload your file from the dashboard, click "Run Analysis", get results.
- **REST API (streaming)**: 30 minutes including integration code. Detection runs automatically on every telemetry POST — no separate analysis trigger.
- **YAMCS / InfluxDB**: Under 5 minutes. Enter your server URL and credentials in the Import tab dashboard form. Click Connect. Done.

**I'm a startup with one engineer. Is this too complex for me?**
No. The dashboard is designed for single-engineer teams. Upload a CSV, see anomalies, set up an email alert. That's the full flow. We've had operators go from zero to first anomaly report in under 10 minutes.

**Can I run it on-premises?**
Yes, for Enterprise customers. Sentinel is packaged as a single Docker Compose stack. We provide a license key and support contract.

**What's your uptime SLA?**
Pro: 99.5% monthly uptime (planned maintenance excluded).
Enterprise: 99.9% with dedicated infrastructure.
Free/Starter: Best-effort (shared infrastructure).

**Can I export my data?**
Yes. Full anomaly history, telemetry statistics, and alert logs are exportable via API or CSV download at any time. No lock-in.

**Is the model explainable?**
Yes. Every anomaly includes: which detectors fired, the statistical evidence (z-score, CUSUM value, etc.), a natural-language root-cause summary, and contributing parameter correlations.

**What telemetry formats do you support?**
- JSON push (REST API)
- CSV bulk upload (wide format: timestamp + one column per parameter)
- YAMCS REST API v2 (pagination, auth) — dashboard UI or CLI
- InfluxDB Flux queries — dashboard UI or CLI
- SatNOGS API — CLI (large archival pulls, rate-limited)
- Custom connectors via `DataConnector` ABC (Python, open interface)

**Do you support CCSDS?**
We support a telemetry adapter layer — you send us engineering values (not raw CCSDS frames). Most ground systems (YAMCS, OpenMCT, etc.) can export engineering values directly.

---

## ROI Estimate

| Scenario | Without Sentinel | With Sentinel |
|---|---|---|
| Engineer-hours/month on manual telemetry review | 80 hrs × $150/hr = $12,000 | 10 hrs × $150/hr = $1,500 |
| False-positive pages per month | 40 (2/day) | 4–8 |
| Anomaly detection latency | Hours to days | Minutes |
| Early degradation catch rate | ~40% | ~85% |

**Payback on Pro tier: first month.**

For a seed-stage startup with one engineer: eliminating 20 hrs/month of manual review ($3,000 value) more than offsets the $299 Starter tier on day one.

---

## Contact

For Enterprise quotes, ITAR requirements, pilot program enrollment, or startup partnerships:

**Email:** [your email]
**Calendar:** [Calendly link]
**Demo:** Live dashboard available on request
