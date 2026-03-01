# Sentinel — Pricing

## SaaS Tiers

| | **Starter** | **Pro** | **Enterprise** |
|---|---|---|---|
| **Price** | $299/mo | $999/mo | Custom |
| **Satellites** | Up to 5 | Up to 25 | Unlimited |
| **Telemetry points/day** | 500K | 5M | Unlimited |
| **Data retention** | 90 days | 1 year | Custom |
| **Users** | 3 | 15 | Unlimited |
| **API keys** | 5 | 25 | Unlimited |
| **Alert channels** | Email | Email + Webhook | Email + Webhook + Custom |
| **Connectors** | REST + CSV | + YAMCS + InfluxDB | + Custom connectors |
| **Support** | Community | Email (48h SLA) | Dedicated + SLA |
| **Deployment** | Cloud (shared) | Cloud (shared) | Cloud (dedicated) or On-prem |
| **ITAR/Export control** | No | No | Available |
| **Custom ML models** | No | No | Available |

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

## Pilot Program

**First 10 customers get:**
- 90 days free on any tier
- Direct engineering access (Slack channel)
- Anomaly validation report against your historical data
- Input on roadmap priorities

Applies to operators with at least one operational satellite.

---

## Frequently Asked Questions

**Do I need labeled training data?**
No. Sentinel uses unsupervised detection (z-score, Isolation Forest, CUSUM, PELT, rolling variance). It learns your telemetry baseline automatically in the first 24–48 hours of operation.

**How long does onboarding take?**
30 minutes for REST API or CSV upload. A few hours for YAMCS/InfluxDB integration. We provide a sample integration script for your stack.

**Can I run it on-premises?**
Yes, for Enterprise customers. Sentinel is packaged as a single Docker Compose stack. We provide a license key and support contract.

**What's your uptime SLA?**
Pro: 99.5% monthly uptime (planned maintenance excluded).
Enterprise: 99.9% with dedicated infrastructure.

**Can I export my data?**
Yes. Full anomaly history, telemetry statistics, and alert logs are exportable via API or CSV download at any time. No lock-in.

**Is the model explainable?**
Yes. Every anomaly includes: which detectors fired, the statistical evidence (z-score, CUSUM value, etc.), a natural-language root-cause summary, and contributing parameter correlations.

**What telemetry formats do you support?**
- JSON push (REST API)
- CSV bulk upload (wide format: timestamp + one column per parameter)
- YAMCS REST API v2 (pagination, auth)
- InfluxDB Flux queries
- SatNOGS API
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

---

## Contact

For Enterprise quotes, ITAR requirements, or pilot program enrollment:

**Email:** [your email]
**Calendar:** [Calendly link]
**Demo:** Live dashboard available on request
