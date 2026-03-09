# Resources Needed for Dsremo

Paste paths, links, or credentials below. Items marked with 🔴 are BLOCKING — the project needs them to run.

---

## 🔴 Datasets (Real Satellite Telemetry)

| Dataset | What It Is | Where To Get | Your Path/Link |
|---------|-----------|--------------|----------------|
| ESA OPS-SAT | Real 3U CubeSat telemetry from ESA, 18 features, labeled anomalies | https://zenodo.org/records/12528696 | `PASTE_PATH_HERE` |
| SatNOGS | Open-source satellite telemetry database (community-collected) | https://db.satnogs.org/ | `PASTE_PATH_HERE` |
| NASA PCoE | Battery + bearing degradation datasets (useful for EPS anomalies) | https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository | `PASTE_PATH_HERE` |

**How to get these:**
1. ESA OPS-SAT — Download the ZIP from Zenodo (free, no login). ~200MB.
In the Resources folder

2. SatNOGS — Use their API or export CSVs for specific satellites.
curl -H "Authorization: Token $SATNOGS_API_TOKEN" \
"https://db.satnogs.org/api/telemetry/?satellite=KPMW-9188-6390-9743-3148&limit=5"


curl -H "Authorization: Token $SATNOGS_API_TOKEN" \
"https://db.satnogs.org/api/telemetry/?satellite=AEYC-6866-6455-5236-7157&limit=5"

BEST choices (from your page)
Satellite	Why
ROBUSTA-3A	Very high frame count (~700k), academic CubeSat
Monitor-3 (RS58S)	Stable telemetry, long history
ITASAT-1	Brazilian CubeSat, clean EPS + thermal data
LEOPARD	Modern mission, good continuity


api key: Moved to `.env` as `SATNOGS_API_TOKEN` (never put tokens in git-tracked files)



3. NASA PCoE — Download battery datasets (B0005-B0018) for degradation testing.

> Until you provide real data, the simulator generates synthetic telemetry so development is not blocked.

https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

---

## 🔴 PostgreSQL Database

| Item | Value |
|------|-------|
| How to run | `docker-compose up db` (included in project) |
| Manual setup | Install PostgreSQL 15+, create database `dsremo` |

> Docker handles this automatically. No action needed unless deploying to bare metal.

---

## Optional: Domain / Deployment

| Item | Details | Your Value |
|------|---------|------------|
| Domain name | For hosting the API publicly | `PASTE_HERE` |
| Cloud provider | Any VPS with Docker support (DigitalOcean, Hetzner, Railway) | `PASTE_HERE` |
| SMTP for alerts | Gmail app password or Mailgun/SendGrid API key | `PASTE_HERE` |
| Webhook URL | Slack/Discord webhook for anomaly alerts | `PASTE_HERE` |

---

## How To Use This File

1. Download/obtain the resource
2. Replace `PASTE_PATH_HERE` or `PASTE_HERE` with the actual path or value
3. Tell me which ones you've filled in — I'll integrate them
