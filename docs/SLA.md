# Dsremo API — Service Level Objectives

## Ingest Latency (POST /api/v1/telemetry)
- **P50**: < 50ms per batch (up to 1000 points)
- **P99**: < 500ms per batch
- **Max batch size**: 10,000 points per request
- **Backpressure**: 503 returned when >50 concurrent ingest requests

## Detection Latency (streaming, per cycle)
- **P50**: < 200ms per satellite (11-detector ensemble)
- **P99**: < 2s per satellite (includes ML inference if models fitted)
- **ML training**: Offloaded to background thread pool; does not block API

## Anomaly Alert Delivery
- **Webhook**: 3 retries with exponential backoff (2s, 4s, 8s)
- **Dead-letter**: Failed alerts preserved in dead-letter queue (max 1000)
- **Cooldown**: Per-channel, per-severity (CRITICAL: 25% of base cooldown)

## Availability
- **Target**: 99.9% uptime (single-node deployment)
- **Recovery**: Stateless API + persistent DB; restart recovers full state

## Rate Limits
- **Default**: 300 requests/minute per API key
- **Configurable**: Per-tenant via dsremo.yaml or API key metadata

## Data Retention
- **Telemetry**: Configurable per tenant (default 90 days, TimescaleDB compression after 7 days)
- **Anomalies**: Indefinite (< 1% of telemetry volume)
- **Alerts**: 90 days
