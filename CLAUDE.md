# Sentinel — AI Telemetry Anomaly Detection Engine

## What This Is
Ground-based satellite telemetry anomaly detection. Customers push JSON telemetry → we detect anomalies and explain them.

## Architecture
`Telemetry → API → DB → Feature Engine → Anomaly Model → Alert Service → Dashboard/API`

## Key Rules
- No CCSDS protocol implementation — telemetry adapter layer only
- Statistical detection first (z-score, Isolation Forest, PELT). No deep learning in MVP.
- PostgreSQL + TimescaleDB for storage. No Redis, Kafka, or distributed infra.
- All secrets via environment variables. Never in config files or code.
- Parameterized SQL queries only. No string interpolation in queries.
- Every API input validated at the boundary. Reject before processing.
- Frozen dataclasses for domain models. Immutability by default.

## Code Standards
- Python 3.10+, type hints everywhere
- `ruff` for linting (bandit security checks enabled)
- `structlog` for all logging (structured JSON in production)
- `asyncpg` for DB (no ORM)
- Tests: pytest + pytest-asyncio, target >85% coverage

## Project Layout
- `src/sentinel/` — main package
- `configs/` — YAML configuration
- `tests/` — unit + integration + security tests
- `dashboard/` — minimal vanilla JS UI
- `scripts/` — utilities (seed data, etc.)
