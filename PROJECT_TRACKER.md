# Sentinel — Project Tracker

## Status: Phase 1 — Scaffolding

### Completed
- [x] Project directory structure created
- [x] pyproject.toml with lean dependency list (15 deps)
- [x] CLAUDE.md project memory

### In Progress
- [ ] Domain models (core/models.py)
- [ ] Configuration system (core/config.py, configs/sentinel.yaml)
- [ ] Security primitives (core/security.py)

### Upcoming Phases
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding + config + domain models | IN PROGRESS |
| 2 | Database schema + connection + queries | PENDING |
| 3 | Telemetry adapter + API ingest endpoint | PENDING |
| 4 | Spacecraft simulator + fault injector | PENDING |
| 5 | Feature engine | PENDING |
| 6 | Anomaly detectors (statistical, isolation, changepoint) | PENDING |
| 7 | Explainer + root-cause grouping | PENDING |
| 8 | Alerts + WebSocket + security middleware | PENDING |
| 9 | Minimal dashboard | PENDING |
| 10 | Docker + tests + demo | PENDING |

### Architecture Decisions Log
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Serialization | frozen dataclasses | Immutable, stdlib, zero-dep |
| DB | PostgreSQL + asyncpg | Async, no ORM bloat, parameterized queries |
| Config | dynaconf + YAML | Layered: file → env vars → CLI |
| ML Phase 1 | Statistical methods | Works without training data, catches 80% of issues |
| Auth | API key + HMAC | Simple, auditable, no OAuth complexity for MVP |
| Frontend | Vanilla JS | Zero deps, serves from FastAPI static |
| Logging | structlog | Structured JSON, filterable, production-ready |

### Security Checklist
- [ ] API key authentication on all endpoints
- [ ] Input validation at every boundary
- [ ] Parameterized queries (no SQL injection)
- [ ] HMAC frame signing (tamper detection)
- [ ] Rate limiting per API key
- [ ] Audit logging on all requests
- [ ] No secrets in code or config files
- [ ] CORS strict whitelist
- [ ] Payload size limits
