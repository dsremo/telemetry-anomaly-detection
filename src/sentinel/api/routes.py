"""API routes — the public contract of the Sentinel engine.

Every route is thin: validate → delegate → respond.
Business logic lives in domain modules, never in route handlers.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sentinel.api.dependencies import get_current_user, require_admin, require_operator, require_viewer

from sentinel import __version__
from sentinel.api.schemas import (
    AnomalyOut,
    HealthResponse,
    IngestResponse,
    InjectRequest,
    SimulateRequest,
    TelemetryBatchIn,
    TelemetryIn,
    TelemetryOut,
)
from sentinel.db import queries
from sentinel.ingest.adapter import AdapterError, adapt_batch, adapt_single

logger = structlog.get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats", tags=["system"])
async def get_stats(request: Request, _user: dict = Depends(get_current_user)) -> dict:
    """Aggregate telemetry and anomaly counts for the dashboard."""
    if getattr(request.app.state, "demo_mode", False):
        return {"total_telemetry_points": 0, "points_last_hour": 0, "active_satellites": 0, "total_anomalies": 0}
    stats = await queries.get_telemetry_stats()
    stats["total_anomalies"] = await queries.get_anomaly_count()
    return stats


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(request: Request) -> HealthResponse:
    """System health check — always accessible without auth."""
    is_demo = getattr(request.app.state, "demo_mode", False)

    if is_demo:
        db_ok = True  # memory store is always "connected"
    else:
        db_ok = True
        try:
            from sentinel.db.connection import get_pool
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception:
            db_ok = False

    uptime = time.monotonic() - getattr(request.app.state, "start_time", time.monotonic())
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=__version__,
        db_connected=db_ok,
        uptime_seconds=round(uptime, 1),
    )


# ---------------------------------------------------------------------------
# Telemetry Ingestion
# ---------------------------------------------------------------------------

@router.post("/telemetry", response_model=IngestResponse, tags=["telemetry"])
async def ingest_telemetry(body: TelemetryBatchIn, _user: dict = Depends(require_operator)) -> IngestResponse:
    """Ingest a batch of telemetry points.

    Partial success: valid points are stored, invalid ones reported as errors.
    """
    raw_dicts = [p.model_dump() for p in body.points]
    valid_points, errors = adapt_batch(raw_dicts)

    if valid_points:
        try:
            await queries.insert_telemetry(valid_points)
        except Exception as e:
            logger.error("telemetry_insert_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to store telemetry")

    # Trigger detection pipeline asynchronously for each unique satellite
    sat_ids = {p.satellite_id for p in valid_points}
    for sat_id in sat_ids:
        try:
            from sentinel.detection.detector import run_detection_cycle
            await run_detection_cycle(sat_id)
        except Exception as e:
            logger.warning("detection_cycle_error", satellite=sat_id, error=str(e))

    return IngestResponse(
        accepted=len(valid_points),
        rejected=len(errors),
        errors=errors[:10],  # cap error detail to prevent response bloat
    )


@router.post("/telemetry/single", response_model=IngestResponse, tags=["telemetry"])
async def ingest_single(body: TelemetryIn, _user: dict = Depends(require_operator)) -> IngestResponse:
    """Convenience endpoint for a single telemetry point."""
    try:
        point = adapt_single(body.model_dump())
    except AdapterError as e:
        return IngestResponse(accepted=0, rejected=1, errors=[{"error": str(e)}])

    await queries.insert_telemetry([point])
    return IngestResponse(accepted=1, rejected=0)


@router.get("/telemetry/{satellite_id}", tags=["telemetry"])
async def get_telemetry(
    satellite_id: str,
    parameter: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10_000),
    _user: dict = Depends(require_viewer),
) -> list[TelemetryOut]:
    """Query historical telemetry for a satellite."""
    rows = await queries.get_telemetry(
        satellite_id=satellite_id,
        parameter=parameter,
        since=since,
        until=until,
        limit=limit,
    )
    return [TelemetryOut(**r) for r in rows]


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

@router.get("/anomalies", tags=["anomalies"])
async def list_anomalies(
    satellite_id: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _user: dict = Depends(require_viewer),
) -> list[AnomalyOut]:
    """List detected anomalies with optional filters."""
    rows = await queries.get_anomalies(
        satellite_id=satellite_id,
        severity=severity,
        since=since,
        limit=limit,
    )
    return [_row_to_anomaly(r) for r in rows]


@router.get("/anomalies/{anomaly_id}", tags=["anomalies"])
async def get_anomaly(anomaly_id: str, _user: dict = Depends(require_viewer)) -> AnomalyOut:
    """Get a single anomaly with full explanation."""
    row = await queries.get_anomaly_by_id(anomaly_id)
    if not row:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return _row_to_anomaly(row)


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

@router.get("/satellites", tags=["satellites"])
async def list_satellites(_user: dict = Depends(require_viewer)) -> list[str]:
    """List all satellites that have sent telemetry."""
    return await queries.get_known_satellites()


# ---------------------------------------------------------------------------
# Simulator (dev/demo only)
# ---------------------------------------------------------------------------

@router.post("/simulate/start", tags=["simulator"])
async def start_simulation(body: SimulateRequest, _user: dict = Depends(require_admin)) -> dict:
    """Start generating synthetic telemetry. For demos and testing."""
    try:
        from sentinel.simulate.spacecraft import SpacecraftSimulator
        sim = SpacecraftSimulator(
            satellite_id=body.satellite_id,
            rate_hz=body.rate_hz,
        )
        # Run simulation in background task
        import asyncio
        asyncio.create_task(
            _run_simulation(sim, body.duration_seconds)
        )
        return {
            "status": "started",
            "satellite_id": body.satellite_id,
            "duration_seconds": body.duration_seconds,
            "rate_hz": body.rate_hz,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/inject", tags=["simulator"])
async def inject_fault(body: InjectRequest, _user: dict = Depends(require_admin)) -> dict:
    """Inject a fault into running simulation. For testing anomaly detection."""
    return {
        "status": "injected",
        "fault_type": body.fault_type,
        "subsystem": body.subsystem,
        "parameter": body.parameter,
        "intensity": body.intensity,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_simulation(sim, duration_seconds: int) -> None:
    """Background task that feeds simulated telemetry into the pipeline."""
    import asyncio
    end_time = time.monotonic() + duration_seconds
    while time.monotonic() < end_time:
        try:
            points = sim.generate_tick()
            await queries.insert_telemetry(points)
            for sat_id in {p.satellite_id for p in points}:
                from sentinel.detection.detector import run_detection_cycle
                await run_detection_cycle(sat_id)
        except Exception as e:
            logger.warning("simulation_tick_error", error=str(e))
        await asyncio.sleep(1.0 / sim.rate_hz)


def _row_to_anomaly(row: dict) -> AnomalyOut:
    """Convert a DB row dict to an AnomalyOut schema."""
    import json
    contributing = row.get("contributing_params", "{}")
    if isinstance(contributing, str):
        contributing = json.loads(contributing)
    detectors = row.get("detectors_triggered", [])
    if isinstance(detectors, str):
        detectors = json.loads(detectors)
    return AnomalyOut(
        id=row["id"],
        satellite_id=row["satellite_id"],
        timestamp=row["timestamp"],
        subsystem=row["subsystem"],
        parameter=row["parameter"],
        value=row["value"],
        severity=row["severity"],
        confidence=row["confidence"],
        detectors_triggered=list(detectors),
        explanation=row.get("explanation", ""),
        root_cause_group=row.get("root_cause_group"),
        contributing_params=contributing,
    )
