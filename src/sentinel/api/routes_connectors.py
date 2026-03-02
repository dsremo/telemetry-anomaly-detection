"""Connector routes — pull telemetry from YAMCS and InfluxDB on demand.

Each endpoint accepts connection parameters, fetches data from the external
source via the existing connector modules, bulk-inserts into the DB, and
runs anomaly detection — identical pipeline to the CSV upload + analyze flow.

These endpoints let operators trigger pulls from the dashboard UI or via
scheduled cron jobs (e.g. curl every 5 min for continuous near-real-time).
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException

from sentinel.api.dependencies import require_operator
from sentinel.api.schemas import ConnectorResult, InfluxDBConnectRequest, YAMCSConnectRequest
from sentinel.db import queries
from sentinel.ingest.bulk_loader import run_bulk_detection

logger = structlog.get_logger()
connectors_router = APIRouter()


@connectors_router.post(
    "/connectors/yamcs",
    response_model=ConnectorResult,
    tags=["connectors"],
)
async def pull_yamcs(
    body: YAMCSConnectRequest,
    _user: dict = Depends(require_operator),
) -> ConnectorResult:
    """Pull archived telemetry from a YAMCS server and run anomaly detection.

    Fetches the specified parameters for the given time range, inserts them
    into the DB (idempotent — channels already at 50 K rows are skipped),
    then runs the full 6-detector pipeline over all stored channels.

    For continuous ingestion: call this endpoint periodically (e.g. via cron
    or a scheduled task) with overlapping time windows to catch new data.
    """
    from sentinel.ingest.yamcs_connector import YAMCSConnector

    # Parse optional ISO-8601 start/stop strings → datetime objects.
    start_dt = None
    stop_dt = None
    if body.start:
        try:
            from datetime import datetime, timezone
            start_dt = datetime.fromisoformat(body.start.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid start time: {body.start!r}. Use ISO-8601 format.",
            )
    if body.stop:
        try:
            from datetime import datetime, timezone
            stop_dt = datetime.fromisoformat(body.stop.replace("Z", "+00:00"))
            if stop_dt.tzinfo is None:
                stop_dt = stop_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid stop time: {body.stop!r}. Use ISO-8601 format.",
            )

    connector = YAMCSConnector(
        base_url=body.yamcs_url,
        instance=body.instance,
        parameters=body.parameters,
        satellite_id=body.satellite_id,
        start=start_dt,
        stop=stop_dt,
        api_key=body.api_key,
        subsystem=body.subsystem,
    )

    t0 = time.monotonic()
    try:
        totals = await connector.bulk_load_to_db(resample_minutes=body.resample_minutes)
    except Exception as exc:
        logger.error("yamcs_pull_failed", satellite=body.satellite_id, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=f"YAMCS connection failed: {exc}",
        )

    results = await _detect(body.satellite_id)
    elapsed = round(time.monotonic() - t0, 2)

    return ConnectorResult(
        satellite_id=body.satellite_id,
        source=connector.source_name,
        channels_loaded=len(totals),
        total_rows_inserted=sum(totals.values()),
        total_anomalies=sum(len(a) for a in results.values()),
        elapsed_s=elapsed,
    )


@connectors_router.post(
    "/connectors/influxdb",
    response_model=ConnectorResult,
    tags=["connectors"],
)
async def pull_influxdb(
    body: InfluxDBConnectRequest,
    _user: dict = Depends(require_operator),
) -> ConnectorResult:
    """Pull telemetry from InfluxDB v2 and run anomaly detection.

    Runs a Flux query for each field, inserts results into the DB, then
    runs the full 6-detector pipeline over all stored channels.

    For continuous ingestion: schedule this endpoint to run every N minutes
    with `start='-Nm'` (e.g. `start='-10m'`) to process only new data.
    """
    from sentinel.ingest.influxdb_connector import InfluxDBConnector

    connector = InfluxDBConnector(
        base_url=body.influxdb_url,
        org=body.org,
        bucket=body.bucket,
        token=body.token,
        measurement=body.measurement,
        fields=body.fields,
        satellite_id=body.satellite_id,
        start=body.start,
        stop=body.stop,
        subsystem=body.subsystem,
    )

    t0 = time.monotonic()
    try:
        totals = await connector.bulk_load_to_db(resample_minutes=body.resample_minutes)
    except Exception as exc:
        logger.error("influxdb_pull_failed", satellite=body.satellite_id, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=f"InfluxDB connection failed: {exc}",
        )

    results = await _detect(body.satellite_id)
    elapsed = round(time.monotonic() - t0, 2)

    return ConnectorResult(
        satellite_id=body.satellite_id,
        source=connector.source_name,
        channels_loaded=len(totals),
        total_rows_inserted=sum(totals.values()),
        total_anomalies=sum(len(a) for a in results.values()),
        elapsed_s=elapsed,
    )


async def _detect(satellite_id: str) -> dict:
    """Run bulk detection over all channels for a satellite. Returns results dict."""
    channel_rows = await queries.get_channel_stats(satellite_id)
    if not channel_rows:
        return {}
    parameters = [r["parameter"] for r in channel_rows]
    subsystem_map = {r["parameter"]: r["subsystem"] for r in channel_rows}
    try:
        return await run_bulk_detection(satellite_id, parameters, subsystem_map)
    except Exception as exc:
        logger.error("connector_detection_failed", satellite=satellite_id, error=str(exc))
        return {}
