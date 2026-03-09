"""YAMCS — Sentinel Anomaly Detection.

Thin CLI wrapper.  All pipeline logic lives in production modules:
  sentinel.ingest.yamcs_connector  — YAMCSConnector.bulk_load_to_db()
  sentinel.ingest.bulk_loader      — run_bulk_detection(), print_detection_report()
  sentinel.ingest.pipeline         — db_context, phase, print_run_header

YAMCS REST API v2: https://docs.yamcs.org/yamcs-http-api/

Run:
    python3 scripts/analyze_yamcs.py \\
        --url http://localhost:8090 \\
        --instance simulator \\
        --parameters /YSS/SIMULATOR/BatteryVoltage /YSS/SIMULATOR/BatteryCurrent \\
        --satellite-id YAMCS-SIM

    # With authentication:
    python3 scripts/analyze_yamcs.py \\
        --url https://yamcs.example.com \\
        --instance mysat \\
        --parameters /MyMission/EPS/voltage \\
        --satellite-id MYSAT-1 \\
        --api-key YOUR_TOKEN

    # Fetch a specific time range:
    python3 scripts/analyze_yamcs.py \\
        --url http://localhost:8090 \\
        --instance sim \\
        --parameters /YSS/SIMULATOR/BatteryVoltage \\
        --satellite-id SIM-01 \\
        --start 2024-01-01T00:00:00Z --stop 2024-02-01T00:00:00Z

Requires:
    Sentinel DB running (postgres).  API server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dsremo.ingest.bulk_loader import print_detection_report, run_bulk_detection
from dsremo.ingest.pipeline import db_context, phase, print_run_header
from dsremo.ingest.yamcs_connector import YAMCSConnector


async def main(
    url: str,
    instance: str,
    parameters: list[str],
    satellite_id: str,
    api_key: str,
    start: datetime | None,
    stop: datetime | None,
    subsystem: str,
    resample_minutes: int,
    skip_if_rows_gte: int,
) -> None:
    async with db_context():
        connector = YAMCSConnector(
            base_url=url,
            instance=instance,
            parameters=parameters,
            satellite_id=satellite_id,
            start=start,
            stop=stop,
            api_key=api_key,
            subsystem=subsystem,
        )

        print_run_header(
            "YAMCS — Sentinel Anomaly Detection",
            URL=url,
            Instance=instance,
            Satellite=satellite_id,
            Parameters=str(len(parameters)),
            Subsystem=subsystem,
        )

        with phase("Phase 1: Fetch + Bulk Load"):
            totals = await connector.bulk_load_to_db(
                resample_minutes=resample_minutes,
                skip_if_rows_gte=skip_if_rows_gte,
            )
            loaded_params = list(totals.keys())
            print(f"  {sum(totals.values()):,} rows, {len(loaded_params)} channels")

        if not loaded_params:
            print("  No channels loaded — check YAMCS connection and parameter paths.")
            return

        with phase("Phase 2: Streaming Detection"):
            results = await run_bulk_detection(
                satellite_id=satellite_id,
                parameters=loaded_params,
                subsystem_map={p: subsystem for p in loaded_params},
            )
            print(f"  {sum(len(v) for v in results.values())} anomalies")

        print_detection_report(results, title=f"ANOMALY DETECTION RESULTS — {satellite_id}")


def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 string → UTC-aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch YAMCS archive parameters and run Sentinel anomaly detection"
    )
    parser.add_argument("--url", required=True,
                        help="YAMCS server URL (e.g. http://localhost:8090)")
    parser.add_argument("--instance", required=True,
                        help="YAMCS instance name (e.g. simulator)")
    parser.add_argument("--parameters", required=True, nargs="+",
                        metavar="PARAM",
                        help="Fully-qualified parameter path(s) (e.g. /YSS/SIMULATOR/BatteryVoltage)")
    parser.add_argument("--satellite-id", required=True, metavar="SAT_ID",
                        help="Sentinel satellite identifier (e.g. YAMCS-SIM)")
    parser.add_argument("--api-key", default="",
                        help="YAMCS Bearer token (optional)")
    parser.add_argument("--start", default=None, metavar="ISO8601",
                        help="Archive start time (default: 30 days ago)")
    parser.add_argument("--stop", default=None, metavar="ISO8601",
                        help="Archive stop time (default: now)")
    parser.add_argument("--subsystem", default="yamcs",
                        help="Subsystem label for all parameters (default: yamcs)")
    parser.add_argument("--resample-minutes", type=int, default=1,
                        help="Resample to N-min intervals via median (default: 1)")
    parser.add_argument("--skip-if-rows-gte", type=int, default=50_000, metavar="N",
                        help="Skip channels already having >= N rows (default: 50000)")
    args = parser.parse_args()

    asyncio.run(main(
        url=args.url,
        instance=args.instance,
        parameters=args.parameters,
        satellite_id=args.satellite_id,
        api_key=args.api_key,
        start=_parse_dt(args.start) if args.start else None,
        stop=_parse_dt(args.stop) if args.stop else None,
        subsystem=args.subsystem,
        resample_minutes=args.resample_minutes,
        skip_if_rows_gte=args.skip_if_rows_gte,
    ))
