"""InfluxDB — Dsremo Anomaly Detection.

Thin CLI wrapper.  All pipeline logic lives in production modules:
  dsremo.ingest.influxdb_connector — InfluxDBConnector.bulk_load_to_db()
  dsremo.ingest.bulk_loader        — run_bulk_detection(), print_detection_report()
  dsremo.ingest.pipeline           — db_context, phase, print_run_header

InfluxDB v2 Flux API: https://docs.influxdata.com/influxdb/v2/api/

Run:
    python3 scripts/analyze_influxdb.py \\
        --url http://localhost:8086 \\
        --org myorg \\
        --bucket telemetry \\
        --token mytoken \\
        --measurement satellite \\
        --fields battery_voltage panel_current temperature \\
        --satellite-id MYSAT-1

    # Custom time range (Flux duration literals or ISO-8601):
    python3 scripts/analyze_influxdb.py \\
        --url http://localhost:8086 \\
        --org myorg \\
        --bucket telemetry \\
        --token mytoken \\
        --measurement satellite \\
        --fields voltage current \\
        --satellite-id MYSAT-1 \\
        --start -90d --stop now()

Requires:
    Dsremo DB running (postgres).  API server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dsremo.ingest.bulk_loader import print_detection_report, run_bulk_detection
from dsremo.ingest.influxdb_connector import InfluxDBConnector
from dsremo.ingest.pipeline import db_context, phase, print_run_header


async def main(
    url: str,
    org: str,
    bucket: str,
    token: str,
    measurement: str,
    fields: list[str],
    satellite_id: str,
    start: str,
    stop: str,
    subsystem: str,
    resample_minutes: int,
    skip_if_rows_gte: int,
) -> None:
    async with db_context():
        connector = InfluxDBConnector(
            base_url=url,
            org=org,
            bucket=bucket,
            token=token,
            measurement=measurement,
            fields=fields,
            satellite_id=satellite_id,
            start=start,
            stop=stop,
            subsystem=subsystem,
        )

        print_run_header(
            "InfluxDB — Dsremo Anomaly Detection",
            URL=url,
            Org=org,
            Bucket=bucket,
            Measurement=measurement,
            Fields=str(len(fields)),
            Satellite=satellite_id,
        )

        with phase("Phase 1: Fetch + Bulk Load"):
            totals = await connector.bulk_load_to_db(
                resample_minutes=resample_minutes,
                skip_if_rows_gte=skip_if_rows_gte,
            )
            loaded_params = list(totals.keys())
            print(f"  {sum(totals.values()):,} rows, {len(loaded_params)} channels")

        if not loaded_params:
            print("  No fields loaded — check InfluxDB connection, org, bucket and field names.")
            return

        with phase("Phase 2: Streaming Detection"):
            results = await run_bulk_detection(
                satellite_id=satellite_id,
                parameters=loaded_params,
                subsystem_map={p: subsystem for p in loaded_params},
            )
            print(f"  {sum(len(v) for v in results.values())} anomalies")

        print_detection_report(results, title=f"ANOMALY DETECTION RESULTS — {satellite_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch InfluxDB v2 telemetry and run Dsremo anomaly detection"
    )
    parser.add_argument("--url", required=True,
                        help="InfluxDB server URL (e.g. http://localhost:8086)")
    parser.add_argument("--org", required=True,
                        help="InfluxDB organization name or ID")
    parser.add_argument("--bucket", required=True,
                        help="InfluxDB bucket name")
    parser.add_argument("--token", required=True,
                        help="InfluxDB v2 API token")
    parser.add_argument("--measurement", required=True,
                        help="InfluxDB measurement name (table equivalent)")
    parser.add_argument("--fields", required=True, nargs="+", metavar="FIELD",
                        help="Field key(s) to fetch and analyse")
    parser.add_argument("--satellite-id", required=True, metavar="SAT_ID",
                        help="Dsremo satellite identifier")
    parser.add_argument("--start", default="-30d",
                        help="Flux start time: duration ('-30d') or ISO-8601 (default: -30d)")
    parser.add_argument("--stop", default="now()",
                        help="Flux stop time: 'now()' or ISO-8601 (default: now())")
    parser.add_argument("--subsystem", default="influxdb",
                        help="Subsystem label for all fields (default: influxdb)")
    parser.add_argument("--resample-minutes", type=int, default=1,
                        help="Resample to N-min intervals via median (default: 1)")
    parser.add_argument("--skip-if-rows-gte", type=int, default=50_000, metavar="N",
                        help="Skip fields already having >= N rows (default: 50000)")
    args = parser.parse_args()

    asyncio.run(main(
        url=args.url,
        org=args.org,
        bucket=args.bucket,
        token=args.token,
        measurement=args.measurement,
        fields=args.fields,
        satellite_id=args.satellite_id,
        start=args.start,
        stop=args.stop,
        subsystem=args.subsystem,
        resample_minutes=args.resample_minutes,
        skip_if_rows_gte=args.skip_if_rows_gte,
    ))
