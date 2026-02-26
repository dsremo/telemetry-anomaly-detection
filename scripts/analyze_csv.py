"""CSV Telemetry — Full Benchmark Analysis.

Thin CLI wrapper.  All pipeline logic lives in production modules:
  sentinel.ingest.csv_connector — CSVConnector.bulk_load_to_db()
  sentinel.ingest.bulk_loader   — run_bulk_detection(), print_detection_report()
  sentinel.ingest.pipeline      — db_context, phase, print_run_header

CSV format (wide):
    timestamp,param1,param2,...
    2024-01-01T00:00:00Z,1.2,3.4,...

Run:
    python3 scripts/analyze_csv.py --file telemetry.csv --satellite-id MYSAT-1
    python3 scripts/analyze_csv.py --file eps.csv --satellite-id MYSAT-1 \\
        --subsystem eps --resample-minutes 5

Requires:
    Sentinel DB running (postgres).  API server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.ingest.bulk_loader import print_detection_report, run_bulk_detection
from sentinel.ingest.csv_connector import CSVConnector
from sentinel.ingest.pipeline import db_context, phase, print_run_header


async def main(
    file_path: Path,
    satellite_id: str,
    subsystem: str,
    timestamp_col: str,
    resample_minutes: int,
    skip_if_rows_gte: int,
) -> None:
    async with db_context():
        connector = CSVConnector(file_path, satellite_id, subsystem, timestamp_col)
        resample_label = (
            f"{resample_minutes}-min resampling" if resample_minutes > 1 else "raw timestamps"
        )
        print_run_header(
            "CSV Telemetry — Sentinel Anomaly Detection",
            File=str(file_path),
            Satellite=satellite_id,
            Subsystem=subsystem,
            Resolution=resample_label,
            Skip_if_gte=f"{skip_if_rows_gte:,} rows/channel",
        )

        with phase("Phase 1: Bulk Load"):
            totals = await connector.bulk_load_to_db(
                resample_minutes=resample_minutes,
                skip_if_rows_gte=skip_if_rows_gte,
            )
            parameters = list(totals.keys())
            print(f"  {sum(totals.values()):,} rows, {len(parameters)} channels")

        if not parameters:
            print("  No channels loaded — check file format and column names.")
            return

        with phase("Phase 2: Streaming Detection"):
            results = await run_bulk_detection(
                satellite_id=satellite_id,
                parameters=parameters,
                subsystem_map={p: subsystem for p in parameters},
            )
            print(f"  {sum(len(v) for v in results.values())} anomalies")

        print_detection_report(results, title=f"ANOMALY DETECTION RESULTS — {satellite_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load wide-format CSV telemetry and run anomaly detection",
    )
    parser.add_argument("--file", required=True, type=Path, metavar="FILE",
                        help="Wide-format CSV (timestamp + parameter columns)")
    parser.add_argument("--satellite-id", required=True, metavar="SAT_ID",
                        help="Satellite identifier (e.g. MYSAT-1)")
    parser.add_argument("--subsystem", default="unknown",
                        help="Subsystem label for all columns (default: unknown)")
    parser.add_argument("--timestamp-col", default="timestamp", metavar="COL",
                        help="Timestamp column name (default: timestamp)")
    parser.add_argument("--resample-minutes", type=int, default=1,
                        help="Resample to N-min intervals via median (default: 1 = no resample)")
    parser.add_argument("--skip-if-rows-gte", type=int, default=50_000, metavar="N",
                        help="Skip channels already having >= N rows (default: 50000)")
    args = parser.parse_args()
    asyncio.run(main(
        file_path=args.file,
        satellite_id=args.satellite_id,
        subsystem=args.subsystem,
        timestamp_col=args.timestamp_col,
        resample_minutes=args.resample_minutes,
        skip_if_rows_gte=args.skip_if_rows_gte,
    ))
