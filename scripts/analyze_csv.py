"""CSV Telemetry — Full Benchmark Analysis.

Thin CLI entry-point.  All pipeline logic lives in production modules:
  sentinel.ingest.csv_connector  — CSVConnector.bulk_load_to_db()
  sentinel.ingest.bulk_loader    — run_bulk_detection(), print_detection_report()

CSV format (wide):
    timestamp,param1,param2,...
    2024-01-01T00:00:00Z,1.2,3.4,...

The timestamp column is used as the time index.  All other columns are
treated as telemetry parameters for the given satellite.

Run:
    # Basic — load telemetry.csv for satellite MYSAT-1
    python3 scripts/analyze_csv.py --file telemetry.csv --satellite-id MYSAT-1

    # With subsystem label and resampling
    python3 scripts/analyze_csv.py --file eps.csv --satellite-id MYSAT-1 \\
        --subsystem eps --resample-minutes 5

    # Custom timestamp column name
    python3 scripts/analyze_csv.py --file data.csv --satellite-id MYSAT-1 \\
        --timestamp-col time

Requires:
    Sentinel DB running (postgres).  API server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.core.config import load_config
from sentinel.db import connection as db_connection
from sentinel.detection.detector import init_detectors
from sentinel.ingest.bulk_loader import print_detection_report, run_bulk_detection
from sentinel.ingest.csv_connector import CSVConnector


async def main(
    file_path: Path,
    satellite_id: str,
    subsystem: str,
    timestamp_col: str,
    resample_minutes: int,
    skip_if_rows_gte: int,
) -> None:
    cfg = load_config(Path("configs/sentinel.yaml"))
    db = cfg.get("database", {})
    await db_connection.init_pool(
        host=db.get("host", "localhost"),
        port=db.get("port", 5432),
        database=db.get("name", "sentinel"),
        user=db.get("user", "sentinel"),
        password=db.get("password", ""),
        min_size=2,
        max_size=4,
    )
    init_detectors(cfg)

    connector = CSVConnector(
        file_path=file_path,
        satellite_id=satellite_id,
        subsystem=subsystem,
        timestamp_col=timestamp_col,
    )

    resample_label = (
        f"{resample_minutes}-min resampling" if resample_minutes > 1
        else "no resampling (raw timestamps)"
    )
    print("\n" + "=" * 65)
    print("CSV Telemetry — Sentinel Anomaly Detection")
    print("=" * 65)
    print(f"File:        {file_path}")
    print(f"Satellite:   {satellite_id}")
    print(f"Subsystem:   {subsystem}")
    print(f"Resolution:  {resample_label}")
    print(f"Skip if >=:  {skip_if_rows_gte:,} rows/channel\n")

    print("── Phase 1: Bulk Load ──────────────────────────────────────")
    t1 = time.monotonic()
    load_totals = await connector.bulk_load_to_db(
        resample_minutes=resample_minutes,
        skip_if_rows_gte=skip_if_rows_gte,
    )
    total_rows = sum(load_totals.values())
    parameters = list(load_totals.keys())
    print(f"\n  Loaded {total_rows:,} rows across {len(parameters)} channels"
          f" in {time.monotonic() - t1:.1f}s")

    if not parameters:
        print("  No channels loaded — check file format and column names.")
        await db_connection.close_pool()
        return

    print("\n── Phase 2: Streaming Detection ────────────────────────────")
    t2 = time.monotonic()
    results = await run_bulk_detection(
        satellite_id=satellite_id,
        parameters=parameters,
        subsystem_map={p: subsystem for p in parameters},
    )

    n_anomalies = sum(len(v) for v in results.values())
    print(f"\n  Detection complete: {n_anomalies} anomalies in {time.monotonic() - t2:.1f}s")

    print_detection_report(
        results,
        title=f"ANOMALY DETECTION RESULTS — {satellite_id}",
    )

    await db_connection.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load wide-format CSV telemetry and run anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Run:")[1].split("Requires:")[0].strip(),
    )
    parser.add_argument(
        "--file", required=True, type=Path,
        metavar="FILE",
        help="Path to wide-format CSV (timestamp + parameter columns)",
    )
    parser.add_argument(
        "--satellite-id", required=True,
        metavar="SAT_ID",
        help="Satellite identifier to store in DB (e.g. MYSAT-1)",
    )
    parser.add_argument(
        "--subsystem", default="unknown",
        help="Subsystem label for all parameters in this file (default: unknown)",
    )
    parser.add_argument(
        "--timestamp-col", default="timestamp",
        metavar="COL",
        help="Name of the timestamp column (default: timestamp)",
    )
    parser.add_argument(
        "--resample-minutes", type=int, default=1,
        help="Resample to N-min intervals via median (default: 1 = no resampling)",
    )
    parser.add_argument(
        "--skip-if-rows-gte", type=int, default=50_000,
        metavar="N",
        help="Skip channels that already have >= N rows (default: 50000)",
    )
    args = parser.parse_args()
    asyncio.run(main(
        file_path=args.file,
        satellite_id=args.satellite_id,
        subsystem=args.subsystem,
        timestamp_col=args.timestamp_col,
        resample_minutes=args.resample_minutes,
        skip_if_rows_gte=args.skip_if_rows_gte,
    ))
