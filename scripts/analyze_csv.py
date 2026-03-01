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
        --subsystem eps --resample-minutes 5 --tenant my-tenant
    # Auto-adaptive cooldown (recommended for non-satellite data):
    python3 scripts/analyze_csv.py --file sensors.csv --satellite-id SKAB-1 \\
        --tenant skab-bench --auto-cooldown

Requires:
    Sentinel DB running (postgres).  API server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.core.tenant import set_tenant
from sentinel.ingest.bulk_loader import print_detection_report, run_bulk_detection
from sentinel.ingest.csv_connector import CSVConnector
from sentinel.ingest.pipeline import db_context, phase, print_run_header


def _detect_data_frequency(file_path: Path, timestamp_col: str = "timestamp") -> float:
    """Return median inter-sample interval in seconds by peeking at the first 200 rows.

    Used to scale the alert cooldown to the data frequency:
    - 1-second SCADA/industrial data → proportional to minutes
    - 5-minute monitoring data       → proportional to hours
    - 1-hour satellite telemetry     → proportional to days (original 72-360 h range)
    """
    sep = ";" if file_path.suffix == ".csv" and ";" in file_path.read_text()[:500] else ","
    df = pd.read_csv(file_path, sep=sep, parse_dates=[timestamp_col], nrows=200)
    ts = pd.to_datetime(df[timestamp_col], utc=True).sort_values()
    if len(ts) < 2:
        return 3600.0  # fallback: assume 1-hour data
    diffs = ts.diff().dropna().dt.total_seconds()
    return float(diffs.median())


def _adaptive_cooldown(median_interval_s: float) -> float:
    """Compute alert cooldown (hours) proportional to data frequency.

    Formula: cooldown = max(5 minutes, 500 × median_interval).
    - 1-second data  → 500 s  ≈ 8 min   (catches dense multi-window experiments)
    - 5-minute data  → 2500 s ≈ 42 min
    - 1-hour data    → 500 h  → capped at 72 h
    """
    cooldown_s = max(300.0, 500.0 * median_interval_s)
    cooldown_s = min(cooldown_s, 72.0 * 3600.0)   # cap at 72 h
    return cooldown_s / 3600.0


async def main(
    file_path: Path,
    satellite_id: str,
    subsystem: str,
    timestamp_col: str,
    resample_minutes: int,
    skip_if_rows_gte: int,
    tenant_id: str,
    cooldown_hours: float | None,
    recal_factor: float | None,
    auto_cooldown: bool,
) -> None:
    set_tenant(tenant_id)

    # Auto-detect data frequency and compute proportional cooldown.
    eff_cooldown = cooldown_hours
    if auto_cooldown and cooldown_hours is None:
        median_s = _detect_data_frequency(file_path, timestamp_col)
        eff_cooldown = _adaptive_cooldown(median_s)
        print(f"  [auto-cooldown] median interval={median_s:.1f}s → "
              f"cooldown={eff_cooldown*60:.0f} min ({eff_cooldown:.3f} h)")

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
            Tenant=tenant_id,
            Resolution=resample_label,
            Cooldown=f"{eff_cooldown:.2f} h" if eff_cooldown is not None else "config default",
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
                cooldown_hours=eff_cooldown,
                recal_factor=recal_factor,
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
    parser.add_argument("--tenant", type=str, default="default", metavar="TENANT_ID",
                        help="Tenant ID for data isolation (default: default)")
    parser.add_argument("--cooldown-hours", type=float, default=None, metavar="H",
                        help="Override alert cooldown in hours (default: use config value). "
                             "Scale to data frequency: ~0.1 for 1-second, 72 for hourly.")
    parser.add_argument("--recal-factor", type=float, default=None, metavar="F",
                        help="Override CUSUM recalibration sensitivity (default: config). "
                             "Higher = more stable baseline. Try 5.0–8.0 for short datasets.")
    parser.add_argument("--auto-cooldown", action="store_true",
                        help="Auto-detect data frequency and scale cooldown proportionally. "
                             "Recommended for non-satellite CSV data. Ignored if --cooldown-hours set.")
    args = parser.parse_args()
    asyncio.run(main(
        file_path=args.file,
        satellite_id=args.satellite_id,
        subsystem=args.subsystem,
        timestamp_col=args.timestamp_col,
        resample_minutes=args.resample_minutes,
        skip_if_rows_gte=args.skip_if_rows_gte,
        tenant_id=args.tenant,
        cooldown_hours=args.cooldown_hours,
        recal_factor=args.recal_factor,
        auto_cooldown=args.auto_cooldown,
    ))
