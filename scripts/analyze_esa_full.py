"""ESA OPS-SAT — Full Benchmark Analysis.

Thin CLI wrapper.  All pipeline logic lives in production modules:
  sentinel.ingest.esa_loader  — ESADataLoader.bulk_load_channels_to_db()
  sentinel.ingest.bulk_loader — run_bulk_detection(), print_detection_report()
  sentinel.ingest.pipeline    — db_context, phase, print_run_header

Run:
    python3 scripts/analyze_esa_full.py [--resample-minutes N] [--channels N]

Requires:
    Sentinel DB running (postgres).  Server does NOT need to be up.
    Configure via SENTINEL_DB_* env vars or configs/sentinel.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.core.tenant import set_tenant
from sentinel.ingest.bulk_loader import print_detection_report, run_bulk_detection
from sentinel.ingest.esa_loader import ESADataLoader
from sentinel.ingest.pipeline import db_context, phase, print_run_header

_SATELLITE_ID = "ESA-MISSION1"
_DEFAULT_RESAMPLE_MINUTES = 60


async def main(
    resample_minutes: int,
    max_channels: int | None,
    cooldown_hours: int,
    tenant_id: str,
) -> None:
    # Set tenant context so all DB writes go to the correct tenant.
    set_tenant(tenant_id)
    # ESA historical data spans 13 years — override cooldown so the detector
    # suppresses repeated alarms during long-term aging drift.
    async with db_context(cooldown_hours=cooldown_hours):
        loader = ESADataLoader()
        loader.load_metadata()
        channels = loader.target_channels[:max_channels] if max_channels else loader.target_channels
        summary = loader.summary()

        print_run_header(
            "ESA OPS-SAT Mission — Full Production Benchmark",
            Dataset=f"{summary['total_channels']} channels, "
                    f"{summary['target_channels']} with anomaly labels",
            Benchmark=f"{len(channels)} channels, {resample_minutes}-min resolution",
            Labels=f"{summary['anomaly_labels']} anomalies "
                   f"({summary['anomaly_categories']})",
            Pipeline="STL → CUSUM(0.30) → EWMA(0.25) → Z-score(0.20) → PELT(0.15)",
        )

        with phase("Phase 1: Bulk Load"):
            load_totals = await loader.bulk_load_channels_to_db(
                channels,
                satellite_id=_SATELLITE_ID,
                resample_minutes=resample_minutes,
            )
            print(f"  {sum(load_totals.values()):,} rows, {len(channels)} channels")

        subsystem_map = {
            ch: loader._channels_meta[ch].subsystem
            for ch in channels
            if ch in loader._channels_meta
        }

        with phase("Phase 2: Streaming Detection"):
            results = await run_bulk_detection(
                satellite_id=_SATELLITE_ID,
                parameters=channels,
                subsystem_map=subsystem_map,
            )
            print(f"  {sum(len(v) for v in results.values())} anomalies")

        print_detection_report(
            results,
            title="ANOMALY DETECTION RESULTS — ESA OPS-SAT MISSION1 (FULL DATA)",
            ground_truth_note=(
                "ESA ground truth: 200 labeled events across 58 target channels.\n"
                "No per-anomaly timestamps in public dataset → qualitative comparison only."
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESA OPS-SAT full benchmark")
    parser.add_argument("--resample-minutes", type=int, default=_DEFAULT_RESAMPLE_MINUTES,
                        help=f"Resample interval in minutes (default: {_DEFAULT_RESAMPLE_MINUTES})")
    parser.add_argument("--channels", type=int, default=None,
                        help="Limit to first N channels (default: all 58)")
    parser.add_argument("--cooldown-hours", type=int, default=1440,
                        help="Min hours between anomaly reports per channel "
                             "(default: 1440 = 60d for historical data)")
    parser.add_argument("--tenant", type=str, default="esa-mission1",
                        help="Tenant ID for data isolation (default: esa-mission1)")
    args = parser.parse_args()
    asyncio.run(main(args.resample_minutes, args.channels, args.cooldown_hours, args.tenant))
