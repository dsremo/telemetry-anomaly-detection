"""SatNOGS Network — Full Benchmark Analysis.

Thin CLI wrapper.  All pipeline logic lives in production modules:
  dsremo.ingest.satnogs_fetcher — SatNOGSFetcher.bulk_load_to_db()
  dsremo.ingest.bulk_loader     — run_bulk_detection(), print_detection_report()
  dsremo.ingest.pipeline        — db_context, phase, print_run_header

Run:
    # ISS + two CubeSats (NORAD IDs from celestrak.org)
    python3 scripts/analyze_satnogs_full.py --norad 25544 43017 46926

    # Single satellite, larger frame cap
    python3 scripts/analyze_satnogs_full.py --norad 25544 --max-frames 5000

    # Resample sparse passes to 10-min intervals
    python3 scripts/analyze_satnogs_full.py --norad 25544 --resample-minutes 10

Requires:
    SATNOGS_API_TOKEN in .env
    Dsremo DB running (postgres).  Server does NOT need to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dsremo.core.tenant import set_tenant
from dsremo.ingest.bulk_loader import print_detection_report, run_bulk_detection
from dsremo.ingest.pipeline import db_context, phase, print_run_header
from dsremo.ingest.satnogs_fetcher import SatNOGSFetcher


async def main(
    norad_ids: list[str],
    max_frames: int,
    resample_minutes: int | None,
    tenant_id: str,
) -> None:
    set_tenant(tenant_id)
    async with db_context():
        fetcher = SatNOGSFetcher()
        resample_label = (
            f"{resample_minutes}-min resampling" if resample_minutes else "raw timestamps"
        )
        print_run_header(
            "SatNOGS Network — Full Production Benchmark",
            Satellites=", ".join(norad_ids) + " (NORAD IDs)",
            Max_frames=f"{max_frames} per satellite",
            Resolution=resample_label,
            Parameters=", ".join(SatNOGSFetcher.PARAMETERS),
            Pipeline="STL → CUSUM(0.30) → EWMA(0.25) → Z-score(0.20) → PELT(0.15)",
        )

        with phase("Phase 1: Bulk Load"):
            load_totals = await fetcher.bulk_load_to_db(
                norad_ids,
                max_frames=max_frames,
                resample_minutes=resample_minutes,
            )
            total_rows = sum(r for sat in load_totals.values() for r in sat.values())
            print(f"  {total_rows:,} rows across {len(load_totals)} satellites")

        all_results: dict[str, list] = {}
        with phase("Phase 2: Streaming Detection"):
            for norad_id in norad_ids:
                sat_id = f"SATNOGS-{norad_id}"
                if sat_id not in load_totals:
                    continue
                sat_results = await run_bulk_detection(
                    satellite_id=sat_id,
                    parameters=list(SatNOGSFetcher.PARAMETERS),
                    subsystem_map={p: "comms" for p in SatNOGSFetcher.PARAMETERS},
                )
                for param, anoms in sat_results.items():
                    all_results[f"{sat_id}/{param}"] = anoms
            print(f"  {sum(len(v) for v in all_results.values())} anomalies")

        print_detection_report(
            all_results,
            title="ANOMALY DETECTION RESULTS — SatNOGS NETWORK",
            ground_truth_note=(
                "SatNOGS data: signal-level metrics (frame size, entropy, gaps).\n"
                "No external ground truth — anomalies reflect genuine signal changes."
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatNOGS full benchmark analysis")
    parser.add_argument("--norad", nargs="+", default=["25544"], metavar="NORAD_ID",
                        help="NORAD catalog IDs (default: 25544 = ISS)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames per satellite (default: 500)")
    parser.add_argument("--resample-minutes", type=int, default=None,
                        help="Resample to N-min intervals (default: no resampling)")
    parser.add_argument("--tenant", type=str, default="satnogs",
                        help="Tenant ID for data isolation (default: satnogs)")
    args = parser.parse_args()
    asyncio.run(main(args.norad, args.max_frames, args.resample_minutes, args.tenant))
