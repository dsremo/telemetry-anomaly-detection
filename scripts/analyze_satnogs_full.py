"""SatNOGS Network — Full Benchmark Analysis.

Thin CLI entry-point.  All pipeline logic lives in production modules:
  sentinel.ingest.satnogs_fetcher — SatNOGSFetcher.bulk_load_to_db()
  sentinel.ingest.bulk_loader     — run_bulk_detection(), print_detection_report()

Run:
    # ISS + two CubeSats (NORAD IDs from celestrak.org)
    python3 scripts/analyze_satnogs_full.py --norad 25544 43017 46926

    # Single satellite, larger frame cap
    python3 scripts/analyze_satnogs_full.py --norad 25544 --max-frames 5000

    # Resample sparse passes to 10-min intervals
    python3 scripts/analyze_satnogs_full.py --norad 25544 --resample-minutes 10

Requires:
    SATNOGS_API_TOKEN in .env
    Sentinel DB running (postgres).  Server does NOT need to be up.
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
from sentinel.ingest.satnogs_fetcher import SatNOGSFetcher

_PARAMETERS = ("frame_length", "byte_mean", "byte_entropy", "frame_gap")


async def main(norad_ids: list[str], max_frames: int, resample_minutes: int | None) -> None:
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

    fetcher = SatNOGSFetcher()

    resample_label = (
        f"{resample_minutes}-min resampling" if resample_minutes
        else "no resampling (raw timestamps)"
    )
    print("\n" + "=" * 65)
    print("SatNOGS Network — Full Production Benchmark")
    print("=" * 65)
    print(f"Satellites: {', '.join(norad_ids)} (NORAD IDs)")
    print(f"Max frames: {max_frames} per satellite")
    print(f"Resolution: {resample_label}")
    print(f"Parameters: {', '.join(_PARAMETERS)}")
    print(f"Pipeline:   STL → CUSUM(0.30) → EWMA(0.25) → Z-score(0.20) → PELT(0.15)\n")

    print("── Phase 1: Bulk Load ──────────────────────────────────────")
    t1 = time.monotonic()
    load_totals = await fetcher.bulk_load_to_db(
        norad_ids,
        max_frames=max_frames,
        resample_minutes=resample_minutes,
    )
    total_rows = sum(r for sat in load_totals.values() for r in sat.values())
    print(f"\n  Loaded {total_rows:,} rows in {time.monotonic() - t1:.1f}s")

    print("\n── Phase 2: Streaming Detection ────────────────────────────")
    all_results: dict[str, list] = {}
    t2 = time.monotonic()
    for norad_id in norad_ids:
        sat_id = f"SATNOGS-{norad_id}"
        if sat_id not in load_totals:
            continue
        sat_results = await run_bulk_detection(
            satellite_id=sat_id,
            parameters=list(_PARAMETERS),
            subsystem_map={p: "comms" for p in _PARAMETERS},
        )
        for param, anoms in sat_results.items():
            all_results[f"{sat_id}/{param}"] = anoms

    n_anomalies = sum(len(v) for v in all_results.values())
    print(f"\n  Detection complete: {n_anomalies} anomalies in {time.monotonic() - t2:.1f}s")

    print_detection_report(
        all_results,
        title="ANOMALY DETECTION RESULTS — SatNOGS NETWORK",
        ground_truth_note=(
            "SatNOGS data: signal-level metrics (frame size, entropy, gaps).\n"
            "No external ground truth — anomalies reflect genuine signal changes."
        ),
    )

    await db_connection.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatNOGS full benchmark analysis")
    parser.add_argument(
        "--norad", nargs="+", default=["25544"],
        metavar="NORAD_ID",
        help="NORAD catalog IDs (default: 25544 = ISS)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Max frames per satellite (default: 500). SatNOGS serves ~25 frames/page "
             "and rate-limits at ~150 frames — increase with --inter-page-delay 3",
    )
    parser.add_argument(
        "--resample-minutes", type=int, default=None,
        help="Resample to N-min intervals (default: no resampling)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.norad, args.max_frames, args.resample_minutes))
