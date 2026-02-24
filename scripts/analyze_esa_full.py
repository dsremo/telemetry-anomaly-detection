"""ESA OPS-SAT — Full Benchmark Analysis.

Thin CLI entry-point.  All pipeline logic lives in production modules:
  sentinel.ingest.esa_loader  — ESADataLoader.bulk_load_channels_to_db()
  sentinel.ingest.bulk_loader — run_bulk_detection(), print_detection_report()

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
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.core.config import load_config
from sentinel.db import connection as db_connection
from sentinel.detection.detector import init_detectors
from sentinel.ingest.bulk_loader import print_detection_report, run_bulk_detection
from sentinel.ingest.esa_loader import ESADataLoader

_SATELLITE_ID = "ESA-MISSION1"
_DEFAULT_RESAMPLE_MINUTES = 60


async def main(resample_minutes: int, max_channels: int | None) -> None:
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

    loader = ESADataLoader()
    loader.load_metadata()
    channels = loader.target_channels[:max_channels] if max_channels else loader.target_channels
    summary = loader.summary()

    print("\n" + "=" * 65)
    print("ESA OPS-SAT Mission — Full Production Benchmark")
    print("=" * 65)
    print(f"Dataset:   {summary['total_channels']} channels, "
          f"{summary['target_channels']} with anomaly labels")
    print(f"Benchmark: {len(channels)} channels, {resample_minutes}-min resolution")
    print(f"Labels:    {summary['anomaly_labels']} anomalies "
          f"({summary['anomaly_categories']})")
    print(f"Pipeline:  STL → CUSUM(0.30) → EWMA(0.25) → Z-score(0.20) → PELT(0.15)\n")

    print("── Phase 1: Bulk Load ──────────────────────────────────────")
    t1 = time.monotonic()
    load_totals = await loader.bulk_load_channels_to_db(
        channels,
        satellite_id=_SATELLITE_ID,
        resample_minutes=resample_minutes,
    )
    print(f"\n  Loaded {sum(load_totals.values()):,} rows in {time.monotonic() - t1:.1f}s")

    print("\n── Phase 2: Streaming Detection ────────────────────────────")
    subsystem_map = {
        ch: loader._channels_meta[ch].subsystem
        for ch in channels
        if ch in loader._channels_meta
    }
    t2 = time.monotonic()
    results = await run_bulk_detection(
        satellite_id=_SATELLITE_ID,
        parameters=channels,
        subsystem_map=subsystem_map,
    )
    n_anomalies = sum(len(v) for v in results.values())
    print(f"\n  Detection complete: {n_anomalies} anomalies in {time.monotonic() - t2:.1f}s")

    print_detection_report(
        results,
        title="ANOMALY DETECTION RESULTS — ESA OPS-SAT MISSION1 (FULL DATA)",
        ground_truth_note=(
            "ESA ground truth: 200 labeled events across 58 target channels.\n"
            "No per-anomaly timestamps in public dataset → qualitative comparison only."
        ),
    )

    await db_connection.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESA OPS-SAT full benchmark")
    parser.add_argument(
        "--resample-minutes", type=int, default=_DEFAULT_RESAMPLE_MINUTES,
        help=f"Resample interval in minutes (default: {_DEFAULT_RESAMPLE_MINUTES})",
    )
    parser.add_argument(
        "--channels", type=int, default=None,
        help="Limit to first N channels (default: all 58)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.resample_minutes, args.channels))
