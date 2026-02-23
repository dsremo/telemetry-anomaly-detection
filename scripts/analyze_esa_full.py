"""ESA OPS-SAT — Full Benchmark Analysis.

Bypasses the REST API entirely for production-grade throughput:

  Phase 1 — BULK LOAD
    Each of the 58 target channels is loaded from its zip file, resampled
    to 1-hour intervals (median aggregation), and inserted directly into
    PostgreSQL using UNNEST batch inserts (10 000 rows per round-trip).
    Channels already present in the DB with >= 50 000 rows are skipped.

  Phase 2 — STREAMING DETECTION
    analyze_channel_history() replays every stored row chronologically
    through the 5-detector pipeline (STL → CUSUM → EWMA → Z-score →
    PELT) without touching the REST API.  Anomalies are written to the
    anomalies table with precise historical timestamps.

  Phase 3 — REPORT
    Detected anomalies are printed sorted by confidence, with subsystem
    and date breakdowns.

Run:
    python3 scripts/analyze_esa_full.py [--resample-minutes N] [--channels N]

Requires:
    Sentinel DB running (postgres).  Server does NOT need to be up.
    Set SENTINEL_DB_* env vars or use .env file.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

# ── repo root on sys.path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sentinel.core.config import load_config
from sentinel.core.models import TelemetryPoint
from sentinel.db import connection as db_connection
from sentinel.db import queries
from sentinel.detection.detector import (
    analyze_channel_history,
    flush_all_states,
    init_detectors,
)
from sentinel.ingest.esa_loader import ESADataLoader

logger = structlog.get_logger()

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_RESAMPLE_MINUTES = 60     # 1-hour intervals → ~114 K rows/channel
DEFAULT_INSERT_BATCH     = 10_000 # rows per UNNEST call
SKIP_IF_ROWS_GTE         = 50_000 # skip channel if already has this many rows
SATELLITE_ID             = "ESA-MISSION1"


# ── bulk insert helper ────────────────────────────────────────────────────────

async def _bulk_insert_channel(
    satellite_id: str,
    channel_name: str,
    subsystem: str,
    unit: str,
    rows: pd.DataFrame,
    batch_size: int,
    pbar: tqdm,
) -> int:
    """Insert resampled channel rows directly to DB using UNNEST.

    Returns total rows accepted.
    """
    accepted = 0
    points_buf: list[TelemetryPoint] = []

    for ts, val in rows.items():
        if pd.isna(val):
            continue
        points_buf.append(TelemetryPoint(
            satellite_id=satellite_id,
            timestamp=ts.to_pydatetime(),
            subsystem=subsystem,
            parameter=channel_name,
            value=float(val),
            unit=unit,
            quality=1.0,
        ))
        if len(points_buf) >= batch_size:
            n = await queries.insert_telemetry(points_buf)
            accepted += len(points_buf)
            pbar.update(len(points_buf))
            points_buf = []

    if points_buf:
        await queries.insert_telemetry(points_buf)
        accepted += len(points_buf)
        pbar.update(len(points_buf))

    return accepted


# ── phase 1: load ─────────────────────────────────────────────────────────────

async def phase_load(
    loader: ESADataLoader,
    channels: list[str],
    resample_minutes: int,
    insert_batch: int,
) -> dict[str, int]:
    """Load ESA channels to DB.  Returns {channel: rows_inserted}."""
    resample_rule = f"{resample_minutes}min"
    totals: dict[str, int] = {}

    # Check which channels are already loaded.
    existing: dict[str, int] = {}
    for ch in channels:
        meta = loader._channels_meta.get(ch)
        if meta is None:
            continue
        rows = await queries.get_telemetry_batch_ordered(
            SATELLITE_ID, ch, limit=1
        )
        # Count existing rows.
        async with db_connection.acquire() as conn:
            cnt = await conn.fetchval(
                "SELECT COUNT(*) FROM telemetry WHERE satellite_id=$1 AND parameter=$2",
                SATELLITE_ID, ch,
            )
        existing[ch] = int(cnt)

    to_load  = [c for c in channels if existing.get(c, 0) < SKIP_IF_ROWS_GTE]
    to_skip  = [c for c in channels if existing.get(c, 0) >= SKIP_IF_ROWS_GTE]

    if to_skip:
        print(f"  Skipping {len(to_skip)} channels already loaded (>= {SKIP_IF_ROWS_GTE} rows)")
    print(f"  Loading {len(to_load)} channels at {resample_minutes}-min resolution ...\n")

    for ch in tqdm(to_load, desc="Loading channels", unit="ch"):
        meta = loader._channels_meta.get(ch)
        if meta is None:
            tqdm.write(f"  SKIP {ch} — no metadata")
            continue

        try:
            df = loader.load_channel(ch)
        except FileNotFoundError:
            tqdm.write(f"  SKIP {ch} — file not found")
            continue

        # Resample: median aggregation at the target resolution.
        # This aggregates high-frequency noise and keeps one value per interval.
        series = df.iloc[:, 0].copy()
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")

        resampled = series.resample(resample_rule).median().dropna()

        n_rows = len(resampled)
        tqdm.write(
            f"  {ch}: {len(df):>12,} raw → {n_rows:>7,} rows "
            f"@ {resample_minutes}-min  ({resample_rule})"
        )

        # Register satellite and channel.
        await queries.upsert_satellite_seen(SATELLITE_ID, resampled.index[0].to_pydatetime())
        await queries.upsert_channel_seen(SATELLITE_ID, ch, meta.subsystem, meta.unit)

        with tqdm(total=n_rows, desc=f"  {ch}", unit="pt", leave=False) as pbar:
            accepted = await _bulk_insert_channel(
                SATELLITE_ID, ch, meta.subsystem, meta.unit,
                resampled, insert_batch, pbar,
            )

        totals[ch] = accepted

    # Channels that were already present count their existing rows.
    for ch in to_skip:
        totals[ch] = existing[ch]

    return totals


# ── phase 2: detect ───────────────────────────────────────────────────────────

async def phase_detect(
    loader: ESADataLoader,
    channels: list[str],
) -> dict[str, list]:
    """Run streaming detection over all stored channels.

    Returns {channel: [Anomaly, ...]}
    """
    results: dict[str, list] = {}

    for ch in tqdm(channels, desc="Detecting", unit="ch"):
        meta = loader._channels_meta.get(ch)
        if meta is None:
            continue

        # Count stored rows first.
        async with db_connection.acquire() as conn:
            cnt = await conn.fetchval(
                "SELECT COUNT(*) FROM telemetry WHERE satellite_id=$1 AND parameter=$2",
                SATELLITE_ID, ch,
            )
        if cnt == 0:
            tqdm.write(f"  SKIP {ch} — no data in DB")
            continue

        t0 = time.monotonic()
        with tqdm(total=int(cnt), desc=f"  {ch}", unit="pt", leave=False) as pbar:
            anomalies = await analyze_channel_history(
                satellite_id=SATELLITE_ID,
                parameter=ch,
                subsystem=meta.subsystem,
                batch_size=600,
                progress_cb=lambda n, last=[0], p=pbar: (
                    p.update(n - last[0]), last.__setitem__(0, n)
                ),
            )

        elapsed = time.monotonic() - t0
        n_anom  = len(anomalies)
        if n_anom:
            tqdm.write(
                f"  {ch}: {int(cnt):>7,} pts, {n_anom} anomalies  "
                f"[{elapsed:.1f}s]"
            )
        results[ch] = anomalies

    # Persist CUSUM/EWMA/calibration states to DB.
    await flush_all_states()
    return results


# ── phase 3: report ───────────────────────────────────────────────────────────

def phase_report(results: dict[str, list], loader: ESADataLoader) -> None:
    all_anomalies = [a for anoms in results.values() for a in anoms]

    print("\n" + "=" * 65)
    print("ANOMALY DETECTION RESULTS — ESA OPS-SAT MISSION1 (FULL DATA)")
    print("=" * 65)
    print(f"\nTotal anomalies found:  {len(all_anomalies)}")

    if not all_anomalies:
        print("\n  No anomalies detected above threshold.")
        print("  Possible causes:")
        print("    • All channels still in warm-up (need 100 samples)")
        print("    • Thresholds too conservative — lower watch threshold in sentinel.yaml")
        print("    • Dataset is predominantly nominal (expected for real telemetry)")
        return

    # Sort by confidence descending.
    all_anomalies.sort(key=lambda a: a.confidence, reverse=True)

    # By severity.
    from collections import Counter
    sev_counts = Counter(a.severity.value for a in all_anomalies)
    print("\nBy severity:")
    for sev in ("critical", "warning", "watch", "nominal"):
        c = sev_counts.get(sev, 0)
        if c:
            print(f"  {sev:8s}: {c}")

    # By subsystem.
    sub_groups: dict[str, list] = {}
    for a in all_anomalies:
        sub_groups.setdefault(a.subsystem, []).append(a)
    print("\nBy subsystem:")
    for sub, anoms in sorted(sub_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {sub:8s}: {len(anoms)} anomalies")

    # Top 20 anomalies.
    print(f"\nTop {min(20, len(all_anomalies))} by confidence:")
    print(f"  {'Channel':<14} {'Severity':<10} {'Conf':>6} {'Detectors':<35} {'Timestamp'}")
    print("  " + "-" * 90)
    for a in all_anomalies[:20]:
        dets  = "+".join(a.detectors_triggered) if a.detectors_triggered else "none"
        ts    = str(a.timestamp)[:19]
        print(f"  {a.parameter:<14} {a.severity.value:<10} {a.confidence:>6.3f} "
              f"{dets:<35} {ts}")

    # Per-channel breakdown.
    print("\nPer-channel anomaly counts (channels with detections only):")
    ch_anom = {ch: anoms for ch, anoms in results.items() if anoms}
    for ch, anoms in sorted(ch_anom.items(), key=lambda x: -len(x[1])):
        timestamps = sorted(str(a.timestamp)[:10] for a in anoms)
        t_range = f"{timestamps[0]} → {timestamps[-1]}" if len(timestamps) > 1 else timestamps[0]
        top = max(anoms, key=lambda a: a.confidence)
        print(f"  {ch:<14} {len(anoms):>3} anomalies  "
              f"[{top.severity.value}/{top.confidence:.2f}]  {t_range}")

    print("\n" + "=" * 65)
    print(f"ESA ground truth: 200 labeled events across 58 target channels.")
    print(f"No per-anomaly timestamps in public dataset → qualitative comparison only.")
    print("=" * 65)


# ── main ─────────────────────────────────────────────────────────────────────

async def main(resample_minutes: int, max_channels: int | None) -> None:
    cfg = load_config(Path("configs/sentinel.yaml"))
    db  = cfg.get("database", {})
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
    channels = loader.target_channels
    if max_channels:
        channels = channels[:max_channels]

    summary = loader.summary()
    print("\n" + "=" * 65)
    print("ESA OPS-SAT Mission — Full Production Benchmark")
    print("=" * 65)
    print(f"Dataset:   {summary['total_channels']} channels, "
          f"{summary['target_channels']} with anomaly labels")
    print(f"Benchmark: {len(channels)} channels, "
          f"{resample_minutes}-min resolution")
    print(f"Labels:    {summary['anomaly_labels']} anomalies "
          f"({summary['anomaly_categories']})")
    print(f"Pipeline:  STL → CUSUM(0.30) → EWMA(0.25) → Z-score(0.20) "
          f"→ PELT(0.15)")
    print()

    # Phase 1.
    print("── Phase 1: Bulk Load ──────────────────────────────────────")
    t1 = time.monotonic()
    load_totals = await phase_load(loader, channels, resample_minutes,
                                   DEFAULT_INSERT_BATCH)
    total_rows = sum(load_totals.values())
    print(f"\n  Loaded {total_rows:,} rows in {time.monotonic()-t1:.1f}s")

    # Phase 2.
    print("\n── Phase 2: Streaming Detection ────────────────────────────")
    t2 = time.monotonic()
    detect_results = await phase_detect(loader, channels)
    n_total = sum(len(v) for v in detect_results.values())
    print(f"\n  Detection complete: {n_total} anomalies in {time.monotonic()-t2:.1f}s")

    # Phase 3.
    phase_report(detect_results, loader)

    await db_connection.close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESA full benchmark analysis")
    parser.add_argument(
        "--resample-minutes", type=int, default=DEFAULT_RESAMPLE_MINUTES,
        help=f"Resample interval in minutes (default: {DEFAULT_RESAMPLE_MINUTES})",
    )
    parser.add_argument(
        "--channels", type=int, default=None,
        help="Limit to first N channels (default: all 58)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.resample_minutes, args.channels))
