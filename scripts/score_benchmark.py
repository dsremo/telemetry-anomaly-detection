"""Standalone benchmark scorer — queries DB and scores against GT windows.

Usage:
    python3 scripts/score_benchmark.py --tenant gecco-water --gt /tmp/bench_new/gecco_water/gt_windows.json
    python3 scripts/score_benchmark.py --tenant cats-spacecraft --gt /tmp/bench_isro/cats/gt_windows.json
    python3 scripts/score_benchmark.py --tenant skab-bench --gt /path/to/gt_windows.json

No API server required — queries the DB directly as the sentinel user.
Uses sentinel.eval.scoring for event-level P/R/F1 with ±window tolerance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentinel.eval.scoring import cluster_events, score


DB_CONFIG = dict(
    host="localhost",
    port=5432,
    database="sentinel",
    user="sentinel",
    password="sentinel_dev_only",
)


async def fetch_anomalies(tenant_id: str, satellite_id: str | None = None) -> list[datetime]:
    """Return all anomaly detection timestamps for a tenant (with proper RLS context)."""
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Set RLS tenant context (FORCE RLS blocks all rows without this)
        await conn.execute(
            "SELECT set_config('app.tenant_id', $1, false)", tenant_id
        )
        if satellite_id:
            rows = await conn.fetch(
                "SELECT timestamp FROM anomalies WHERE satellite_id = $1 ORDER BY timestamp",
                satellite_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT timestamp FROM anomalies ORDER BY timestamp"
            )
        return [r["timestamp"] for r in rows]
    finally:
        await conn.close()


def parse_gt(gt_path: Path) -> list[tuple[datetime, datetime]]:
    """Parse GT windows JSON: list of [start_str, end_str] pairs."""
    raw = json.loads(gt_path.read_text())
    result = []
    for pair in raw:
        start_str, end_str = pair[0], pair[1]
        # Parse with or without timezone
        for fmt in [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ]:
            try:
                start = datetime.strptime(start_str, fmt)
                end   = datetime.strptime(end_str,   fmt)
                # Normalize to UTC
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                result.append((start, end))
                break
            except ValueError:
                continue
    return result


def print_score(
    label: str,
    detected: list[datetime],
    ground_truth: list[tuple[datetime, datetime]],
    gap_s: float = 3600.0,
    window_s: float = 1800.0,
) -> None:
    result = score(detected, ground_truth, window_s=window_s, gap_s=gap_s)
    clusters = cluster_events(detected, gap_s=gap_s)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Raw detections:   {len(detected):>6}")
    print(f"  Detected events:  {result.detected_count:>6}  (gap≤{gap_s/60:.0f} min)")
    print(f"  GT windows:       {result.event_count:>6}")
    print(f"  TP: {result.tp}  FP: {result.fp}  FN: {result.fn}")
    print(f"  Precision: {result.precision:.1%}")
    print(f"  Recall:    {result.recall:.1%}")
    print(f"  F1:        {result.f1:.1%}")
    print()

    # Show earliest detection per cluster for first few events
    print("  First 5 detected events:")
    for i, cluster in enumerate(clusters[:5]):
        print(f"    [{i+1}] {cluster[0].strftime('%Y-%m-%d %H:%M')}  ({len(cluster)} alarms)")
    if len(clusters) > 5:
        print(f"    ... {len(clusters)-5} more events")
    print()

    # Show which GT windows were missed
    missed = []
    for idx, (gs, ge) in enumerate(ground_truth):
        gs_e = gs.timestamp()
        ge_e = ge.timestamp()
        reps = [c[0].timestamp() for c in clusters]
        matched = any(
            (gs_e - window_s) <= r <= (ge_e + window_s) for r in reps
        )
        if not matched:
            missed.append((idx + 1, gs, ge))

    if missed:
        print(f"  Missed GT windows ({len(missed)}):")
        for num, gs, ge in missed[:10]:
            print(f"    [#{num}] {gs.strftime('%Y-%m-%d %H:%M')} → {ge.strftime('%Y-%m-%d %H:%M')}")
        if len(missed) > 10:
            print(f"    ... {len(missed)-10} more")
    else:
        print("  All GT windows detected!")
    print()


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score anomaly detection against GT windows")
    parser.add_argument("--tenant", required=True, help="Tenant ID to score")
    parser.add_argument("--satellite-id", default=None, help="Satellite ID filter (optional)")
    parser.add_argument("--gt", required=True, type=Path, help="Path to gt_windows.json")
    parser.add_argument("--gap-min", type=float, default=60.0,
                        help="Event clustering gap in minutes (default: 60)")
    parser.add_argument("--window-min", type=float, default=30.0,
                        help="GT matching window tolerance in minutes (default: 30)")
    args = parser.parse_args(argv)

    gt_path = args.gt
    if not gt_path.exists():
        print(f"ERROR: GT file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nScoring tenant: {args.tenant}")
    print(f"GT file:        {gt_path}")
    print(f"Clustering gap: {args.gap_min} min  |  Matching window: ±{args.window_min} min")

    # Fetch detections
    detected = await fetch_anomalies(args.tenant, args.satellite_id)
    if not detected:
        print(f"\nERROR: No anomalies found for tenant '{args.tenant}' in DB.")
        print("Make sure the benchmark has been run with analyze_csv.py first.")
        sys.exit(1)

    print(f"Fetched {len(detected)} anomaly rows from DB.")

    # Parse GT
    ground_truth = parse_gt(gt_path)
    print(f"Parsed {len(ground_truth)} GT windows.")

    # Make datetimes timezone-aware
    aware = []
    for dt in detected:
        if dt.tzinfo is None:
            aware.append(dt.replace(tzinfo=timezone.utc))
        else:
            aware.append(dt)

    # Score
    print_score(
        label=f"Benchmark: {args.tenant}",
        detected=aware,
        ground_truth=ground_truth,
        gap_s=args.gap_min * 60,
        window_s=args.window_min * 60,
    )

    # Also try tighter gap for high-frequency datasets
    if args.gap_min == 60.0 and len(detected) > 5000:
        print("(Also scoring with 10-min clustering gap for high-frequency data)")
        print_score(
            label=f"Benchmark: {args.tenant}  [10-min gap]",
            detected=aware,
            ground_truth=ground_truth,
            gap_s=600,
            window_s=args.window_min * 60,
        )


if __name__ == "__main__":
    asyncio.run(main())
