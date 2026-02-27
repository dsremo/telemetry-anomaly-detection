#!/usr/bin/env python3
"""Import XTCE parameter definitions into the Sentinel channel registry.

Usage
-----
    python scripts/import_xtce.py \\
        --file path/to/mission.xml \\
        --satellite-id MY-SAT-01

The script parses the XTCE XML file, registers every telemetry parameter
in the channel registry (channels_seen), and prints a summary table.

No telemetry data is loaded — this is metadata-only bootstrapping.
Parameters can then receive telemetry via POST /api/v1/telemetry or any
connector (YAMCS, InfluxDB, CSV) and will already be classified correctly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import XTCE parameter definitions into Sentinel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--file", "-f",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the XTCE XML file (.xml).",
    )
    p.add_argument(
        "--satellite-id", "-s",
        required=True,
        metavar="SAT_ID",
        help="Satellite identifier to associate with the parameters.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print results without writing to the database.",
    )
    return p


async def _run(xml_path: Path, satellite_id: str, dry_run: bool) -> None:
    from sentinel.ingest.xtce_parser import parse_xtce
    from sentinel.ingest.pipeline import db_context, phase, print_run_header

    print_run_header(
        "XTCE Parameter Import",
        file=str(xml_path),
        satellite_id=satellite_id,
        dry_run=dry_run,
    )

    # --- Parse XTCE (no DB needed) ---
    with phase("Parsing XTCE"):
        params = parse_xtce(xml_path)
        print(f"  {len(params)} parameters found")

    if not params:
        print("  Nothing to import. Exiting.")
        return

    # Subsystem breakdown
    by_sub: dict[str, int] = {}
    for p in params:
        by_sub[p.subsystem] = by_sub.get(p.subsystem, 0) + 1
    print("  Subsystems:")
    for sub, n in sorted(by_sub.items()):
        print(f"    {sub:<20} {n} parameters")

    if dry_run:
        # Print table and exit without touching the DB
        print("\nDry run — no database writes.\n")
        print(f"  {'Name':<40} {'Subsystem':<16} {'Unit':<8} {'Watch low':>10} {'Watch high':>10}")
        print("  " + "-" * 90)
        for p in params:
            wl = f"{p.watch_range.low:.3g}" if p.watch_range and p.watch_range.low is not None else "—"
            wh = f"{p.watch_range.high:.3g}" if p.watch_range and p.watch_range.high is not None else "—"
            print(f"  {p.name:<40} {p.subsystem:<16} {p.unit:<8} {wl:>10} {wh:>10}")
        return

    # --- Write to DB ---
    from datetime import datetime, timezone
    from sentinel.db import queries

    async with db_context():
        with phase("Registering channels"):
            now = datetime.now(timezone.utc)
            await queries.upsert_satellite_seen(satellite_id, now)
            for p in params:
                await queries.upsert_channel_seen(satellite_id, p.name, p.subsystem, p.unit)
            print(f"  {len(params)} channels registered / updated")

    print(f"\nDone. {len(params)} parameters imported for satellite '{satellite_id}'.")
    print("Telemetry can now be ingested and will be correctly classified.\n")


def main() -> None:
    args = _build_parser().parse_args()

    xml_path: Path = args.file
    if not xml_path.exists():
        print(f"Error: file not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(xml_path, args.satellite_id, args.dry_run))


if __name__ == "__main__":
    main()
