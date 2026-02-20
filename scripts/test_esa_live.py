"""ESA real satellite data → Sentinel API comprehensive test.

Tests all 4 subsystems using real ESA OPS-SAT mission data:
  - EPS (subsystem_1/2): channel_61, channel_1
  - ADCS (subsystem_3/4): channel_70, channel_73
  - Thermal (subsystem_5): channel_41, channel_43
  - Comms (subsystem_6): channel_12, channel_15

Run:  python3 scripts/test_esa_live.py
Requires: sentinel serve running on localhost:8400
"""

import asyncio
import sys
import time

import httpx

from sentinel.ingest.esa_loader import ESADataLoader

API_BASE = "http://localhost:8400"
BATCH_SIZE = 50
POINTS_PER_CHANNEL = 100
SAMPLE_RATE = 50  # every 50th point — ESA channels have millions

# One representative target channel per subsystem
TEST_CHANNELS = [
    ("channel_61", "eps"),       # subsystem_1 — target channel with anomalies
    ("channel_1", "eps"),        # subsystem_1 — non-target, baseline
    ("channel_70", "adcs"),      # subsystem_3 — target
    ("channel_73", "adcs"),      # subsystem_4 — target
    ("channel_41", "thermal"),   # subsystem_5 — target
    ("channel_43", "thermal"),   # subsystem_5 — target
    ("channel_12", "comms"),     # subsystem_6 — target
    ("channel_15", "comms"),     # subsystem_6 — target
]


async def push_points(client: httpx.AsyncClient, points: list) -> tuple[int, int]:
    """Push telemetry points to API in batches. Returns (accepted, rejected)."""
    accepted = 0
    rejected = 0

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        payload = [
            {
                "satellite_id": p.satellite_id,
                "timestamp": p.timestamp.isoformat(),
                "subsystem": p.subsystem,
                "parameter": p.parameter,
                "value": p.value,
                "unit": p.unit,
            }
            for p in batch
        ]
        resp = await client.post(f"{API_BASE}/api/v1/telemetry", json={"points": payload})
        result = resp.json()
        if resp.status_code != 200:
            print(f"    ERROR {resp.status_code} — {result}")
            continue
        accepted += result.get("accepted", 0)
        rejected += result.get("rejected", 0)

    return accepted, rejected


async def main() -> None:
    # --- 1. Load ESA metadata ---
    loader = ESADataLoader()
    try:
        loader.load_metadata()
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    summary = loader.summary()
    print(f"ESA Dataset: {summary['total_channels']} channels, {summary['target_channels']} with anomaly labels")
    print(f"Subsystems: {summary['subsystems']}")
    print(f"Anomaly labels: {summary['anomaly_labels']} ({summary['anomaly_categories']})")
    print()

    # --- 2. Check server is up ---
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{API_BASE}/api/v1/health")
            health = resp.json()
            print(f"Server: {health['status']} (v{health.get('version', '?')})")
        except httpx.ConnectError:
            print("FAIL: Cannot connect to server. Run 'sentinel serve' first.")
            sys.exit(1)

        # --- 3. Stream each channel and push to API ---
        total_accepted = 0
        total_rejected = 0
        start = time.time()

        print(f"\nPushing {POINTS_PER_CHANNEL} points per channel (sample_rate={SAMPLE_RATE}):\n")

        for channel_name, expected_subsystem in TEST_CHANNELS:
            try:
                points = list(loader.stream_channel(
                    channel_name,
                    max_points=POINTS_PER_CHANNEL,
                    sample_rate=SAMPLE_RATE,
                ))
            except FileNotFoundError:
                print(f"  {channel_name:12s} [{expected_subsystem:8s}]  SKIP — not extracted")
                continue

            actual_sub = points[0].subsystem if points else "?"
            accepted, rejected = await push_points(client, points)
            total_accepted += accepted
            total_rejected += rejected

            status = "OK" if rejected == 0 else f"WARN ({rejected} rejected)"
            print(f"  {channel_name:12s} [{actual_sub:8s}]  {len(points):4d} pts → accepted={accepted}, rejected={rejected}  {status}")

        elapsed = time.time() - start
        print(f"\nTotal: {total_accepted} accepted, {total_rejected} rejected in {elapsed:.1f}s")

        # --- 4. Query anomalies ---
        resp = await client.get(f"{API_BASE}/api/v1/anomalies?limit=100")
        anomalies = resp.json()
        print(f"\nAnomalies in DB: {len(anomalies)}")

        # Group by severity
        by_severity = {}
        for a in anomalies:
            sev = a.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        print(f"  By severity: {by_severity}")

        # Group by subsystem
        by_sub = {}
        for a in anomalies:
            sub = a.get("subsystem", "unknown")
            by_sub[sub] = by_sub.get(sub, 0) + 1
        print(f"  By subsystem: {by_sub}")

        # Show top 5
        print(f"\n  Latest anomalies:")
        for a in anomalies[:5]:
            print(f"    [{a['severity']:8s}] {a['parameter']:15s} = {a['value']:.4f}  conf={a['confidence']:.1%}")

        # --- 5. Satellites ---
        resp = await client.get(f"{API_BASE}/api/v1/satellites")
        satellites = resp.json()
        print(f"\nSatellites: {satellites}")

        # --- 6. Stored telemetry ---
        resp = await client.get(f"{API_BASE}/api/v1/telemetry/ESA-MISSION1")
        telem = resp.json()
        print(f"Stored telemetry for ESA-MISSION1: {len(telem)} points")

    print("\n--- ESA comprehensive test complete ---")


if __name__ == "__main__":
    asyncio.run(main())
