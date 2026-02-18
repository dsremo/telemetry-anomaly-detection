"""ESA real data → Sentinel API live test.

Run:  python3 scripts/test_esa_live.py
Requires: sentinel serve running on localhost:8400
"""

import asyncio
import sys

import httpx

from sentinel.ingest.esa_loader import ESADataLoader

API_BASE = "http://localhost:8400"
BATCH_SIZE = 50
MAX_POINTS = 200
SAMPLE_RATE = 5  # every 5th point (ESA has millions per channel)


async def main() -> None:
    # --- 1. Load ESA metadata ---
    loader = ESADataLoader()
    try:
        loader.load_metadata()
    except FileNotFoundError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print(loader.summary())
    print()

    # --- 2. Check server is up ---
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{API_BASE}/api/v1/health")
            health = resp.json()
            print(f"Server health: {health['status']}")
        except httpx.ConnectError:
            print("FAIL: Cannot connect to server. Run 'sentinel serve' first.")
            sys.exit(1)

        # --- 3. Stream ESA channel data and push to API ---
        channel = "channel_1"  # EPS channel
        print(f"\nStreaming {MAX_POINTS} points from ESA {channel} (sample_rate={SAMPLE_RATE})...")

        points = list(loader.stream_channel(channel, max_points=MAX_POINTS, sample_rate=SAMPLE_RATE))
        print(f"Loaded {len(points)} points, subsystem: {points[0].subsystem}")

        accepted_total = 0
        rejected_total = 0

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
                print(f"  Batch {i // BATCH_SIZE + 1}: ERROR {resp.status_code} — {result}")
                continue
            accepted_total += result.get("accepted", 0)
            rejected_total += result.get("rejected", 0)
            print(f"  Batch {i // BATCH_SIZE + 1}: accepted={result['accepted']}, rejected={result['rejected']}")

        print(f"\nTotal: accepted={accepted_total}, rejected={rejected_total}")

        # --- 4. Query back anomalies ---
        resp = await client.get(f"{API_BASE}/api/v1/anomalies")
        anomalies = resp.json()
        print(f"\nAnomalies in DB: {len(anomalies)}")
        for a in anomalies[:3]:
            print(f"  [{a.get('severity', '?')}] {a.get('satellite_id', '?')} — {a.get('explanation', '')[:80]}")

        # --- 5. Query telemetry for the satellite ---
        resp = await client.get(f"{API_BASE}/api/v1/telemetry/ESA-MISSION1")
        telem = resp.json()
        print(f"\nStored telemetry points for ESA-MISSION1: {len(telem)}")

    print("\n--- ESA live test complete ---")


if __name__ == "__main__":
    asyncio.run(main())
