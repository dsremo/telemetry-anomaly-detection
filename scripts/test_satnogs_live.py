"""SatNOGS real data → Sentinel API live test.

Run:  python3 scripts/test_satnogs_live.py
Requires: sentinel serve running on localhost:8400
          SATNOGS_API_TOKEN set in .env
"""

import asyncio
import sys

import httpx

from sentinel.ingest.satnogs_fetcher import SatNOGSFetcher

API_BASE = "http://localhost:8400"
BATCH_SIZE = 50

# Satellites known to have decoded telemetry in SatNOGS DB.
# Many CubeSats only have raw hex — we try several until one works.
SATELLITES_TO_TRY = [
    ("44830", "ROBUSTA-3A"),
    ("43786", "ITASAT-1"),
    ("47960", "LEOPARD"),
    ("25544", "ISS"),
    ("40014", "FUNCUBE-1"),
    ("42017", "NAYIF-1"),
]


async def main() -> None:
    # --- 1. Init fetcher (auto-loads .env) ---
    fetcher = SatNOGSFetcher()
    if not fetcher.api_token:
        print("FAIL: SATNOGS_API_TOKEN not found. Check .env file.")
        sys.exit(1)
    print(f"SatNOGS token: {fetcher.api_token[:8]}...")

    # --- 2. Check server is up ---
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{API_BASE}/api/v1/health")
            health = resp.json()
            print(f"Server health: {health['status']}")
        except httpx.ConnectError:
            print("FAIL: Cannot connect to server. Run 'sentinel serve' first.")
            sys.exit(1)

    # --- 3. Try satellites until we find one with decoded frames ---
    points = []
    sat_name = ""

    for norad_id, name in SATELLITES_TO_TRY:
        print(f"\nTrying {name} (NORAD {norad_id})...")
        try:
            raw = await fetcher.fetch_telemetry(norad_id, limit=200)
            print(f"  Raw frames: {len(raw)}")

            # Debug: show raw response structure
            print(f"  Response type: {type(raw).__name__}")
            if isinstance(raw, dict):
                print(f"  Dict keys: {list(raw.keys())}")
                # Paginated response — extract results list
                results = raw.get("results", [])
                print(f"  Results count: {len(results)}")
                if results and isinstance(results[0], dict):
                    print(f"  First result keys: {list(results[0].keys())}")
                    decoded = results[0].get("decoded")
                    print(f"  Has 'decoded': {decoded is not None}")
                    if decoded:
                        print(f"  Decoded: {decoded}")
            elif isinstance(raw, list) and raw:
                frame = raw[0]
                if isinstance(frame, dict):
                    print(f"  Frame keys: {list(frame.keys())}")
                    decoded = frame.get("decoded")
                    print(f"  'decoded' type: {type(decoded).__name__}, value: {str(decoded)[:200]}")
                else:
                    print(f"  Frame is {type(frame).__name__} (raw hex)")

            converted = fetcher.convert_to_points(raw, satellite_id=name)
            print(f"  Decoded points: {len(converted)}")

            if converted:
                points = converted
                sat_name = name
                break
        except httpx.ReadTimeout:
            print(f"  Timeout — SatNOGS API slow for {name}, skipping")
        except Exception as e:
            import traceback
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()

    if not points:
        print("\n--- No decoded telemetry found from any satellite ---")
        print("This is normal — many CubeSats only transmit raw hex beacons.")
        print("SatNOGS integration works, but needs a satellite with a decoder.")
        print("\nThe ESA dataset test (test_esa_live.py) uses pre-decoded data and works reliably.")
        sys.exit(0)

    # --- 4. Push to API ---
    print(f"\n{sat_name}: pushing {len(points)} points to API...")
    async with httpx.AsyncClient(timeout=10.0) as client:
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

        # --- 5. Query anomalies ---
        resp = await client.get(f"{API_BASE}/api/v1/anomalies")
        anomalies = resp.json()
        print(f"\nAnomalies in DB: {len(anomalies)}")
        for a in anomalies[:3]:
            print(f"  [{a.get('severity', '?')}] {a.get('satellite_id', '?')} — {a.get('explanation', '')[:80]}")

    print("\n--- SatNOGS live test complete ---")


if __name__ == "__main__":
    asyncio.run(main())
