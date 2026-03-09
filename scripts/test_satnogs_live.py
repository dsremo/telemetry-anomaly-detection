"""SatNOGS real satellite data → Dsremo API live test.

Fetches raw frames from the SatNOGS network and extracts signal-level
metrics (frame length, byte entropy, inter-frame gaps) that flow through
our anomaly detection pipeline.

Run:  python3 scripts/test_satnogs_live.py
Requires: dsremo serve running on localhost:8400
          SATNOGS_API_TOKEN set in .env
"""

import asyncio
import sys
import time

import httpx

# Delay between satellite API calls to avoid SatNOGS rate limiting
# SatNOGS enforces ~1 req/5s per token — 6s gives comfortable headroom
INTER_SAT_DELAY_S = 6.0

from dsremo.ingest.satnogs_fetcher import SatNOGSFetcher

API_BASE = "http://localhost:8400"
BATCH_SIZE = 50
LOGIN_EMAIL = "admin@satnogs-net.com"
LOGIN_PASSWORD = "AdminPass123"
TENANT_ID = "satnogs"

# Active satellites with high frame counts on SatNOGS DB
# Source: https://db.satnogs.org/ — "Latest Data" section
# Note: kept to 2 satellites to stay within SatNOGS API rate limits
SATELLITES_TO_TRY = [
    ("35933", "BEESAT"),        # 6.6M frames, TU Berlin CubeSat, active
    ("39446", "UWE-3"),         # 131K frames, Uni Würzburg, active
    ("40014", "FUNCUBE-1"),     # FUNcube, educational sat
    ("44830", "ROBUSTA-3A"),    # French academic CubeSat
]

# Frames per satellite — SatNOGS returns up to 500 per page
FRAMES_PER_SAT = 100


async def get_auth_token() -> str:
    """Login and return a fresh JWT access token."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{API_BASE}/api/v1/auth/login",
            json={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD, "tenant_id": TENANT_ID},
        )
        if resp.status_code != 200:
            print(f"WARN: Login failed ({resp.status_code}) — requests may be rejected with 401")
            return ""
        return resp.json()["access_token"]


async def main() -> None:
    # --- 1. Init fetcher ---
    fetcher = SatNOGSFetcher()
    if not fetcher.api_token:
        print("FAIL: SATNOGS_API_TOKEN not found. Check .env file.")
        sys.exit(1)
    print(f"SatNOGS token: {fetcher.api_token[:8]}...")

    # --- 1b. Auth ---
    token = await get_auth_token()
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
    if token:
        print(f"Auth:     JWT Bearer ({LOGIN_EMAIL})")

    # --- 2. Check server ---
    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers) as client:
        try:
            resp = await client.get(f"{API_BASE}/api/v1/health")
            health = resp.json()
            print(f"Server: {health['status']} (v{health.get('version', '?')})")
        except httpx.ConnectError:
            print("FAIL: Cannot connect to server. Run 'dsremo serve' first.")
            sys.exit(1)

    # --- 3. Fetch and convert from each satellite ---
    all_points = []
    start = time.time()

    for norad_id, name in SATELLITES_TO_TRY:
        print(f"\n{name} (NORAD {norad_id}):")
        try:
            raw = await fetcher.fetch_telemetry(norad_id, limit=FRAMES_PER_SAT)
            print(f"  Raw frames: {len(raw)}")

            if not raw:
                print("  No frames available")
                continue

            # Show sample frame info
            frame = raw[0] if isinstance(raw[0], dict) else None
            if frame:
                hex_data = frame.get("frame", "")
                ts = frame.get("timestamp", "")
                observer = frame.get("observer", "")
                print(f"  Latest: {ts} | observer={observer} | hex_len={len(hex_data)}")

            points = fetcher.convert_to_points(raw, satellite_id=name)
            print(f"  Extracted {len(points)} signal metrics")

            if points:
                # Show sample metrics
                params = set(p.parameter for p in points)
                print(f"  Parameters: {params}")
                for p in points[:3]:
                    print(f"    {p.parameter}: {p.value} {p.unit}")
                all_points.extend(points)

        except httpx.ReadTimeout:
            print(f"  Timeout — skipping")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

        # Brief pause to avoid SatNOGS API rate limiting
        await asyncio.sleep(INTER_SAT_DELAY_S)

    elapsed = time.time() - start
    print(f"\nFetched {len(all_points)} total signal metrics from SatNOGS in {elapsed:.1f}s")

    if not all_points:
        print("\nNo data extracted. Check your SatNOGS API token and internet connection.")
        sys.exit(1)

    # --- 4. Push to API ---
    print(f"\nPushing {len(all_points)} points to Dsremo API...")
    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers) as client:
        accepted_total = 0
        rejected_total = 0

        for i in range(0, len(all_points), BATCH_SIZE):
            batch = all_points[i : i + BATCH_SIZE]
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

        print(f"  Accepted: {accepted_total}, Rejected: {rejected_total}")

        # --- 5. Query anomalies (SatNOGS satellites only) ---
        satnogs_names = {s[1] for s in SATELLITES_TO_TRY}
        resp = await client.get(f"{API_BASE}/api/v1/anomalies?limit=200")
        anomalies = resp.json()
        satnogs_anomalies = [a for a in anomalies if a.get("satellite_id") in satnogs_names]

        print(f"\nSatNOGS anomalies detected: {len(satnogs_anomalies)}")
        if satnogs_anomalies:
            by_sat: dict[str, list] = {}
            for a in satnogs_anomalies:
                by_sat.setdefault(a["satellite_id"], []).append(a)
            for sat, sat_anoms in by_sat.items():
                sevs = [a["severity"] for a in sat_anoms]
                print(f"  {sat}: {len(sat_anoms)} anomalies  (critical={sevs.count('critical')}, warning={sevs.count('warning')})")
                for a in sat_anoms[:3]:
                    print(f"    [{a['severity']:8s}] {a['parameter']:15s} = {a['value']:.4f}  conf={a.get('confidence', 0)*100:.1f}%")
        else:
            print("  None — signal metrics nominal (satellite operating normally)")

        # --- 6. Satellites ---
        resp = await client.get(f"{API_BASE}/api/v1/satellites")
        satellites = resp.json()
        print(f"\nSatellites in DB: {satellites}")

    print("\n--- SatNOGS live test complete ---")


if __name__ == "__main__":
    asyncio.run(main())
