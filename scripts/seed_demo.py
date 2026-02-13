"""Seed demo data — generates telemetry with injected faults for demos.

Run: python scripts/seed_demo.py
Requires: Sentinel API running on localhost:8400

Generates 10 minutes of normal telemetry, then injects battery degradation.
The dashboard should show the anomaly detection in real-time.
"""

import asyncio
import sys
import time

import httpx

API_URL = "http://localhost:8400"


async def main():
    print("Sentinel Demo Seeder")
    print("=" * 50)
    print(f"API: {API_URL}")
    print()

    # Check health
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{API_URL}/api/v1/health")
            health = resp.json()
            print(f"Status: {health.get('status')}")
            print(f"DB: {'connected' if health.get('db_connected') else 'NOT connected'}")
        except httpx.HTTPError as e:
            print(f"Cannot reach API: {e}")
            print("Start the server first: sentinel serve")
            sys.exit(1)

    print()
    print("Step 1: Starting simulation (5 min normal)...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{API_URL}/api/v1/simulate/start", json={
            "satellite_id": "DEMO-SAT-01",
            "duration_seconds": 300,
            "rate_hz": 1.0,
        })
        print(f"  Response: {resp.json()}")

    print()
    print("Waiting 60 seconds for baseline to build...")
    await asyncio.sleep(60)

    print()
    print("Step 2: Injecting battery degradation...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{API_URL}/api/v1/simulate/inject", json={
            "fault_type": "degradation",
            "subsystem": "eps",
            "parameter": "battery_voltage",
            "intensity": 0.6,
            "duration_seconds": 120,
        })
        print(f"  Response: {resp.json()}")

    print()
    print("Monitor the dashboard at: http://localhost:8400/dashboard")
    print("Check anomalies at: http://localhost:8400/api/v1/anomalies")
    print()
    print("Waiting for simulation to complete...")
    await asyncio.sleep(240)

    print()
    print("Step 3: Checking results...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{API_URL}/api/v1/anomalies?limit=10")
        anomalies = resp.json()
        print(f"  Anomalies detected: {len(anomalies)}")
        for a in anomalies[:5]:
            print(f"    [{a.get('severity', '?').upper():>8}] {a.get('parameter', '?')}: {a.get('explanation', '')[:80]}")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
