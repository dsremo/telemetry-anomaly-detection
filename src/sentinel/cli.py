"""Sentinel CLI — the operational interface for the engine.

Commands:
  sentinel serve          — Start the API server
  sentinel simulate       — Run spacecraft simulator
  sentinel key generate   — Generate a new API key (plaintext shown ONCE)
  sentinel key list       — List active API keys (hashed, never plaintext)
  sentinel key revoke     — Revoke an API key
  sentinel health         — Check system health
  sentinel scenarios      — List available fault injection scenarios
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="sentinel",
    help="Sentinel — AI Telemetry Anomaly Detection Engine",
    no_args_is_help=True,
)
key_app = typer.Typer(help="API key management")
app.add_typer(key_app, name="key")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8400, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
    config: Path | None = typer.Option(None, help="Path to sentinel.yaml"),
    demo: bool = typer.Option(False, help="Demo mode — runs in-memory, no PostgreSQL needed"),
) -> None:
    """Start the Sentinel API server."""
    import uvicorn

    if demo:
        typer.echo(f"Starting Sentinel DEMO on {host}:{port} (in-memory, no DB)")
        typer.echo("  Dashboard: http://localhost:8400/dashboard")
        typer.echo("  API docs:  http://localhost:8400/docs")
        typer.echo()

        # Pass demo=True to the factory
        from functools import partial
        from sentinel.api.app import create_app

        app_instance = create_app(config_path=config, demo=True)
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="info",
        )
    else:
        typer.echo(f"Starting Sentinel on {host}:{port}")
        uvicorn.run(
            "sentinel.api.app:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
            log_level="info",
        )


@app.command()
def simulate(
    satellite: str = typer.Option("DEMO-SAT-01", help="Satellite ID"),
    duration: int = typer.Option(300, help="Duration in seconds"),
    rate: float = typer.Option(1.0, help="Telemetry rate in Hz"),
    inject: str = typer.Option("", help="Fault scenario to inject (e.g., battery_degradation)"),
    inject_after: int = typer.Option(60, help="Seconds before fault injection"),
    api_url: str = typer.Option("http://localhost:8400", help="Sentinel API URL"),
) -> None:
    """Run the spacecraft simulator and push telemetry to Sentinel API."""
    asyncio.run(_run_simulator(satellite, duration, rate, inject, inject_after, api_url))


async def _run_simulator(
    satellite: str, duration: int, rate: float, inject: str, inject_after: int, api_url: str,
) -> None:
    import time

    import httpx

    from sentinel.simulate.injector import SCENARIOS, apply_scenario
    from sentinel.simulate.spacecraft import SpacecraftSimulator

    sim = SpacecraftSimulator(satellite_id=satellite, rate_hz=rate)
    typer.echo(f"Simulator: {satellite} | Duration: {duration}s | Rate: {rate}Hz")

    if inject:
        if inject not in SCENARIOS:
            typer.echo(f"Unknown scenario: {inject}. Available: {list(SCENARIOS.keys())}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Will inject '{inject}' after {inject_after}s")

    injected = False
    start = time.monotonic()
    tick_count = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.monotonic() - start < duration:
            elapsed = time.monotonic() - start

            # Inject fault at scheduled time
            if inject and not injected and elapsed >= inject_after:
                scenario = apply_scenario(sim, inject)
                typer.echo(f"\n[{int(elapsed)}s] INJECTING: {scenario.name} — {scenario.description}")
                injected = True

            points = sim.generate_tick()
            payload = {
                "points": [
                    {
                        "satellite_id": p.satellite_id,
                        "timestamp": p.timestamp.isoformat(),
                        "subsystem": p.subsystem,
                        "parameter": p.parameter,
                        "value": p.value,
                        "unit": p.unit,
                        "quality": p.quality,
                    }
                    for p in points
                ]
            }

            try:
                resp = await client.post(f"{api_url}/api/v1/telemetry", json=payload)
                tick_count += 1
                if tick_count % 30 == 0:
                    body = resp.json()
                    typer.echo(
                        f"  [{int(elapsed):>4}s] sent {len(points)} pts | "
                        f"accepted: {body.get('accepted', '?')} | "
                        f"phase: {sim.orbital_phase:.2f} | "
                        f"sun: {'☀' if sim.in_sunlight else '🌑'}"
                    )
            except httpx.HTTPError as e:
                typer.echo(f"  [!] API error: {e}", err=True)

            await asyncio.sleep(1.0 / rate)

    typer.echo(f"\nSimulation complete. {tick_count} ticks sent to {api_url}")


@key_app.command("generate")
def key_generate(
    label: str = typer.Option(..., prompt="Label for this API key (e.g., 'dev-testing')"),
    db_url: str = typer.Option("", help="PostgreSQL connection string"),
) -> None:
    """Generate a new API key. The plaintext is shown ONCE — save it securely."""
    from sentinel.core.security import generate_api_key

    plaintext, hashed = generate_api_key()

    typer.echo("\n" + "=" * 60)
    typer.echo("  NEW API KEY GENERATED")
    typer.echo("=" * 60)
    typer.echo(f"\n  Label:     {label}")
    typer.echo(f"  🔴 Key:     {plaintext}")
    typer.echo(f"  Hash:      {hashed[:16]}...")
    typer.echo("\n  ⚠  SAVE THIS KEY NOW. It will NOT be shown again.")
    typer.echo("  ⚠  Store it in your .env file or secrets manager.")
    typer.echo("=" * 60)

    # If DB is available, store the hash
    if db_url:
        typer.echo(f"\n  Storing hash in database...")
        asyncio.run(_store_key(hashed, label))
        typer.echo("  Done.")
    else:
        typer.echo(f"\n  To register in DB, run:")
        typer.echo(f"    sentinel key register --hash {hashed} --label '{label}'")


@key_app.command("register")
def key_register(
    hash: str = typer.Option(..., help="The SHA-256 hash of the API key"),
    label: str = typer.Option(..., help="Label for this key"),
) -> None:
    """Register a pre-hashed API key in the database."""
    asyncio.run(_store_key(hash, label))
    typer.echo(f"API key registered: {label} ({hash[:16]}...)")


@key_app.command("revoke")
def key_revoke(
    hash_prefix: str = typer.Option(..., help="First 16 chars of the key hash"),
) -> None:
    """Revoke an API key by its hash prefix."""
    asyncio.run(_revoke_key(hash_prefix))
    typer.echo(f"API key revoked: {hash_prefix}...")


@app.command()
def scenarios() -> None:
    """List available fault injection scenarios."""
    from sentinel.simulate.injector import list_scenarios

    typer.echo("\nAvailable Fault Scenarios:\n")
    for s in list_scenarios():
        typer.echo(f"  {s['name']}")
        typer.echo(f"    {s['description']}\n")


@app.command()
def health(
    api_url: str = typer.Option("http://localhost:8400", help="Sentinel API URL"),
) -> None:
    """Check system health."""
    import httpx

    try:
        resp = httpx.get(f"{api_url}/api/v1/health", timeout=5.0)
        data = resp.json()
        typer.echo(f"Status:       {data.get('status', 'unknown')}")
        typer.echo(f"Version:      {data.get('version', '?')}")
        typer.echo(f"DB Connected: {data.get('db_connected', False)}")
        typer.echo(f"Uptime:       {data.get('uptime_seconds', 0):.0f}s")
    except httpx.HTTPError as e:
        typer.echo(f"Health check failed: {e}", err=True)
        raise typer.Exit(1)


# --- DB helpers for key management ---

async def _store_key(key_hash: str, label: str) -> None:
    from sentinel.core.config import load_config
    from sentinel.db.connection import close_pool, init_pool
    from sentinel.db.queries import store_api_key

    settings = load_config()
    db = settings.get("database", {})
    await init_pool(
        host=db.get("host", "localhost"),
        database=db.get("name", "sentinel"),
        user=db.get("user", "sentinel"),
        password=db.get("password", ""),
    )
    await store_api_key(key_hash, label)
    await close_pool()


async def _revoke_key(hash_prefix: str) -> None:
    from sentinel.db.connection import acquire, close_pool, init_pool
    from sentinel.core.config import load_config

    settings = load_config()
    db = settings.get("database", {})
    await init_pool(
        host=db.get("host", "localhost"),
        database=db.get("name", "sentinel"),
        user=db.get("user", "sentinel"),
        password=db.get("password", ""),
    )
    async with acquire() as conn:
        await conn.execute(
            "UPDATE api_keys SET active = FALSE WHERE key_hash LIKE $1 || '%'",
            hash_prefix,
        )
    await close_pool()
