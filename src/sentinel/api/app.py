"""FastAPI application factory — single entrypoint for the HTTP server.

Wires up middleware, routes, lifespan events, and static file serving.
No global state — everything flows through the app's dependency injection.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from sentinel import __version__
from sentinel.api.middleware import (
    ApiKeyMiddleware,
    AuditLogMiddleware,
    PayloadLimitMiddleware,
    RateLimitMiddleware,
)
from sentinel.api.routes import router
from sentinel.api.websocket import ws_router
from sentinel.core.config import load_config

logger = structlog.get_logger()

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dashboard"

# Global flag checked by routes and detection pipeline
demo_mode: bool = False


def _activate_demo_mode() -> None:
    """Swap DB queries module with in-memory store throughout the app."""
    global demo_mode
    demo_mode = True

    from sentinel.db import memory_store

    # Patch routes and detection pipeline to use memory_store instead of DB queries
    import sentinel.api.routes as routes_mod
    import sentinel.detection.detector as detector_mod

    routes_mod.queries = memory_store  # type: ignore[attr-defined]
    detector_mod.queries = memory_store  # type: ignore[attr-defined]

    logger.info("demo_mode_activated", storage="in-memory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect DB (or use memory store in demo mode). Shutdown: cleanup."""
    settings = app.state.settings

    if app.state.demo_mode:
        _activate_demo_mode()
    else:
        from sentinel.db.connection import close_pool, init_pool
        from sentinel.db.migrations import run_migrations

        db = settings.get("database", {})
        await init_pool(
            host=db.get("host", "localhost"),
            port=db.get("port", 5432),
            database=db.get("name", "sentinel"),
            user=db.get("user", "sentinel"),
            password=db.get("password", ""),
            min_size=db.get("pool_min", 2),
            max_size=db.get("pool_max", 10),
        )
        await run_migrations()

    # Wire config thresholds into detector singletons
    from sentinel.detection.detector import init_detectors
    init_detectors(settings)

    # Wire alert service — dispatches webhooks/email on WARNING+ anomalies.
    from sentinel.alerts.service import init_alert_service
    al = settings.get("alerts", {})
    init_alert_service(
        webhook_url=al.get("webhook_url", ""),
        dedup_window_sec=float(al.get("dedup_window_seconds", 300)),
        escalation_delay_sec=float(al.get("escalation_delay_seconds", 600)),
    )

    app.state.start_time = time.monotonic()
    logger.info("sentinel_started", version=__version__, demo=app.state.demo_mode)
    yield
    if not app.state.demo_mode:
        from sentinel.db.connection import close_pool
        await close_pool()
    logger.info("sentinel_stopped")


def create_app(config_path: Path | None = None, demo: bool = False) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        config_path: Path to sentinel.yaml. Uses default discovery if None.
        demo: If True, runs entirely in-memory with no PostgreSQL needed.
    """
    settings = load_config(config_path)

    app = FastAPI(
        title="Sentinel",
        description="AI Telemetry Anomaly Detection Engine",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.demo_mode = demo

    # --- Middleware stack (order matters: outermost runs first) ---
    sec = settings.get("security", {})

    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=sec.get("rate_limit_per_minute", 300),
    )
    app.add_middleware(
        ApiKeyMiddleware,
        enabled=False,  # disabled until first key is generated via CLI
    )
    app.add_middleware(
        PayloadLimitMiddleware,
        max_bytes=sec.get("max_payload_bytes", 1_048_576),
    )

    cors = settings.get("server", {}).get("cors_origins", ["http://localhost:8400"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")

    # --- Root redirect → dashboard ---
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/dashboard/", status_code=301)

    # --- Dashboard static files ---
    if _DASHBOARD_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")

    return app
