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
from fastapi.staticfiles import StaticFiles

from sentinel import __version__
from sentinel.api.middleware import (
    ApiKeyMiddleware,
    AuditLogMiddleware,
    PayloadLimitMiddleware,
    RateLimitMiddleware,
)
from sentinel.api.routes import router
from sentinel.core.config import load_config
from sentinel.db.connection import close_pool, init_pool
from sentinel.db.migrations import run_migrations

logger = structlog.get_logger()

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect DB, run migrations. Shutdown: close pool."""
    settings = app.state.settings

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

    app.state.start_time = time.monotonic()
    logger.info("sentinel_started", version=__version__)
    yield
    await close_pool()
    logger.info("sentinel_stopped")


def create_app(config_path: Path | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
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

    # --- Dashboard static files ---
    if _DASHBOARD_DIR.exists():
        app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")

    return app
