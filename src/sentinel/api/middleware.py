"""Security middleware — auth, rate limiting, audit logging, payload limits.

Every request passes through these layers before reaching route handlers.
Defense in depth: even if one layer fails, others catch threats.
"""

from __future__ import annotations

import hashlib
import time

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from sentinel.core.security import RateLimiter
from sentinel.core.tenant import set_tenant

logger = structlog.get_logger()


def _is_websocket(scope: dict) -> bool:
    """Check if the current request is a WebSocket upgrade."""
    return scope.get("type") == "websocket"


class PayloadLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding the size limit.

    First line of defense against memory exhaustion attacks.
    """

    def __init__(self, app, max_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_websocket(request.scope):
            return await call_next(request)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            logger.warning("payload_too_large", size=content_length)
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload exceeds {self.max_bytes} bytes"},
            )
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key on every request (except health check and docs).

    The key→tenant map is read from request.app.state.api_key_tenant_map on
    every request — populated by lifespan after DB connects, refreshable at
    any time (e.g., when new keys are created via CLI).

    On success, the request's tenant is set via ContextVar so all downstream
    DB calls are filtered to that tenant by PostgreSQL RLS.

    The unhashed key never touches disk or logs.
    """

    EXEMPT_PATHS = frozenset({"/api/v1/health", "/docs", "/openapi.json", "/", "/dashboard"})

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_websocket(request.scope):
            return await call_next(request)
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if path in self.EXEMPT_PATHS or path.startswith("/dashboard"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return JSONResponse(status_code=401, content={"detail": "Missing API key"})

        key_hash = hashlib.sha256(f"sentinel::{api_key}".encode()).hexdigest()
        # Read the live map from app.state — safe to refresh without restart.
        key_tenant_map: dict[str, str] = getattr(
            request.app.state, "api_key_tenant_map", {}
        )
        tenant_id = key_tenant_map.get(key_hash)
        if tenant_id is None:
            logger.warning("auth_failed", path=path, key_prefix=api_key[:8] if api_key else "")
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        # Propagate tenant to all DB calls made during this request.
        set_tenant(tenant_id)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key sliding window rate limiting."""

    def __init__(self, app, max_requests: int = 300, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, window_seconds)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_websocket(request.scope):
            return await call_next(request)
        # Rate limit by API key or by IP for unauthenticated endpoints
        key = request.headers.get("X-API-Key", "")
        if not key:
            key = request.client.host if request.client else "unknown"

        if not self.limiter.is_allowed(key):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log every API request for security auditing.

    Captures: timestamp, method, path, status code, latency, API key prefix.
    Never logs request bodies or full API keys.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _is_websocket(request.scope):
            return await call_next(request)
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        api_key = request.headers.get("X-API-Key", "")
        logger.info(
            "api_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 1),
            key_prefix=api_key[:8] if api_key else "none",
            client=request.client.host if request.client else "unknown",
        )
        return response
