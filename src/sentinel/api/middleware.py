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

logger = structlog.get_logger()


class PayloadLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding the size limit.

    First line of defense against memory exhaustion attacks.
    """

    def __init__(self, app, max_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
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

    Keys are hashed with SHA-256 and compared against stored hashes.
    The unhashed key never touches disk or logs.
    """

    EXEMPT_PATHS = frozenset({"/api/v1/health", "/docs", "/openapi.json", "/", "/dashboard"})

    def __init__(self, app, api_key_hashes: set[str] | None = None, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.key_hashes = api_key_hashes or set()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if path in self.EXEMPT_PATHS or path.startswith("/dashboard"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return JSONResponse(status_code=401, content={"detail": "Missing API key"})

        key_hash = hashlib.sha256(f"sentinel::{api_key}".encode()).hexdigest()
        if key_hash not in self.key_hashes:
            logger.warning("auth_failed", path=path, key_prefix=api_key[:8] if api_key else "")
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key sliding window rate limiting."""

    def __init__(self, app, max_requests: int = 300, window_seconds: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(max_requests, window_seconds)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
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
