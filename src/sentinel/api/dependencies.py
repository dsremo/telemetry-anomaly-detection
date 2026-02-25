"""FastAPI auth dependencies — JWT + API key, unified interface.

Every protected route declares one of these as a Depends parameter.
The dependency checks Bearer JWT first, then falls back to X-API-Key,
so both human users (JWT) and machine clients (API key) work transparently.

Demo mode: returns a synthetic admin user without any credential check,
so existing tests and the demo UI work with zero auth plumbing.
"""

from __future__ import annotations

import hashlib

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sentinel.core.security import decode_access_token
from sentinel.core.tenant import set_tenant

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Validate Bearer JWT or X-API-Key. Returns {user_id, tenant_id, role}.

    Priority:
      1. Bearer JWT — for human users / dashboard login
      2. X-API-Key  — for machine clients (CLI scripts, external integrations)

    In demo_mode, skips all credential checks and returns a synthetic admin
    so the demo UI and existing integration tests work without any auth setup.

    On success, sets the tenant ContextVar so RLS filters DB queries correctly.
    """
    if getattr(request.app.state, "demo_mode", False):
        return {"user_id": "demo", "tenant_id": "default", "role": "admin"}

    # 1. Try JWT Bearer token
    if credentials:
        secret = getattr(request.app.state, "jwt_secret", "")
        if not secret:
            raise HTTPException(
                status_code=503,
                detail="Auth not configured — set SENTINEL_JWT_SECRET",
            )
        try:
            payload = decode_access_token(credentials.credentials, secret)
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid or expired token: {exc}",
            ) from exc
        set_tenant(payload["tid"])
        return {
            "user_id": payload["sub"],
            "tenant_id": payload["tid"],
            "role": payload["role"],
        }

    # 2. Fall back to X-API-Key (machine clients / service accounts)
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        key_hash = hashlib.sha256(f"sentinel::{api_key}".encode()).hexdigest()
        key_tenant_map: dict[str, str] = getattr(
            request.app.state, "api_key_tenant_map", {}
        )
        tenant_id = key_tenant_map.get(key_hash)
        if tenant_id:
            set_tenant(tenant_id)
            # API key holders are treated as operator-level service accounts.
            return {"user_id": None, "tenant_id": tenant_id, "role": "operator"}

    raise HTTPException(status_code=401, detail="Authentication required")


def require_role(*allowed_roles: str):
    """RBAC Depends factory.

    Usage:
        @router.get("/protected")
        async def handler(_user: dict = Depends(require_role("admin", "operator"))):
            ...

    Returns a FastAPI dependency function that delegates to get_current_user and
    then checks the role. Re-uses the same user dict so there's no double auth.
    """
    async def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Role '{user['role']}' is not permitted. "
                    f"Required: {', '.join(allowed_roles)}"
                ),
            )
        return user
    return checker


# ---------------------------------------------------------------------------
# Convenience shortcuts — import directly in route files
# ---------------------------------------------------------------------------

require_admin    = require_role("admin")
require_operator = require_role("admin", "operator")
# viewer includes all roles that should be able to read data
require_viewer   = require_role("admin", "operator", "viewer", "report_only")
