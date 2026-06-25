"""Single sign-on via the dsremo identity provider (auth.dsremo.com).

Flow:
  1. GET /auth/login          → redirect to the dsremo SSO authorize screen
  2. GET /auth/login/callback → exchange code → issue JWT → redirect to /dashboard/

Security:
  - State parameter: HMAC-SHA256(nonce, JWT_SECRET) prevents CSRF.
    The nonce is embedded in the state so the callback can verify it
    without server-side session storage.
  - The provider access token is exchanged for the user identity via the
    standard OIDC userinfo endpoint — no ID-token parsing needed.
  - New users get their own tenant (id = "t-{uuid4}") and the 'free' plan.
  - avatar_url and display_name are refreshed on every login when present.

Environment variables required:
  DSREMO_SSO_CLIENT_ID      — registered client id at auth.dsremo.com
  DSREMO_SSO_CLIENT_SECRET  — the client secret for that registration
  DSREMO_SSO_REDIRECT_URI   — must exactly match the registered redirect_uri,
                              e.g. https://yourhost/api/v1/auth/login/callback
  DSREMO_JWT_SECRET         — already required for existing auth

Optional overrides (default to auth.dsremo.com):
  DSREMO_SSO_AUTHORIZE_URL, DSREMO_SSO_TOKEN_URL, DSREMO_SSO_USERINFO_URL
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import urllib.parse
import uuid

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from dsremo.core.security import create_access_token
from dsremo.db import queries

logger = structlog.get_logger()
oauth_router = APIRouter(prefix="/auth", tags=["auth"])

_SSO_BASE = os.environ.get("DSREMO_SSO_BASE_URL", "https://auth.dsremo.com").rstrip("/")
_SSO_AUTHORIZE_URL = os.environ.get("DSREMO_SSO_AUTHORIZE_URL", f"{_SSO_BASE}/authorize")
_SSO_TOKEN_URL     = os.environ.get("DSREMO_SSO_TOKEN_URL", f"{_SSO_BASE}/token")
_SSO_USERINFO_URL  = os.environ.get("DSREMO_SSO_USERINFO_URL", f"{_SSO_BASE}/userinfo")

_DEFAULT_ACCESS_TTL  = 900       # 15 min
_DEFAULT_REFRESH_TTL = 604_800   # 7 days


def _sso_client_id() -> str:
    value = os.environ.get("DSREMO_SSO_CLIENT_ID", "")
    if not value:
        raise HTTPException(status_code=503, detail="SSO not configured")
    return value


def _sso_client_secret() -> str:
    value = os.environ.get("DSREMO_SSO_CLIENT_SECRET", "")
    if not value:
        raise HTTPException(status_code=503, detail="SSO not configured")
    return value


def _redirect_uri() -> str:
    return os.environ.get(
        "DSREMO_SSO_REDIRECT_URI",
        "http://localhost:8400/api/v1/auth/login/callback",
    )


def _make_state(nonce: str, secret: str) -> str:
    """HMAC-sign the nonce so the callback can verify it without server-side storage."""
    signature = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return f"{nonce}.{signature}"


def _verify_state(state: str, secret: str) -> bool:
    """Verify the HMAC-signed state parameter. Constant-time comparison."""
    try:
        nonce, signature = state.rsplit(".", 1)
    except ValueError:
        return False
    expected = hmac.new(secret.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# GET /auth/login  → redirect to the dsremo SSO authorize screen
# ---------------------------------------------------------------------------

@oauth_router.get("/login")
async def sso_login(request: Request) -> RedirectResponse:
    """Redirect the user's browser to the dsremo SSO sign-in screen."""
    secret = getattr(request.app.state, "jwt_secret", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Auth not configured — set DSREMO_JWT_SECRET")

    nonce = secrets.token_urlsafe(16)
    state = _make_state(nonce, secret)

    params = {
        "client_id":     _sso_client_id(),
        "redirect_uri":  _redirect_uri(),
        "response_type": "code",
        "scope":         "openid email",
        "state":         state,
    }
    url = f"{_SSO_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)


# ---------------------------------------------------------------------------
# GET /auth/login/callback  → exchange code → issue JWT → redirect to /dashboard/
# ---------------------------------------------------------------------------

@oauth_router.get("/login/callback")
async def sso_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Handle the SSO redirect back after the user signs in (or cancels)."""
    if error:
        logger.warning("sso_denied", error=error)
        return RedirectResponse(url="/?error=access_denied", status_code=302)

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")

    secret = getattr(request.app.state, "jwt_secret", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Auth not configured")

    if not _verify_state(state, secret):
        logger.warning("sso_state_mismatch", state=state[:20])
        raise HTTPException(status_code=400, detail="Invalid state parameter — possible CSRF")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                _SSO_TOKEN_URL,
                data={
                    "code":          code,
                    "client_id":     _sso_client_id(),
                    "client_secret": _sso_client_secret(),
                    "redirect_uri":  _redirect_uri(),
                    "grant_type":    "authorization_code",
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            user_resp = await client.get(
                _SSO_USERINFO_URL,
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            user_resp.raise_for_status()
            profile = user_resp.json()

    except httpx.HTTPError as exc:
        logger.error("sso_token_exchange_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Failed to communicate with SSO provider")

    subject      = profile.get("sub", "")
    email        = profile.get("email", "")
    display_name = profile.get("name", "") or (email.split("@")[0] if email else "")
    avatar_url   = profile.get("picture", "")

    if not subject or not email:
        raise HTTPException(status_code=502, detail="SSO provider did not return user identity")

    settings    = getattr(request.app.state, "settings", {})
    access_ttl  = int(settings.get("auth", {}).get("access_token_ttl_seconds", _DEFAULT_ACCESS_TTL))
    refresh_ttl = int(settings.get("auth", {}).get("refresh_token_ttl_seconds", _DEFAULT_REFRESH_TTL))

    existing    = await queries.get_user_by_google_id(subject)
    is_new_user = False

    if existing:
        user      = existing
        tenant_id = user["tenant_id"]
    else:
        tenant_id = f"t-{uuid.uuid4().hex[:12]}"
        await queries.create_tenant(tenant_id, name=email, plan="free")
        user = await queries.upsert_google_user(
            google_id=subject,
            email=email,
            display_name=display_name,
            avatar_url=avatar_url,
            tenant_id=tenant_id,
            role="admin",
            plan="free",
        )
        logger.info("sso_signup", email=email[:3] + "***", tenant_id=tenant_id)
        is_new_user = True

    if not user["active"]:
        return RedirectResponse(url="/?error=account_inactive", status_code=302)

    access_token = create_access_token(
        user_id=user["id"],
        tenant_id=tenant_id,
        role=user["role"],
        secret=secret,
        ttl_seconds=access_ttl,
        email=email,
    )

    import hashlib as _hl, secrets as _sec
    from datetime import datetime, timedelta, timezone
    from dsremo.core.tenant import set_tenant

    refresh_token = _sec.token_urlsafe(48)
    token_hash    = _hl.sha256(refresh_token.encode()).hexdigest()
    expires_at    = datetime.now(timezone.utc) + timedelta(seconds=refresh_ttl)

    set_tenant(tenant_id)
    await queries.store_refresh_token(user["id"], token_hash, expires_at)
    await queries.update_last_login(user["id"])

    logger.info("sso_login_success", user_id=user["id"], tenant=tenant_id)

    fragment = urllib.parse.urlencode({
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "plan":          user.get("plan") or "free",
        "new_user":      "1" if is_new_user else "0",
    })
    return RedirectResponse(url=f"/dashboard/#{fragment}", status_code=302)
