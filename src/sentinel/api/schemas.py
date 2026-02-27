"""Pydantic schemas — the API contract.

These define exactly what goes in and out of every endpoint.
Strict validation here is our first line of defense.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class TelemetryIn(BaseModel):
    """Single telemetry point from a customer."""

    satellite_id: str = Field(..., min_length=1, max_length=128, examples=["SAT-01"])
    timestamp: datetime | float | str
    subsystem: str = Field(..., min_length=1, max_length=32, examples=["eps"])
    parameter: str = Field(..., min_length=1, max_length=128, examples=["battery_voltage"])
    value: float
    unit: str = Field(default="", max_length=16, examples=["V"])
    quality: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("satellite_id", "subsystem", "parameter")
    @classmethod
    def no_control_chars(cls, v: str) -> str:
        if any(ord(c) < 32 for c in v):
            raise ValueError("control characters not allowed")
        return v


class TelemetryBatchIn(BaseModel):
    """Batch of telemetry points — up to 500 per request."""

    points: list[TelemetryIn] = Field(..., min_length=1, max_length=500)
    hmac_signature: str | None = Field(default=None, max_length=128)


class TelemetryOut(BaseModel):
    """Telemetry point as returned by the API."""

    satellite_id: str
    timestamp: datetime
    subsystem: str
    parameter: str
    value: float
    unit: str
    quality: float


class AnomalyOut(BaseModel):
    """Anomaly record as returned by the API."""

    id: str
    satellite_id: str
    timestamp: datetime
    subsystem: str
    parameter: str
    value: float
    severity: str
    confidence: float
    detectors_triggered: list[str]
    explanation: str
    root_cause_group: str | None
    contributing_params: dict[str, float]


class IngestResponse(BaseModel):
    """Response after telemetry ingestion."""

    accepted: int
    rejected: int
    errors: list[dict[str, Any]] = Field(default_factory=list)


class CsvUploadResult(BaseModel):
    """Result of a CSV telemetry file upload."""

    satellite_id: str
    channels_loaded: int
    channels_skipped: int        # channels that already had >= skip_if_rows_gte rows
    total_rows_inserted: int
    rows_per_channel: dict[str, int]
    source_name: str


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    version: str
    db_connected: bool
    uptime_seconds: float


class SimulateRequest(BaseModel):
    """Request to start the simulator."""

    satellite_id: str = Field(default="DEMO-SAT-01", max_length=128)
    duration_seconds: int = Field(default=300, ge=10, le=86400)
    rate_hz: float = Field(default=1.0, ge=0.1, le=10.0)


class InjectRequest(BaseModel):
    """Request to inject a fault into the simulator."""

    fault_type: str = Field(..., examples=["drift", "spike", "dropout", "degradation"])
    subsystem: str = Field(..., examples=["eps", "thermal"])
    parameter: str = Field(..., examples=["battery_voltage"])
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_seconds: int = Field(default=60, ge=1, le=3600)


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""

    data: list[Any]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Auth schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Email + password login. tenant_id scopes the lookup (B2B pattern)."""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=64)


class RefreshRequest(BaseModel):
    """Opaque refresh token sent by the client to obtain a new access token."""

    refresh_token: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Returned on successful login or token refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires


class UserOut(BaseModel):
    """Current user info (GET /auth/me)."""

    user_id: str
    email: str
    role: str
    tenant_id: str


# ---------------------------------------------------------------------------
# Tenant management schemas
# ---------------------------------------------------------------------------

class TenantIn(BaseModel):
    """Create a new tenant."""

    id: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9\-]+$")
    name: str = Field(..., min_length=1, max_length=128)
    plan: str = Field(default="free", max_length=32)


class TenantPatch(BaseModel):
    """Partial update for a tenant (name and/or active)."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    active: bool | None = None


class TenantOut(BaseModel):
    """Tenant record as returned by the API."""

    id: str
    name: str
    plan: str
    active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# User management schemas
# ---------------------------------------------------------------------------

_VALID_TENANT_ROLES = frozenset({"admin", "tenant_manager", "operator", "viewer", "report_only"})


class UserCreateRequest(BaseModel):
    """Create a new user within the current tenant."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="viewer")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in _VALID_TENANT_ROLES:
            raise ValueError(f"Invalid role. Choose from: {', '.join(sorted(_VALID_TENANT_ROLES))}")
        return v


class UpdateRoleRequest(BaseModel):
    """Change the role of a tenant user."""

    role: str = Field(..., min_length=1)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in _VALID_TENANT_ROLES:
            raise ValueError(f"Invalid role. Choose from: {', '.join(sorted(_VALID_TENANT_ROLES))}")
        return v


class UserDetailOut(BaseModel):
    """Full user record returned by user management endpoints."""

    id: str
    email: str
    role: str
    active: bool
    created_at: datetime
    last_login: datetime | None = None


# ---------------------------------------------------------------------------
# Password management schemas
# ---------------------------------------------------------------------------

class ChangePasswordRequest(BaseModel):
    """Authenticated user changes their own password."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


# ---------------------------------------------------------------------------
# API key management schemas
# ---------------------------------------------------------------------------

class ApiKeyCreateRequest(BaseModel):
    """Request to generate a new API key for the current tenant."""

    label: str = Field(..., min_length=1, max_length=64)


class ApiKeyCreateResponse(BaseModel):
    """Returned once on key creation — key plaintext shown ONCE."""

    key: str          # full plaintext key (never stored)
    label: str
    hash_prefix: str  # first 16 chars of hash for identification
    tenant_id: str


class ApiKeyOut(BaseModel):
    """API key record as returned by the list endpoint."""

    label: str
    hash_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    active: bool


# ---------------------------------------------------------------------------
# Channel registry + per-channel threshold config schemas
# ---------------------------------------------------------------------------

class ChannelOut(BaseModel):
    """Channel metadata, calibration state, and current effective thresholds."""

    satellite_id: str
    parameter: str
    subsystem: str
    unit: str
    total_points: int
    first_seen: datetime | None
    last_seen: datetime | None
    # Calibration
    calibration_state: str | None   # "warming_up" | "calibrated" | "recalibrating" | None
    has_overrides: bool             # True if a channel_config row exists for this channel
    # Effective thresholds (per-channel override merged with global defaults)
    effective_z_threshold: float
    effective_min_confidence: float
    effective_alert_cooldown_s: int


class ChannelConfigIn(BaseModel):
    """Per-channel threshold overrides — all fields optional (partial update via PUT).

    Omit any field to keep the existing DB value.  Set to null explicitly to
    clear an override and revert that field to the global default.
    """

    z_threshold: float | None = Field(
        default=None, gt=0, description="z-score alarm threshold (> 0)"
    )
    cusum_h: float | None = Field(
        default=None, gt=0, description="CUSUM alarm level H (> 0)"
    )
    cusum_k: float | None = Field(
        default=None, gt=0, description="CUSUM allowance k (> 0)"
    )
    ewma_lambda: float | None = Field(
        default=None, gt=0, le=1.0,
        description="EWMA smoothing factor λ (0 < λ ≤ 1)"
    )
    ewma_sigma_mult: float | None = Field(
        default=None, gt=0,
        description="EWMA UCL/LCL sigma multiplier (> 0)"
    )
    min_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum ensemble confidence to emit an anomaly (0.0–1.0)"
    )
    alert_cooldown_s: int | None = Field(
        default=None, ge=0,
        description="Per-channel alert cooldown in seconds (≥ 0)"
    )


class ChannelConfigOut(BaseModel):
    """Current per-channel config: raw DB overrides + effective merged thresholds."""

    satellite_id: str
    parameter: str
    overrides: dict[str, Any]   # non-None DB fields (empty dict if no row exists)
    effective: dict[str, Any]   # full merged thresholds as used by detection pipeline
    updated_at: datetime | None
