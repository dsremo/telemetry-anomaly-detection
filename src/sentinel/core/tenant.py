"""Tenant context — propagated via asyncio ContextVar.

Each HTTP request (asyncio task) carries its own copy of the ContextVar.
Middleware sets it after API key validation; connection.acquire() reads it
to enforce Row Level Security on every query.

Default: 'default' — existing CLI scripts and tests work with zero changes.
"""
from __future__ import annotations

from contextvars import ContextVar

_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="default")


def set_tenant(tenant_id: str) -> None:
    """Set the active tenant for the current asyncio task."""
    _tenant_id.set(tenant_id)


def get_tenant() -> str:
    """Return the active tenant ID (defaults to 'default')."""
    return _tenant_id.get()
