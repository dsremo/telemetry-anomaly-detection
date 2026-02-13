"""Async PostgreSQL connection pool — single pool per process.

Uses asyncpg for raw performance. No ORM overhead.
Connection parameters come from config, secrets from env vars.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
import structlog

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None


async def init_pool(
    host: str = "localhost",
    port: int = 5432,
    database: str = "sentinel",
    user: str = "sentinel",
    password: str = "",
    min_size: int = 2,
    max_size: int = 10,
) -> asyncpg.Pool:
    """Create the connection pool. Call once at startup."""
    global _pool
    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        min_size=min_size,
        max_size=max_size,
        command_timeout=30,
    )
    logger.info("db_pool_created", host=host, database=database, pool_size=f"{min_size}-{max_size}")
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool. Call at shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db_pool_closed")


def get_pool() -> asyncpg.Pool:
    """Get the active connection pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized — call init_pool() first")
    return _pool


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection from the pool with automatic release."""
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn
