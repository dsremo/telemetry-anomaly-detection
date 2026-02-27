"""Shared ingest utilities — timezone handling, series preparation, validation, retry.

All connectors (CSVConnector, SatNOGSFetcher, ESADataLoader, …) use these
helpers to ensure consistent behaviour without duplicating code.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Coroutine, TypeVar

import httpx
import pandas as pd
import structlog

logger = structlog.get_logger()

_T = TypeVar("_T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (httpx.TransportError,),
) -> Callable[[Callable[..., Coroutine[Any, Any, _T]]], Callable[..., Coroutine[Any, Any, _T]]]:
    """Decorator: retry an async function on specified exceptions with exponential backoff.

    Args:
        max_attempts: Total attempts before re-raising (default 3).
        base_delay:   Initial delay in seconds; doubles each attempt (1s, 2s, 4s, …).
        exceptions:   Exception types to catch and retry (default: httpx.TransportError).

    Usage:
        @retry_with_backoff(max_attempts=3)
        async def fetch_page(self, url: str) -> bytes:
            ...
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, _T]]
    ) -> Callable[..., Coroutine[Any, Any, _T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        raise
                    wait = base_delay * (2 ** attempt)
                    logger.warning(
                        "retry_backoff",
                        func=func.__qualname__,
                        attempt=attempt + 1,
                        retry_in=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)
            raise RuntimeError("retry_with_backoff: unreachable")  # pragma: no cover
        return wrapper
    return decorator


def ensure_utc_series(series: pd.Series) -> pd.Series:
    """Return series with a UTC-aware DatetimeIndex.

    If the index is already tz-aware this is a no-op (no copy).
    If the index is naive (no tz), it is localized to UTC.
    """
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")
    return series


def prepare_series(series: pd.Series, resample_minutes: int = 1) -> pd.Series:
    """Normalize timezone, optionally resample to coarser resolution, drop NaN.

    Pipeline:
        1. Localize naive index → UTC  (ensure_utc_series)
        2. Resample via median if resample_minutes > 1
        3. Drop NaN rows

    Always returns a new Series; never mutates the input.
    """
    series = ensure_utc_series(series)
    if resample_minutes > 1:
        series = series.resample(f"{resample_minutes}min").median()
    return series.dropna()


def validated_resample(minutes: int) -> int:
    """Return `minutes` unchanged, or raise ValueError if < 1."""
    if minutes < 1:
        raise ValueError(f"resample_minutes must be >= 1, got {minutes!r}")
    return minutes


def validated_satellite_id(sid: str) -> str:
    """Strip whitespace and return the satellite ID, or raise ValueError if empty."""
    sid = sid.strip() if isinstance(sid, str) else ""
    if not sid:
        raise ValueError("satellite_id must not be empty")
    return sid
