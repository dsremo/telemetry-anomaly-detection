"""Shared ingest utilities — timezone handling, series preparation, validation, retry.

All connectors (CSVConnector, SatNOGSFetcher, ESADataLoader, …) use these
helpers to ensure consistent behaviour without duplicating code.

Data-frequency utilities (usable by all analyze_* scripts):
    detect_data_frequency()   — peek at a CSV file and return the median
                                inter-sample interval in seconds.
    adaptive_cooldown_hours() — convert a sampling interval to a proportional
                                alert cooldown.  Formula: max(5 min, 500×interval),
                                capped at 72 h.  Scales automatically from
                                1-second SCADA data (≈8 min) to 1-hour satellite
                                telemetry (72 h cap).
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
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


# ── Data-frequency helpers (shared by all analyze_* scripts) ─────────────────

def detect_data_frequency(
    file_path: Path,
    timestamp_col: str = "timestamp",
) -> float:
    """Return median inter-sample interval in seconds by peeking at the first 200 rows.

    Used to scale the alert cooldown to the data frequency before running
    anomaly detection — avoids the foot-gun of applying a 72-hour cooldown
    to 1-second SCADA data (which would suppress all multi-window events).

    Handles both comma and semicolon CSV delimiters (auto-detected from the
    first 500 bytes of the file).

    Args:
        file_path:     Path to a wide-format CSV (timestamp + parameter columns).
        timestamp_col: Name of the timestamp column (default "timestamp").

    Returns:
        Median inter-sample interval in seconds (float).
        Falls back to 3600.0 (1-hour assumption) on any error or if fewer
        than 2 rows are present.
    """
    try:
        text_peek = file_path.read_text(errors="replace")[:500]
        sep = ";" if ";" in text_peek else ","
        df = pd.read_csv(file_path, sep=sep, nrows=200)
        if timestamp_col not in df.columns:
            return 3600.0
        ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            return 3600.0
        diffs = ts.diff().dropna().dt.total_seconds()
        return float(diffs.median())
    except Exception:
        return 3600.0


def adaptive_cooldown_hours(median_interval_s: float) -> float:
    """Compute alert cooldown (hours) proportional to data frequency.

    Formula: cooldown = max(5 minutes, 500 × median_interval_s), capped at 72 h.

    Scaling examples:
        1-second SCADA/industrial data  → 500 s  ≈  8.3 min
        5-minute monitoring data        → 2500 s ≈ 41.7 min
        1-hour satellite telemetry      → 500 h  → capped at 72 h

    The 500× multiplier ensures the cooldown spans many data-frequency
    "periods", so a brief anomaly is not immediately followed by another
    false positive on the receding edge.

    Args:
        median_interval_s: Median inter-sample interval in seconds (from
                           detect_data_frequency()).  Must be ≥ 0.

    Returns:
        Cooldown in hours (float), in the range [0.0833, 72.0].
    """
    cooldown_s = max(300.0, 500.0 * max(0.0, median_interval_s))
    cooldown_s = min(cooldown_s, 72.0 * 3600.0)   # cap at 72 h
    return cooldown_s / 3600.0
