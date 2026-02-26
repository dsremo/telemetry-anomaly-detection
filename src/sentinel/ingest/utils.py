"""Shared ingest utilities — timezone handling, series preparation, validation.

All connectors (CSVConnector, SatNOGSFetcher, ESADataLoader, …) use these
helpers to ensure consistent behaviour without duplicating code.
"""

from __future__ import annotations

import pandas as pd


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
