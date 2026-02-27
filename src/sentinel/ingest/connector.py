"""DataConnector ABC + HTTPConnector base — common ingest interfaces.

All connectors (SatNOGSFetcher, ESADataLoader, CSVConnector, YAMCSConnector,
InfluxDBConnector, …) inherit from DataConnector.  HTTP-based connectors
additionally inherit from HTTPConnector which provides retry logic.

Connector-specific configuration (API tokens, file paths, satellite IDs)
belongs in __init__; bulk_load_to_db() can then be called without extra
arguments by generic pipeline code.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import httpx
import structlog

logger = structlog.get_logger()


class DataConnector(ABC):
    """Abstract base class for Sentinel telemetry ingest connectors."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source label, e.g. 'satnogs', 'esa-mission1', 'csv:file.csv'."""

    @abstractmethod
    async def bulk_load_to_db(
        self,
        *,
        resample_minutes: int = 1,
        skip_if_rows_gte: int = 50_000,
    ) -> dict[str, int]:
        """Load telemetry data, insert to DB, and run anomaly detection.

        Returns:
            Mapping of {parameter_or_satellite_id: rows_inserted}.
            Channels that were skipped (already loaded) are included with
            their existing row count.
        """


class HTTPConnector(DataConnector, ABC):
    """Base class for HTTP-based data sources.

    Provides _get() and _post() helpers with:
      - 429 retry using Retry-After header (or exponential fallback)
      - Exponential backoff on TransportError (3 attempts: 1s, 2s, 4s)
      - Shared httpx.AsyncClient with configurable headers/timeout

    Subclasses only need to implement source_name and bulk_load_to_db().
    """

    _MAX_ATTEMPTS: int = 3

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers: dict[str, str] = headers or {}

    async def _get(
        self,
        path: str,
        params: dict | None = None,
    ) -> httpx.Response:
        """GET with 429 retry (Retry-After) + exponential backoff on TransportError."""
        url = f"{self._base_url}{path}"
        last_resp: httpx.Response | None = None

        async with httpx.AsyncClient(
            timeout=self._timeout, headers=self._headers
        ) as client:
            for attempt in range(self._MAX_ATTEMPTS):
                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                        logger.warning(
                            "http_rate_limited",
                            url=url,
                            attempt=attempt,
                            retry_after=wait,
                        )
                        await asyncio.sleep(wait)
                        last_resp = resp
                        continue
                    resp.raise_for_status()
                    return resp
                except httpx.TransportError as exc:
                    if attempt == self._MAX_ATTEMPTS - 1:
                        raise
                    wait = 2.0 ** attempt
                    logger.warning(
                        "http_transport_error",
                        url=url,
                        attempt=attempt,
                        retry_in=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)

        raise httpx.HTTPStatusError(
            "Max retries exceeded",
            request=httpx.Request("GET", url),
            response=last_resp or httpx.Response(429),
        )

    async def _post(
        self,
        path: str,
        *,
        content: bytes,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST raw bytes with merged headers + TransportError backoff."""
        url = f"{self._base_url}{path}"
        merged = {**self._headers, **(extra_headers or {})}

        async with httpx.AsyncClient(timeout=self._timeout, headers=merged) as client:
            for attempt in range(self._MAX_ATTEMPTS):
                try:
                    resp = await client.post(url, content=content)
                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return resp
                except httpx.TransportError as exc:
                    if attempt == self._MAX_ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(2.0 ** attempt)

        raise httpx.HTTPStatusError(
            "Max retries exceeded",
            request=httpx.Request("POST", url),
            response=httpx.Response(429),
        )
