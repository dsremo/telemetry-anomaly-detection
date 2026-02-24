"""SatNOGS API fetcher — pulls live telemetry from the SatNOGS network.

SatNOGS is the world's largest open-source satellite ground station network.
Their DB has telemetry from hundreds of real satellites collected by
community ground stations worldwide.

API: https://db.satnogs.org/api/
Auth: Token-based (stored in .env as SATNOGS_API_TOKEN)

NOTE: The SatNOGS public REST API returns raw hex frames — decoded telemetry
is stored in their internal InfluxDB and is NOT exposed via REST. We extract
signal-level metrics (frame size, byte statistics, reception patterns) from
the raw frames, which are genuine satellite data useful for anomaly detection.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from sentinel.core.models import TelemetryPoint

logger = structlog.get_logger()

SATNOGS_API_BASE = "https://db.satnogs.org/api"


class SatNOGSFetcher:
    """Fetches satellite telemetry from the SatNOGS DB API."""

    def __init__(self, api_token: str | None = None):
        self.api_token = api_token or os.environ.get("SATNOGS_API_TOKEN", "")
        if not self.api_token:
            # .env might not be loaded yet (standalone usage outside API server)
            from sentinel.core.config import _load_dotenv
            _load_dotenv()
            self.api_token = os.environ.get("SATNOGS_API_TOKEN", "")
        if not self.api_token:
            logger.warning("satnogs_no_token", hint="Set SATNOGS_API_TOKEN in .env")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_token}"}

    async def fetch_telemetry(
        self,
        satellite_norad_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch raw telemetry frames for a satellite by its NORAD catalog ID.

        SatNOGS paginates at 25 frames per page. This method follows `next`
        links until the requested limit is reached.

        Returns list of frame dicts from SatNOGS. Use convert_to_points()
        to extract signal-level metrics as TelemetryPoints.
        """
        if not self.api_token:
            raise ValueError("SATNOGS_API_TOKEN not set — check .env file")

        frames: list[dict] = []
        # SatNOGS paginates — request up to `limit` frames in one page where possible.
        # We single-page to stay within API rate limits; the page cap is ~500.
        url = f"{SATNOGS_API_BASE}/telemetry/"
        params: dict[str, Any] = {
            "satellite": satellite_norad_id,
            "limit": min(limit, 500),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(3):  # up to 3 attempts on rate-limit
                resp = await client.get(url, headers=self._headers, params=params)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning(
                        "satnogs_rate_limited",
                        satellite=satellite_norad_id,
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status_code == 404:
                    logger.info("satnogs_no_data", satellite=satellite_norad_id)
                    return []

                resp.raise_for_status()
                data = resp.json()
                break
            else:
                logger.error("satnogs_rate_limit_exceeded", satellite=satellite_norad_id)
                return []

        # SatNOGS returns paginated dict {"next":, "previous":, "results": [...]}
        if isinstance(data, dict):
            frames = data.get("results", [])
        else:
            frames = data

        frames = frames[:limit]
        logger.info(
            "satnogs_fetched",
            satellite=satellite_norad_id,
            frames=len(frames),
        )
        return frames

    async def fetch_all_telemetry(
        self,
        satellite_norad_id: str,
        max_frames: int = 500,
        inter_page_delay: float = 2.0,
    ) -> list[dict]:
        """Fetch all available frames for a satellite, following pagination.

        SatNOGS returns 25 frames/page for high-traffic satellites (e.g. ISS).
        This method follows ``next`` cursor links until ``max_frames`` frames
        have been collected or the server has no more pages.

        Args:
            satellite_norad_id: NORAD catalog ID (e.g. "25544").
            max_frames:         Hard cap on total frames.  SatNOGS enforces
                                aggressive rate limits; 500 (20 pages) is a
                                safe default.  Increase with care.
            inter_page_delay:   Seconds to sleep between pages.  2 s keeps
                                request rate at ~0.5 req/s, well under the
                                observed ~150-frame / 37 s throttle window.

        Returns:
            List of raw frame dicts in server-returned order.
        """
        if not self.api_token:
            raise ValueError("SATNOGS_API_TOKEN not set — check .env file")

        frames: list[dict] = []
        url: str | None = f"{SATNOGS_API_BASE}/telemetry/"
        params: dict = {
            "satellite": satellite_norad_id,
            "limit": min(100, max_frames),  # SatNOGS silently caps above ~100
        }

        # 60 s read timeout — slow responses occur after rate-limit back-offs.
        async with httpx.AsyncClient(timeout=60.0) as client:
            page = 0
            while url and len(frames) < max_frames:
                page += 1
                data: dict | list | None = None

                for attempt in range(3):
                    try:
                        resp = await client.get(
                            url,
                            headers=self._headers,
                            params=params if page == 1 else None,
                        )
                    except httpx.TimeoutException:
                        wait = 10 * (attempt + 1)  # 10 s, 20 s, 30 s
                        logger.warning(
                            "satnogs_read_timeout",
                            satellite=satellite_norad_id,
                            attempt=attempt + 1,
                            retry_in=wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(
                            "satnogs_rate_limited",
                            satellite=satellite_norad_id,
                            retry_after=retry_after,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status_code == 404:
                        logger.info("satnogs_no_data", satellite=satellite_norad_id)
                        return frames  # satellite has no telemetry in SatNOGS — not an error
                    resp.raise_for_status()
                    data = resp.json()
                    break
                else:
                    logger.error("satnogs_page_failed", satellite=satellite_norad_id, page=page)
                    break  # give up on this satellite

                if data is None:
                    break  # all retries exhausted (timeout path)

                if isinstance(data, dict):
                    page_frames = data.get("results", [])
                    url = data.get("next")   # None → no more pages
                else:
                    page_frames = data
                    url = None

                frames.extend(page_frames)
                logger.info(
                    "satnogs_page_fetched",
                    satellite=satellite_norad_id,
                    page=page,
                    page_frames=len(page_frames),
                    total_so_far=len(frames),
                )

                # Respect max_frames ceiling.
                if len(frames) >= max_frames:
                    frames = frames[:max_frames]
                    break

                # Polite inter-page delay — SatNOGS is a community resource.
                if url and inter_page_delay > 0:
                    await asyncio.sleep(inter_page_delay)

        logger.info(
            "satnogs_fetched_all",
            satellite=satellite_norad_id,
            total_frames=len(frames),
        )
        return frames

    async def fetch_satellite_info(self, norad_id: str) -> dict:
        """Get satellite metadata from SatNOGS."""
        url = f"{SATNOGS_API_BASE}/satellites/{norad_id}/"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            return resp.json()

    async def list_satellites(self, status: str = "alive") -> list[dict]:
        """List satellites in the SatNOGS database.

        status: "alive", "dead", "re-entered", or "future"
        """
        url = f"{SATNOGS_API_BASE}/satellites/"
        params = {"status": status, "limit": 50}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            return resp.json()

    def convert_to_points(
        self,
        raw_frames: list[dict],
        satellite_id: str | None = None,
    ) -> list[TelemetryPoint]:
        """Convert SatNOGS raw frames into Sentinel TelemetryPoints.

        The SatNOGS public API does NOT include decoded telemetry —
        the 'decoded' field is just a flag string, not a data dict.

        Instead, we extract signal-level metrics from the raw hex frames:
          - frame_length: size of the received frame in bytes
          - byte_mean: mean byte value (0-255) — changes indicate payload shifts
          - byte_entropy: Shannon entropy of bytes — detects encoding changes
          - frame_rate: inter-frame timing (computed from timestamps)

        These are genuine satellite signal measurements that our anomaly
        detection pipeline can analyze for reception quality changes.
        """
        points: list[TelemetryPoint] = []
        prev_ts: datetime | None = None

        for frame in raw_frames:
            if not isinstance(frame, dict):
                continue

            sat_id = satellite_id or str(frame.get("norad_cat_id", "UNKNOWN"))
            timestamp_str = frame.get("timestamp", "")

            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            frame_hex = frame.get("frame", "")
            if not frame_hex:
                continue

            try:
                frame_bytes = bytes.fromhex(frame_hex)
            except ValueError:
                continue

            if len(frame_bytes) < 2:
                continue

            # --- Extract signal-level metrics ---

            # Frame length — anomalous if suddenly changes
            points.append(TelemetryPoint(
                satellite_id=sat_id,
                timestamp=ts,
                subsystem="comms",
                parameter="frame_length",
                value=float(len(frame_bytes)),
                unit="bytes",
                quality=0.9,
            ))

            # Mean byte value — shifts indicate payload/encoding changes
            byte_values = list(frame_bytes)
            mean_val = sum(byte_values) / len(byte_values)
            points.append(TelemetryPoint(
                satellite_id=sat_id,
                timestamp=ts,
                subsystem="comms",
                parameter="byte_mean",
                value=round(mean_val, 2),
                unit="",
                quality=0.9,
            ))

            # Byte entropy — low=structured data, high=encrypted/noise
            entropy = _byte_entropy(frame_bytes)
            points.append(TelemetryPoint(
                satellite_id=sat_id,
                timestamp=ts,
                subsystem="comms",
                parameter="byte_entropy",
                value=round(entropy, 4),
                unit="bits",
                quality=0.9,
            ))

            # Inter-frame gap — detects communication dropouts
            if prev_ts is not None:
                gap = abs((ts - prev_ts).total_seconds())
                if gap < 86400:  # ignore gaps > 1 day (different passes)
                    points.append(TelemetryPoint(
                        satellite_id=sat_id,
                        timestamp=ts,
                        subsystem="comms",
                        parameter="frame_gap",
                        value=round(gap, 1),
                        unit="s",
                        quality=0.85,
                    ))
            prev_ts = ts

        logger.info(
            "satnogs_converted",
            raw_frames=len(raw_frames),
            telemetry_points=len(points),
        )
        return points

    async def bulk_load_to_db(
        self,
        norad_ids: list[str],
        max_frames: int = 500,
        resample_minutes: int | None = None,
        skip_if_rows_gte: int | None = None,
        inter_page_delay: float = 2.0,
    ) -> dict[str, dict[str, int]]:
        """Fetch SatNOGS telemetry and bulk-insert into PostgreSQL.

        For each NORAD ID:
          1. Skip satellite if all signal parameters already have >= skip_if_rows_gte rows.
             Defaults to int(max_frames * 0.8) — accounts for frames that yield no valid
             data points (invalid hex, inter-day gaps filtered from frame_gap, etc.).
          2. Fetch raw frames via paginated API with rate-limit backoff.
          3. Extract four signal-level metrics per frame (frame_length, byte_mean,
             byte_entropy, frame_gap).
          4. Deduplicate timestamps, optionally resample, then bulk-insert.

        SatNOGS API limits respected:
          - 25 frames/page observed for high-traffic satellites (e.g. ISS)
          - 429 → Retry-After header honoured
          - max_frames hard cap per satellite (prevents runaway fetches)
          - inter_page_delay courtesy sleep between pages (default 2 s = 0.5 req/s)

        Returns {satellite_id: {parameter: rows_inserted_or_skipped}}.
        """
        # Local imports keep satnogs_fetcher.py usable without pandas/DB (e.g. in tests).
        from collections import defaultdict

        import pandas as pd

        from sentinel.db import queries
        from sentinel.ingest.bulk_loader import bulk_insert_channel, check_channel_row_count

        # ~10-15% of frames produce no valid data point (invalid hex, inter-day gaps).
        # Tie the skip threshold to max_frames so re-runs skip already-loaded satellites
        # instead of wasting API quota on duplicate inserts that ON CONFLICT silently drops.
        _skip_rows = skip_if_rows_gte if skip_if_rows_gte is not None else int(max_frames * 0.8)

        _PARAMETERS: tuple[str, ...] = ("frame_length", "byte_mean", "byte_entropy", "frame_gap")
        _UNITS: dict[str, str] = {
            "frame_length": "bytes",
            "byte_mean": "",
            "byte_entropy": "bits",
            "frame_gap": "s",
        }

        totals: dict[str, dict[str, int]] = {}

        for norad_id in norad_ids:
            sat_id = f"SATNOGS-{norad_id}"
            print(f"\n  Satellite {sat_id} (NORAD {norad_id})")

            existing = {p: await check_channel_row_count(sat_id, p) for p in _PARAMETERS}
            if all(cnt >= _skip_rows for cnt in existing.values()):
                print(f"    Already loaded (>= {_skip_rows} rows/param) — skipping fetch")
                totals[sat_id] = existing
                continue

            print(f"    Fetching up to {max_frames} frames ...")
            t0 = time.monotonic()
            raw_frames = await self.fetch_all_telemetry(
                norad_id,
                max_frames=max_frames,
                inter_page_delay=inter_page_delay,
            )
            if not raw_frames:
                print(f"    No frames returned — skipping")
                continue

            print(f"    Got {len(raw_frames)} frames in {time.monotonic() - t0:.1f}s")

            points = self.convert_to_points(raw_frames, satellite_id=sat_id)
            if not points:
                print(f"    No valid points extracted — skipping")
                continue

            by_param: dict[str, list[tuple]] = defaultdict(list)
            for pt in points:
                by_param[pt.parameter].append((pt.timestamp, pt.value))

            sat_totals: dict[str, int] = {}
            for param in _PARAMETERS:
                items = by_param.get(param, [])
                if not items:
                    sat_totals[param] = 0
                    continue

                items.sort(key=lambda x: x[0])
                series = pd.Series(
                    [v for _, v in items],
                    index=pd.DatetimeIndex([t for t, _ in items], tz="UTC"),
                    name=param,
                )
                # SatNOGS ground stations can produce duplicate timestamps — keep first.
                series = series[~series.index.duplicated(keep="first")]

                if resample_minutes and len(series) > 10:
                    series = series.resample(f"{resample_minutes}min").median().dropna()

                print(f"    {param}: {len(items):>6,} pts → {len(series):>6,} rows")

                await queries.upsert_satellite_seen(sat_id, series.index[0].to_pydatetime())
                await queries.upsert_channel_seen(sat_id, param, "comms", _UNITS[param])

                sat_totals[param] = await bulk_insert_channel(
                    satellite_id=sat_id,
                    channel_name=param,
                    subsystem="comms",
                    unit=_UNITS[param],
                    series=series,
                    quality=0.9,
                )

            totals[sat_id] = sat_totals

        return totals


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy of byte values — 0.0 to 8.0 bits."""
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _guess_subsystem(parameter_name: str) -> str:
    """Best-effort subsystem classification from parameter name.

    Order matters: more specific matches (comms, adcs, thermal) are checked
    before EPS which has broad keywords like 'power' and 'current'.
    """
    name = parameter_name.lower()
    # Comms first — 'radio_power' should be comms, not EPS
    if any(kw in name for kw in ("rssi", "signal", "link", "radio", "beacon", "antenna")):
        return "comms"
    if any(kw in name for kw in ("gyro", "wheel", "pointing", "attitude", "mag")):
        return "adcs"
    if any(kw in name for kw in ("temp", "thermal", "heat")):
        return "thermal"
    # EPS last — broad keywords like 'power', 'current'
    if any(kw in name for kw in ("batt", "solar", "voltage", "current", "power", "bus")):
        return "eps"
    return "unknown"
