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

        async with httpx.AsyncClient(timeout=30.0) as client:
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
