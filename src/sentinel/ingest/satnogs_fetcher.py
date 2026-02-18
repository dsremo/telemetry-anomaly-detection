"""SatNOGS API fetcher — pulls live telemetry from the SatNOGS network.

SatNOGS is the world's largest open-source satellite ground station network.
Their DB has telemetry from hundreds of real satellites collected by
community ground stations worldwide.

API: https://db.satnogs.org/api/
Auth: Token-based (stored in .env as SATNOGS_API_TOKEN)

Recommended satellites for Sentinel testing:
  - ROBUSTA-3A:  ~700k frames, academic CubeSat
  - Monitor-3:   Stable telemetry, long history
  - ITASAT-1:    Clean EPS + thermal data
  - LEOPARD:     Modern mission, good continuity
"""

from __future__ import annotations

import os
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

        Returns raw JSON responses from SatNOGS. Use convert_to_points()
        to transform into Sentinel's format.
        """
        if not self.api_token:
            raise ValueError("SATNOGS_API_TOKEN not set — check .env file")

        url = f"{SATNOGS_API_BASE}/telemetry/"
        params = {
            "satellite": satellite_norad_id,
            "limit": min(limit, 500),  # SatNOGS caps at 500 per page
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        # SatNOGS API may return paginated dict {"count": N, "results": [...]}
        # or a plain list, depending on the endpoint version
        if isinstance(data, dict):
            frames = data.get("results", [])
        else:
            frames = data

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

        SatNOGS frames contain decoded telemetry as JSON. The structure
        varies by satellite — this extracts what we can generically.
        """
        points: list[TelemetryPoint] = []

        for frame in raw_frames:
            # SatNOGS API can return raw hex strings for undecoded frames — skip them
            if not isinstance(frame, dict):
                continue

            sat_id = satellite_id or str(frame.get("norad_cat_id", "UNKNOWN"))
            timestamp_str = frame.get("timestamp", "")

            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            # SatNOGS decoded data is in 'decoded' field (if available)
            decoded = frame.get("decoded", {})
            if not decoded or not isinstance(decoded, dict):
                # Raw frame or non-dict decoded — skip
                continue

            # Flatten decoded telemetry into individual points
            for key, value in _flatten_dict(decoded):
                if not isinstance(value, (int, float)):
                    continue

                subsystem = _guess_subsystem(key)
                points.append(TelemetryPoint(
                    satellite_id=sat_id,
                    timestamp=ts,
                    subsystem=subsystem,
                    parameter=key,
                    value=float(value),
                    unit="",
                    quality=0.9,  # slightly lower quality since community-collected
                ))

        logger.info(
            "satnogs_converted",
            raw_frames=len(raw_frames),
            telemetry_points=len(points),
        )
        return points


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Recursively flatten a nested dict into (key, value) pairs."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, full_key))
        else:
            items.append((full_key, v))
    return items


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
