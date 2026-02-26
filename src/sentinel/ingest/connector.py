"""DataConnector ABC — common interface for all telemetry ingest sources.

All connectors (SatNOGSFetcher, ESADataLoader, CSVConnector, …) inherit from
this class.  Connector-specific configuration (API tokens, file paths, satellite
IDs) belongs in __init__; bulk_load_to_db() can then be called without extra
arguments by generic pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
