"""CSVConnector — load wide-format CSV telemetry into Sentinel DB.

CSV format (wide):
    timestamp,param1,param2,...
    2024-01-01T00:00:00Z,1.2,3.4,...

The timestamp column is used as the DatetimeIndex; all other columns are
treated as telemetry parameters for the given satellite.  Naive timestamps
are assumed to be UTC and localized automatically.

Typical usage::

    connector = CSVConnector(
        file_path="telemetry.csv",
        satellite_id="MYSAT-1",
        subsystem="eps",
    )
    totals = await connector.bulk_load_to_db(resample_minutes=5)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from sentinel.ingest.bulk_loader import bulk_insert_channel, check_channel_row_count
from sentinel.ingest.connector import DataConnector

logger = structlog.get_logger()


class CSVConnector(DataConnector):
    """Load wide-format CSV telemetry into the Sentinel DB.

    Args:
        file_path:      Path to the CSV file.
        satellite_id:   Sentinel satellite identifier (written to every row).
        subsystem:      Subsystem label applied to all parameters in the file.
                        Use a separate connector per subsystem for mixed files.
        timestamp_col:  Column name to use as the time index (default: "timestamp").
    """

    def __init__(
        self,
        file_path: Path | str,
        satellite_id: str,
        subsystem: str = "unknown",
        timestamp_col: str = "timestamp",
    ):
        self.file_path = Path(file_path)
        self.satellite_id = satellite_id
        self.subsystem = subsystem
        self.timestamp_col = timestamp_col

    @property
    def source_name(self) -> str:
        return f"csv:{self.file_path.name}"

    async def bulk_load_to_db(
        self,
        *,
        resample_minutes: int = 1,
        skip_if_rows_gte: int = 50_000,
    ) -> dict[str, int]:
        """Parse the CSV and bulk-insert each parameter column into the DB.

        Steps per parameter column:
          1. Check existing row count — skip if >= skip_if_rows_gte.
          2. Drop NaN values (already handled in bulk_insert_channel, but
             explicit here for clarity).
          3. Resample to resample_minutes resolution (median) if > 1.
          4. Insert via bulk_insert_channel (UNNEST batches).

        Returns:
            {parameter_name: rows_inserted_or_existing} for every column.
        """
        df = pd.read_csv(
            self.file_path,
            parse_dates=[self.timestamp_col],
            index_col=self.timestamp_col,
        )

        if df.empty:
            logger.info("csv_empty_file", file=str(self.file_path))
            return {}

        # Ensure timezone-aware index — PostgreSQL timestamptz requires it.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        if resample_minutes > 1:
            df = df.resample(f"{resample_minutes}min").median()

        totals: dict[str, int] = {}

        for col in df.columns:
            existing = await check_channel_row_count(self.satellite_id, col)
            if existing >= skip_if_rows_gte:
                logger.info(
                    "csv_channel_skipped",
                    satellite=self.satellite_id,
                    param=col,
                    existing_rows=existing,
                )
                totals[col] = existing
                continue

            series = df[col].dropna()
            inserted = await bulk_insert_channel(
                satellite_id=self.satellite_id,
                channel_name=col,
                subsystem=self.subsystem,
                unit="",
                series=series,
            )
            totals[col] = inserted
            logger.info(
                "csv_channel_loaded",
                satellite=self.satellite_id,
                param=col,
                rows_inserted=inserted,
            )

        return totals
