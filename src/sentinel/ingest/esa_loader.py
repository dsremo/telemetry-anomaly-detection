"""ESA OPS-SAT dataset loader — real satellite telemetry integration.

Loads the ESA Anomaly Detection Benchmark (Mission1) dataset:
  - 76 telemetry channels from a real spacecraft
  - 6 subsystems, 13 years of operations (2000-2013)
  - 200 labeled anomalies (Anomaly + Rare Event categories)
  - Pandas pickle format, DatetimeIndex

This loader converts ESA's format into Sentinel's TelemetryPoint stream,
making the real satellite data flow through our detection pipeline
exactly like simulator data or customer data would.

Dataset structure:
  Resources/ESA-Mission1/channels.csv      — channel metadata
  Resources/ESA-Mission1/anomaly_types.csv — labeled anomalies
  Resources/ESA-Mission1/channels/channel_N.zip → channel_N (pickle)
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import structlog

from sentinel.core.models import TelemetryPoint

logger = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "Resources" / "ESA-Mission1"


@dataclass(frozen=True, slots=True)
class ChannelMeta:
    """Metadata for a single ESA telemetry channel."""

    name: str
    subsystem: str
    unit: str
    group: int
    is_target: bool  # True if this channel has labeled anomalies


@dataclass(frozen=True, slots=True)
class LabeledAnomaly:
    """A labeled anomaly from the ESA dataset — ground truth."""

    id: str
    anomaly_class: str
    subclass: str
    category: str        # "Anomaly", "Rare Event", "Communication Gap"
    dimensionality: str  # "Univariate" or "Multivariate"
    locality: str        # "Local" or "Global"
    length: str          # "Point" or "Subsequence"


class ESADataLoader:
    """Loads and streams ESA OPS-SAT telemetry for Sentinel processing."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self._channels_meta: dict[str, ChannelMeta] = {}
        self._anomalies: list[LabeledAnomaly] = []

    def load_metadata(self) -> None:
        """Load channel definitions and anomaly labels."""
        channels_file = self.data_dir / "channels.csv"
        anomalies_file = self.data_dir / "anomaly_types.csv"

        if not channels_file.exists():
            raise FileNotFoundError(
                f"ESA channels.csv not found at {channels_file}. "
                f"Extract ESA-Mission1.zip into Resources/"
            )

        # Load channel metadata
        with open(channels_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["Channel"]
                self._channels_meta[name] = ChannelMeta(
                    name=name,
                    subsystem=row["Subsystem"],
                    unit=row["Physical Unit"],
                    group=int(row["Group"]),
                    is_target=row["Target"] == "YES",
                )

        # Load anomaly labels
        if anomalies_file.exists():
            with open(anomalies_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._anomalies.append(LabeledAnomaly(
                        id=row["ID"],
                        anomaly_class=row["Class"],
                        subclass=row["Subclass"],
                        category=row.get("Category", ""),
                        dimensionality=row.get("Dimensionality", ""),
                        locality=row.get("Locality", ""),
                        length=row.get("Length", ""),
                    ))

        logger.info(
            "esa_metadata_loaded",
            channels=len(self._channels_meta),
            target_channels=sum(1 for c in self._channels_meta.values() if c.is_target),
            anomalies=len(self._anomalies),
        )

    @property
    def channel_names(self) -> list[str]:
        return list(self._channels_meta.keys())

    @property
    def target_channels(self) -> list[str]:
        """Channels that have labeled anomalies — most useful for benchmarking."""
        return [n for n, m in self._channels_meta.items() if m.is_target]

    @property
    def anomaly_labels(self) -> list[LabeledAnomaly]:
        return self._anomalies

    def load_channel(self, channel_name: str) -> pd.DataFrame:
        """Load a single channel's time-series data.

        Handles both pre-extracted pickle files and nested zips.
        Returns a DataFrame with DatetimeIndex and one float column.
        """
        pickle_path = self.data_dir / "channels" / channel_name
        zip_path = self.data_dir / "channels" / f"{channel_name}.zip"

        if pickle_path.exists() and pickle_path.stat().st_size > 100:
            return pd.read_pickle(pickle_path)

        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(channel_name) as f:
                    return pd.read_pickle(f)

        raise FileNotFoundError(
            f"Channel data not found: {channel_name}. "
            f"Extract from ESA-Mission1.zip first."
        )

    def stream_channel(
        self,
        channel_name: str,
        satellite_id: str = "ESA-MISSION1",
        sample_rate: int = 1,
        max_points: int | None = None,
    ) -> Iterator[TelemetryPoint]:
        """Stream a channel's data as TelemetryPoint objects.

        sample_rate: take every Nth point (1=all, 10=every 10th, etc.)
                     ESA data has 10M+ points per channel — sampling avoids OOM.
        max_points: stop after this many points (None=all)
        """
        meta = self._channels_meta.get(channel_name)
        if not meta:
            raise ValueError(f"Unknown channel: {channel_name}. Load metadata first.")

        df = self.load_channel(channel_name)
        count = 0

        for i, (ts, row) in enumerate(df.iterrows()):
            if i % sample_rate != 0:
                continue

            # Ensure timezone-aware timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            yield TelemetryPoint(
                satellite_id=satellite_id,
                timestamp=ts,
                subsystem=meta.subsystem,
                parameter=channel_name,
                value=float(row.iloc[0]),
                unit=meta.unit,
                quality=1.0,
            )

            count += 1
            if max_points and count >= max_points:
                break

    def load_channels_as_matrix(
        self,
        channel_names: list[str] | None = None,
        sample_rate: int = 100,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load multiple channels into a single aligned DataFrame.

        Useful for multivariate Isolation Forest training.
        Channels are resampled to a common time grid via forward-fill.

        Returns (dataframe, column_names)
        """
        names = channel_names or self.target_channels[:10]  # default: first 10 target channels
        frames = {}

        for name in names:
            try:
                df = self.load_channel(name)
                # Sample to reduce memory
                sampled = df.iloc[::sample_rate]
                frames[name] = sampled.iloc[:, 0]
            except FileNotFoundError:
                logger.warning("esa_channel_not_found", channel=name)

        if not frames:
            raise ValueError("No channels could be loaded")

        combined = pd.DataFrame(frames)
        combined = combined.ffill().dropna()

        logger.info(
            "esa_matrix_loaded",
            channels=len(frames),
            samples=len(combined),
            date_range=f"{combined.index.min()} → {combined.index.max()}",
        )

        return combined, list(combined.columns)

    def get_subsystem_map(self) -> dict[str, list[str]]:
        """Group channels by subsystem — matches Sentinel's subsystem concept."""
        groups: dict[str, list[str]] = {}
        for name, meta in self._channels_meta.items():
            groups.setdefault(meta.subsystem, []).append(name)
        return groups

    def summary(self) -> dict:
        """Return a human-readable summary of the loaded dataset."""
        subsystems = self.get_subsystem_map()
        categories = {}
        for a in self._anomalies:
            categories[a.category] = categories.get(a.category, 0) + 1

        return {
            "total_channels": len(self._channels_meta),
            "target_channels": len(self.target_channels),
            "subsystems": {k: len(v) for k, v in subsystems.items()},
            "anomaly_labels": len(self._anomalies),
            "anomaly_categories": categories,
            "data_dir": str(self.data_dir),
        }
