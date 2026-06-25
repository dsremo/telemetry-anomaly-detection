"""Demo data seeding for the in-memory store.

Populates a small, evidently-synthetic satellite fleet so the dashboard,
anomaly feed, and benchmarks render on a public demo deployment that has no
PostgreSQL backend. Data is regenerated on every process start.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from dsremo.core.models import Anomaly, Severity, TelemetryPoint

_FLEET = ("AETHER-1", "AETHER-2")

_CHANNELS = (
    ("eps", "bus_voltage", "V", 28.0, 0.18),
    ("eps", "bus_current", "A", 3.4, 0.12),
    ("eps", "solar_array_power", "W", 118.0, 2.4),
    ("thermal", "battery_temp", "degC", 21.5, 0.6),
    ("thermal", "panel_temp", "degC", 12.0, 3.0),
    ("adcs", "gyro_rate", "deg/s", 0.02, 0.004),
    ("adcs", "reaction_wheel_rpm", "rpm", 2400.0, 40.0),
    ("comms", "downlink_snr", "dB", 11.6, 0.5),
)

_SEEDED_ANOMALIES = (
    ("AETHER-1", "eps", "bus_voltage", 24.1, Severity.WARNING, 0.88,
     "Bus voltage sagged 3.9 V below the nominal band during eclipse entry; battery discharge faster than the modelled depth-of-discharge curve."),
    ("AETHER-1", "thermal", "battery_temp", 41.3, Severity.CRITICAL, 0.95,
     "Battery temperature exceeded the 40 degC red limit; heater duty cycle inconsistent with the commanded thermal setpoint."),
    ("AETHER-1", "eps", "bus_current", 6.8, Severity.WARNING, 0.81,
     "Bus current spike to ~2x nominal with no corresponding load command — possible latch-up on the payload regulator."),
    ("AETHER-2", "adcs", "gyro_rate", 0.91, Severity.WATCH, 0.72,
     "Slow gyro-rate drift on the X axis over 6 hours; consistent with reaction-wheel momentum build-up approaching saturation."),
    ("AETHER-2", "comms", "downlink_snr", 3.2, Severity.CAUTION, 0.64,
     "Downlink SNR degraded ~8 dB during the pass; rain fade or antenna-pointing error under investigation."),
    ("AETHER-2", "thermal", "panel_temp", 78.4, Severity.CRITICAL, 0.93,
     "Solar-panel temperature excursion well above the survival limit while in sunlight; suspected loss of a thermal louver actuator."),
)


def _diurnal(base: float, amplitude: float, minute_of_day: int) -> float:
    fraction = (minute_of_day % 1440) / 1440.0
    return base + amplitude * math.sin(2 * math.pi * fraction)


async def seed_demo_data() -> dict[str, int]:
    """Insert a synthetic fleet's telemetry and a handful of anomalies."""
    from dsremo.db import memory_store

    generator = random.Random(20260625)
    now = datetime.now(timezone.utc)
    points: list[TelemetryPoint] = []

    for satellite_id in _FLEET:
        for step in range(72):
            stamp = now - timedelta(minutes=10 * (71 - step))
            minute_of_day = stamp.hour * 60 + stamp.minute
            for subsystem, parameter, unit, base, noise in _CHANNELS:
                amplitude = base * 0.015 if parameter != "panel_temp" else 18.0
                centre = _diurnal(base, amplitude, minute_of_day)
                value = centre + generator.gauss(0.0, noise)
                points.append(
                    TelemetryPoint(
                        satellite_id=satellite_id,
                        timestamp=stamp,
                        subsystem=subsystem,
                        parameter=parameter,
                        value=round(value, 4),
                        unit=unit,
                        quality=1.0,
                    )
                )
        await memory_store.upsert_satellite_seen(satellite_id, now)

    await memory_store.insert_telemetry(points)

    anomaly_count = 0
    for index, (satellite_id, subsystem, parameter, value, severity, confidence, explanation) in enumerate(_SEEDED_ANOMALIES):
        stamp = now - timedelta(minutes=17 * (index + 1))
        await memory_store.insert_anomaly(
            Anomaly(
                satellite_id=satellite_id,
                timestamp=stamp,
                subsystem=subsystem,
                parameter=parameter,
                value=value,
                severity=severity,
                confidence=confidence,
                explanation=explanation,
                verification_status="suspected",
            )
        )
        anomaly_count += 1

    return {"telemetry_points": len(points), "anomalies": anomaly_count}
