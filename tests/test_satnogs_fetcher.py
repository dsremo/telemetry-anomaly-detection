"""Tests for the SatNOGS API fetcher.

Uses mocked HTTP responses — no real API calls needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sentinel.ingest.satnogs_fetcher import (
    SatNOGSFetcher,
    _flatten_dict,
    _guess_subsystem,
)


class TestFlattenDict:
    def test_flat_dict(self):
        result = _flatten_dict({"a": 1, "b": 2})
        assert ("a", 1) in result
        assert ("b", 2) in result

    def test_nested_dict(self):
        result = _flatten_dict({"eps": {"voltage": 7.4, "current": 1.2}})
        assert ("eps.voltage", 7.4) in result
        assert ("eps.current", 1.2) in result

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 42}}})
        assert ("a.b.c", 42) in result

    def test_empty_dict(self):
        assert _flatten_dict({}) == []

    def test_mixed_types(self):
        result = _flatten_dict({"name": "sat", "value": 3.14, "count": 5})
        assert ("name", "sat") in result
        assert ("value", 3.14) in result
        assert ("count", 5) in result


class TestGuessSubsystem:
    def test_eps_keywords(self):
        assert _guess_subsystem("battery_voltage") == "eps"
        assert _guess_subsystem("solar_array_current") == "eps"
        assert _guess_subsystem("bus_power") == "eps"

    def test_adcs_keywords(self):
        assert _guess_subsystem("gyro_x") == "adcs"
        assert _guess_subsystem("reaction_wheel_speed") == "adcs"
        assert _guess_subsystem("magnetometer_z") == "adcs"
        assert _guess_subsystem("attitude_error") == "adcs"

    def test_thermal_keywords(self):
        assert _guess_subsystem("panel_temperature") == "thermal"
        assert _guess_subsystem("thermal_control") == "thermal"
        assert _guess_subsystem("heater_status") == "thermal"

    def test_comms_keywords(self):
        assert _guess_subsystem("rssi_level") == "comms"
        assert _guess_subsystem("signal_strength") == "comms"
        assert _guess_subsystem("radio_power") == "comms"
        assert _guess_subsystem("beacon_interval") == "comms"

    def test_unknown_fallback(self):
        assert _guess_subsystem("some_random_param") == "unknown"
        assert _guess_subsystem("cpu_load") == "unknown"


class TestSatNOGSFetcher:
    def test_init_with_token(self):
        fetcher = SatNOGSFetcher(api_token="test-token-123")
        assert fetcher.api_token == "test-token-123"

    def test_init_without_token_warns(self, monkeypatch):
        # Should not raise, just warn (clear env so _load_dotenv doesn't find it)
        monkeypatch.delenv("SATNOGS_API_TOKEN", raising=False)
        monkeypatch.setattr("sentinel.core.config._DOTENV_PATH", type("P", (), {"exists": lambda self: False})())
        fetcher = SatNOGSFetcher(api_token="")
        assert fetcher.api_token == ""

    def test_headers_include_token(self):
        fetcher = SatNOGSFetcher(api_token="my-token")
        headers = fetcher._headers
        assert headers["Authorization"] == "Token my-token"

    @pytest.mark.asyncio
    async def test_fetch_without_token_raises(self, monkeypatch):
        monkeypatch.delenv("SATNOGS_API_TOKEN", raising=False)
        monkeypatch.setattr("sentinel.core.config._DOTENV_PATH", type("P", (), {"exists": lambda self: False})())
        fetcher = SatNOGSFetcher(api_token="")
        with pytest.raises(ValueError, match="SATNOGS_API_TOKEN not set"):
            await fetcher.fetch_telemetry("12345")


class TestConvertToPoints:
    @pytest.fixture
    def fetcher(self) -> SatNOGSFetcher:
        return SatNOGSFetcher(api_token="test")

    def test_converts_decoded_frames(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 44830,
                "timestamp": "2024-03-15T10:30:00Z",
                "decoded": {
                    "eps": {"battery_voltage": 7.4, "solar_current": 1.2},
                    "thermal": {"panel_temp": 25.3},
                },
            }
        ]
        points = fetcher.convert_to_points(raw)
        assert len(points) == 3
        assert all(p.satellite_id == "44830" for p in points)
        assert all(p.quality == 0.9 for p in points)

    def test_skips_non_numeric_values(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 44830,
                "timestamp": "2024-03-15T10:30:00Z",
                "decoded": {
                    "status": "nominal",  # string — skipped
                    "battery_voltage": 7.4,  # float — kept
                },
            }
        ]
        points = fetcher.convert_to_points(raw)
        assert len(points) == 1
        assert points[0].parameter == "battery_voltage"

    def test_skips_frames_without_decoded(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 44830,
                "timestamp": "2024-03-15T10:30:00Z",
                # no 'decoded' field
            }
        ]
        points = fetcher.convert_to_points(raw)
        assert len(points) == 0

    def test_skips_invalid_timestamps(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 44830,
                "timestamp": "not-a-date",
                "decoded": {"voltage": 7.4},
            }
        ]
        points = fetcher.convert_to_points(raw)
        assert len(points) == 0

    def test_custom_satellite_id(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "timestamp": "2024-03-15T10:30:00Z",
                "decoded": {"voltage": 7.4},
            }
        ]
        points = fetcher.convert_to_points(raw, satellite_id="CUSTOM-SAT")
        assert len(points) == 1
        assert points[0].satellite_id == "CUSTOM-SAT"

    def test_subsystem_classification(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 99999,
                "timestamp": "2024-01-01T00:00:00Z",
                "decoded": {
                    "battery_voltage": 7.4,
                    "gyro_rate_x": 0.01,
                    "panel_temperature": 20.0,
                    "rssi_db": -90.0,
                    "cpu_usage": 45.0,
                },
            }
        ]
        points = fetcher.convert_to_points(raw)
        subsystems = {p.parameter: p.subsystem for p in points}
        assert subsystems["battery_voltage"] == "eps"
        assert subsystems["gyro_rate_x"] == "adcs"
        assert subsystems["panel_temperature"] == "thermal"
        assert subsystems["rssi_db"] == "comms"
        assert subsystems["cpu_usage"] == "unknown"

    def test_empty_input(self, fetcher: SatNOGSFetcher):
        assert fetcher.convert_to_points([]) == []

    def test_timestamp_parsed_correctly(self, fetcher: SatNOGSFetcher):
        raw = [
            {
                "norad_cat_id": 44830,
                "timestamp": "2024-06-15T14:30:00Z",
                "decoded": {"voltage": 7.0},
            }
        ]
        points = fetcher.convert_to_points(raw)
        assert points[0].timestamp.year == 2024
        assert points[0].timestamp.month == 6
        assert points[0].timestamp.hour == 14
        assert points[0].timestamp.tzinfo is not None
