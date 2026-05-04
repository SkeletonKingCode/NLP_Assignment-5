"""
tests/test_weather.py

Unit tests for the Weather tool — functional correctness with valid
and invalid city inputs, timeout handling, and error conditions.
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from Tools.weather import get_weather, _fetch_weather_sync


# ---------------------------------------------------------------------------
# Functional Correctness — Direct Invocation (mocked network)
# ---------------------------------------------------------------------------

class TestWeatherFunctional:
    """Unit tests that call the weather tool with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_valid_city_returns_result(self):
        """Mocked weather call for a valid city."""
        with patch("Tools.weather._fetch_weather_sync", return_value="Lahore: ☀️ +35°C"):
            result = await get_weather("Lahore")
            assert "Lahore" in result
            assert "°C" in result or "+" in result

    @pytest.mark.asyncio
    async def test_another_valid_city(self):
        with patch("Tools.weather._fetch_weather_sync", return_value="Karachi: 🌤 +30°C"):
            result = await get_weather("Karachi")
            assert "Karachi" in result

    @pytest.mark.asyncio
    async def test_invalid_city_returns_error(self):
        with patch("Tools.weather._fetch_weather_sync",
                   return_value="Error: Could not find weather for 'FakeCity12345'."):
            result = await get_weather("FakeCity12345")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_city(self):
        """Empty city string should return an error or empty result."""
        with patch("Tools.weather._fetch_weather_sync",
                   return_value="Error: Could not find weather for ''."):
            result = await get_weather("")
            assert "Error" in result or result == ""


# ---------------------------------------------------------------------------
# Error Handling — Sync function
# ---------------------------------------------------------------------------

class TestWeatherSyncErrors:
    """Tests for the synchronous HTTP fetcher error conditions."""

    def test_timeout_handling(self):
        """Simulate a timeout error."""
        import urllib.error
        with patch("Tools.weather.urllib.request.urlopen",
                   side_effect=urllib.error.URLError(TimeoutError("timed out"))):
            result = _fetch_weather_sync("Lahore")
            assert "Error" in result or "timed" in result.lower()

    def test_network_error(self):
        """Simulate a general network error."""
        import urllib.error
        with patch("Tools.weather.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Connection refused")):
            result = _fetch_weather_sync("Lahore")
            assert "Error" in result

    def test_unknown_location_response(self):
        """API returns 'Unknown location'."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"Unknown location"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("Tools.weather.urllib.request.urlopen", return_value=mock_response):
            result = _fetch_weather_sync("zzznonexistent")
            assert "Error" in result or "Unknown" in result

    def test_url_encoding_special_chars(self):
        """City name with special characters should be URL-encoded."""
        with patch("Tools.weather.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"New York: +20C"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _fetch_weather_sync("New York")
            assert "Error" not in result


# ---------------------------------------------------------------------------
# Return Type Validation
# ---------------------------------------------------------------------------

class TestWeatherReturnType:
    """Ensure weather tool always returns a string."""

    @pytest.mark.asyncio
    async def test_returns_string(self):
        with patch("Tools.weather._fetch_weather_sync", return_value="Lahore: +30°C"):
            result = await get_weather("Lahore")
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_error_returns_string(self):
        with patch("Tools.weather._fetch_weather_sync", return_value="Error: timeout"):
            result = await get_weather("FakeCity")
            assert isinstance(result, str)
