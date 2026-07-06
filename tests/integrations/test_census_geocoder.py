"""
Unit tests for CensusGeocoder (integrations/census_geocoder.py). All HTTP
calls are mocked -- these never touch the real Census service, so the
suite stays offline and deterministic. See test_live_census_geocoder_smoke
at the bottom for an opt-in test against the real API.
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
import requests

from smart_assignment.integrations import census_geocoder as census_geocoder_module
from smart_assignment.integrations.census_geocoder import CensusGeocoder
from smart_assignment.shared.geo import AddressNotFoundError, GeocodingServiceError

_ADDRESS = "1200 McKinney St, Houston, TX 77010"


@pytest.fixture(autouse=True)
def _clear_geocode_cache():
    # The lookup cache is process-wide (shared across CensusGeocoder
    # instances by design -- see the module docstring), so it must be reset
    # between tests or an earlier test's result leaks into a later one.
    census_geocoder_module._cached_lookup.cache_clear()
    yield
    census_geocoder_module._cached_lookup.cache_clear()


def _match_response(lat: float, lon: float) -> Mock:
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {
        "result": {"addressMatches": [{"coordinates": {"x": lon, "y": lat}}]}
    }
    return resp


def _no_match_response() -> Mock:
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": {"addressMatches": []}}
    return resp


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_successful_match_maps_x_to_longitude_and_y_to_latitude(mock_get):
    mock_get.return_value = _match_response(lat=29.7570, lon=-95.3670)
    point = CensusGeocoder().geocode(_ADDRESS)
    assert point.latitude == 29.7570
    assert point.longitude == -95.3670


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_no_match_raises_address_not_found(mock_get):
    mock_get.return_value = _no_match_response()
    with pytest.raises(AddressNotFoundError) as exc_info:
        CensusGeocoder().geocode(_ADDRESS)
    assert exc_info.value.address == _ADDRESS


def test_blank_address_raises_without_any_http_call():
    with patch("smart_assignment.integrations.census_geocoder.requests.get") as mock_get:
        with pytest.raises(AddressNotFoundError):
            CensusGeocoder().geocode("   ")
        mock_get.assert_not_called()


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_malformed_json_raises_service_error(mock_get):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.side_effect = ValueError("not json")
    mock_get.return_value = resp
    with pytest.raises(GeocodingServiceError):
        CensusGeocoder().geocode(_ADDRESS)


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_missing_coordinates_raises_service_error(mock_get):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": {"addressMatches": [{"coordinates": {}}]}}
    mock_get.return_value = resp
    with pytest.raises(GeocodingServiceError):
        CensusGeocoder().geocode(_ADDRESS)


@patch("smart_assignment.integrations.census_geocoder.time.sleep")
@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_transient_network_error_is_retried_then_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        requests.ConnectionError("boom"),
        _match_response(lat=29.75, lon=-95.36),
    ]
    point = CensusGeocoder().geocode(_ADDRESS)
    assert point.latitude == 29.75
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()  # backoff happened between the two attempts


@patch("smart_assignment.integrations.census_geocoder.time.sleep")
@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_persistent_network_error_raises_service_error_after_max_attempts(mock_get, mock_sleep):
    mock_get.side_effect = requests.ConnectionError("boom")
    with pytest.raises(GeocodingServiceError):
        CensusGeocoder().geocode(_ADDRESS)
    assert mock_get.call_count == census_geocoder_module._MAX_ATTEMPTS


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_address_not_found_is_not_retried(mock_get):
    # Not transient -- a bad address should fail fast, not burn 3 requests.
    mock_get.return_value = _no_match_response()
    with pytest.raises(AddressNotFoundError):
        CensusGeocoder().geocode(_ADDRESS)
    assert mock_get.call_count == 1


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_repeated_geocode_of_same_address_hits_network_once(mock_get):
    mock_get.return_value = _match_response(lat=29.7570, lon=-95.3670)
    geocoder = CensusGeocoder()
    first = geocoder.geocode(_ADDRESS)
    second = CensusGeocoder().geocode(_ADDRESS)  # even a fresh instance shares the cache
    assert first == second
    assert mock_get.call_count == 1


@patch("smart_assignment.integrations.census_geocoder.requests.get")
def test_failed_lookups_are_not_cached(mock_get):
    mock_get.return_value = _no_match_response()
    with pytest.raises(AddressNotFoundError):
        CensusGeocoder().geocode(_ADDRESS)

    mock_get.return_value = _match_response(lat=29.7570, lon=-95.3670)
    point = CensusGeocoder().geocode(_ADDRESS)  # should hit the network again, not a cached failure
    assert point.latitude == 29.7570
    assert mock_get.call_count == 2


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_GEOCODER_TESTS"),
    reason=(
        "Opt-in only: hits the real Census geocoder over the network. "
        "Set RUN_LIVE_GEOCODER_TESTS=1 to run it (could not be verified from "
        "this sandbox -- its egress policy blocks geocoding.geo.census.gov)."
    ),
)
def test_live_census_geocoder_smoke():
    point = CensusGeocoder().geocode("1200 McKinney St, Houston, TX 77010")
    assert 29.0 < point.latitude < 30.5
    assert -96.0 < point.longitude < -94.5
