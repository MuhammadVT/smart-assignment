"""
Real geocoder backed by the U.S. Census Bureau's free, keyless Public
Geocoding Service:
https://geocoding.geo.census.gov/geocoder/

Chosen as the first real (non-mock) `Geocoder` implementation because it is
genuinely free -- no API key, no signup, no billing account, no posted rate
limit -- which fits a US-only domain like this one (Sysco). It has no
uptime SLA, so this module is built to fail *clearly*, not silently:
`AddressNotFoundError` for a bad/unmatched address (retrying won't help)
vs. `GeocodingServiceError` for a network/service problem (retrying might),
so callers -- see `tools/slot_recommendation.py` -- can react differently to
each.

[ASSUMPTION] This targets the documented `onelineaddress` endpoint and the
`Public_AR_Current` benchmark, both stable/long-standing per the Census
Bureau's own docs. This could not be smoke-tested against the live service
in this sandbox (outbound access to geocoding.geo.census.gov is blocked by
this environment's egress policy) -- run the live-only test in
tests/integrations/test_census_geocoder.py (RUN_LIVE_GEOCODER_TESTS=1) in an
environment with real network access before depending on this in production.

[REPLACEABLE] This is the seam to swap in the paid Google Maps Geocoding API
(or any other provider) later: add a new class implementing `Geocoder` (see
shared/geo.py) that raises the same two exceptions, then change the one
import site in tools/slot_recommendation.py. Nothing else needs to change.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

import requests

from smart_assignment.shared.geo import AddressNotFoundError, GeocodingServiceError
from smart_assignment.shared.models import GeoPoint

logger = logging.getLogger(__name__)

CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
CENSUS_BENCHMARK = "Public_AR_Current"

_REQUEST_TIMEOUT_SECONDS = 10
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0  # attempt N waits N * this, before the (N+1)th try


class CensusGeocoder:
    """
    `Geocoder` implementation backed by the Census Bureau's public service.
    See this module's docstring for the reasoning and the failure model.

    Results are cached process-wide by normalized address (see
    `_cached_lookup` below) -- shared across every `CensusGeocoder`
    instance, not per-instance -- because the conversational tools in
    tools/slot_recommendation.py re-geocode the same address on every tool
    call by design (see that module's docstring), and a real network call
    is not free the way MockGeocoder's was.
    """

    def geocode(self, address: str) -> GeoPoint:
        normalized = " ".join(address.split())
        if not normalized:
            raise AddressNotFoundError(address, "address is blank")
        return _cached_lookup(normalized)


@lru_cache(maxsize=2048)
def _cached_lookup(address: str) -> GeoPoint:
    # lru_cache only caches successful returns -- a raised exception here is
    # never cached, so a transient service failure doesn't poison future
    # lookups of the same address.
    return _lookup_with_retries(address)


def _lookup_with_retries(address: str) -> GeoPoint:
    """
    Only the network/transport level is retried -- a connection error or a
    5xx this second might genuinely succeed. A response that *arrived* but
    didn't parse (bad JSON, missing coordinates) is not transient: asking
    the exact same question again won't change a malformed answer, so that
    fails fast via `_parse_response` instead of burning attempts and time.
    """
    last_error: Optional[requests.RequestException] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = _request_once(address)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(
                "Census geocoder attempt %d/%d failed for %r: %s",
                attempt,
                _MAX_ATTEMPTS,
                address,
                exc,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
            continue
        return _parse_response(address, response)
    raise GeocodingServiceError(
        address, f"Census geocoder unreachable after {_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _request_once(address: str) -> requests.Response:
    response = requests.get(
        CENSUS_GEOCODER_URL,
        params={"address": address, "benchmark": CENSUS_BENCHMARK, "format": "json"},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # HTTP errors (5xx/4xx) are RequestException -> retried above
    return response


def _parse_response(address: str, response: requests.Response) -> GeoPoint:
    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingServiceError(
            address, "Census geocoder returned a non-JSON response"
        ) from exc

    matches = payload.get("result", {}).get("addressMatches") or []
    if not matches:
        raise AddressNotFoundError(address, "no address match found")

    try:
        coordinates = matches[0]["coordinates"]
        # Census returns x=longitude, y=latitude -- easy to transpose by accident.
        return GeoPoint(latitude=float(coordinates["y"]), longitude=float(coordinates["x"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingServiceError(
            address, f"Census geocoder response missing expected coordinates: {exc}"
        ) from exc
