"""
Geo utilities shared across workflows: great-circle distance and a small
Geocoder abstraction.

Kept deterministic and dependency-free so the constraint/scoring logic is
unit-testable without any network or API key. The `Geocoder` protocol is the
seam where a real geocoding provider gets swapped in -- see
`integrations/geocoding_client.py` (MockGeocoder, the offline/test default)
and `integrations/census_geocoder.py` (CensusGeocoder, a real, free, US-only
implementation; swap in a paid provider like Google Maps behind the same
protocol later without touching any caller).

Every real `Geocoder` implementation should raise the exceptions below (not
ad hoc errors), so callers can handle "bad address" and "service problem"
distinctly regardless of which provider is behind the protocol.
"""

from __future__ import annotations

import math
from typing import Protocol

from smart_assignment.shared.models import GeoPoint

EARTH_RADIUS_MILES = 3958.8


def haversine_miles(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two points, in miles."""
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(h))


class Geocoder(Protocol):
    """Anything that can turn a street address into a GeoPoint."""

    def geocode(self, address: str) -> GeoPoint: ...


class GeocodingError(Exception):
    """Base for all real-geocoder failures -- always carries the address that failed."""

    def __init__(self, address: str, message: str):
        self.address = address
        super().__init__(f"{message} (address: {address!r})")


class AddressNotFoundError(GeocodingError):
    """The geocoder ran successfully but found no match for this address.

    Not transient -- retrying the same address won't help. The fix is a
    different/corrected address, so this should be relayed to whoever
    supplied it, not retried automatically.
    """


class GeocodingServiceError(GeocodingError):
    """The geocoder itself couldn't be reached or misbehaved: a network
    failure, timeout, HTTP error, or a response that didn't match the
    expected shape. Distinct from `AddressNotFoundError` because this is a
    service/transport problem, not a comment on the address -- callers may
    reasonably retry later.
    """
