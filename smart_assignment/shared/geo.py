"""
Geo utilities shared across workflows: great-circle distance and a small
Geocoder abstraction.

Kept deterministic and dependency-free so the constraint/scoring logic is
unit-testable without any network or API key. The `Geocoder` protocol is the
seam where a real geocoding provider (Google Maps, an internal territory
service, etc.) gets swapped in — see `integrations/geocoding_client.py`.
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
