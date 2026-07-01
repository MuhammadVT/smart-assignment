"""
[MOCK] Geocoding client — turns a street address into a GeoPoint.

Stands in for a real provider (Google Maps Geocoding API, or an internal
Sysco territory/geocoding service). Known demo addresses resolve to curated
Houston-area coordinates; anything else falls back to a deterministic
pseudo-location near the Houston metro so the pipeline never crashes on an
unrecognized address. Replace `MockGeocoder` with a real client implementing
the `Geocoder` protocol (see shared/geo.py) — nothing downstream changes.
"""

from __future__ import annotations

from smart_assignment.shared.models import GeoPoint

# Curated coordinates for the demo customer addresses (Houston metro).
_KNOWN_ADDRESSES: dict[str, GeoPoint] = {
    "1200 McKinney St, Houston, TX 77010": GeoPoint(29.7570, -95.3670),
    "5085 Westheimer Rd, Houston, TX 77056": GeoPoint(29.7400, -95.4630),
    "24600 Katy Fwy, Katy, TX 77494": GeoPoint(29.7830, -95.8240),
    "1201 Lake Woodlands Dr, The Woodlands, TX 77380": GeoPoint(30.1620, -95.4590),
}

# Rough Houston centroid for the deterministic fallback.
_HOUSTON_CENTER = GeoPoint(29.7604, -95.3698)


class MockGeocoder:
    """Deterministic offline geocoder for local development and tests."""

    def geocode(self, address: str) -> GeoPoint:
        if address in _KNOWN_ADDRESSES:
            return _KNOWN_ADDRESSES[address]
        # Deterministic small offset derived from the address text, so an
        # unknown address always maps to the same nearby point (no randomness).
        seed = sum(ord(c) for c in address)
        lat = _HOUSTON_CENTER.latitude + ((seed % 20) - 10) / 100.0
        lng = _HOUSTON_CENTER.longitude + ((seed % 17) - 8) / 100.0
        return GeoPoint(round(lat, 4), round(lng, 4))
