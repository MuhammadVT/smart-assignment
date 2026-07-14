"""
[MOCK] Geocoding client — turns a street address into a GeoPoint.

Stands in for a real provider. Known demo addresses resolve to curated
Houston-area coordinates; anything else falls back to a deterministic
pseudo-location near the Houston metro so the pipeline never crashes on an
unrecognized address. This is deliberately kept as the default for
`pipeline.run_slot_recommendation(...)`, `scripts/run_local.py`, the GitHub
Pages generator, and the test suite, so all of those stay fully offline,
deterministic, and reproducible with no network/API key.

For a real geocoder, see `integrations/census_geocoder.py`'s
`CensusGeocoder` (used by the conversational agent's tools, see
`tools/slot_recommendation.py`) — it implements the same `Geocoder` protocol
(see shared/geo.py), so nothing downstream changes when swapping providers.
"""

from __future__ import annotations

import logging
import os

from smart_assignment.shared.geo import Geocoder
from smart_assignment.shared.models import GeoPoint

logger = logging.getLogger(__name__)

_GEOCODER_ENV = "SMART_ASSIGNMENT_GEOCODER"

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


def resolve_geocoder() -> Geocoder:
    """The geocoder every surface should use, chosen by SMART_ASSIGNMENT_GEOCODER:

      - "census" → the live US Census geocoder (real, accurate coordinates).
                   **Default.**
      - "mock"   → the deterministic offline `MockGeocoder` (curated coords for
                   the demo addresses; a stable pseudo-point otherwise), for fully
                   offline runs and tests.

    Resolving in ONE place means the conversational tools, the deterministic
    web-app path, and `run_slot_recommendation` all pick the same provider from
    the same config instead of quietly diverging -- so `adk web` and the web app
    (which now load the same .env; see smart_assignment/__init__.py) agree.
    """
    choice = os.environ.get(_GEOCODER_ENV, "census").strip().lower()
    if choice in ("mock", "offline"):
        return MockGeocoder()
    if choice not in ("census", "live", "real"):
        logger.warning("Unknown %s=%r; using the live census geocoder.", _GEOCODER_ENV, choice)
    # Imported lazily so an offline/mock run never pulls the HTTP client.
    from smart_assignment.integrations.census_geocoder import CensusGeocoder

    return CensusGeocoder()
