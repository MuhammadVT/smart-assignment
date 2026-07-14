"""
The provider-agnostic `suggest` capability. MockGeocoder is exercised directly
(offline, deterministic); the Census parser is exercised with a canned payload
so no network is needed.
"""

from __future__ import annotations

from smart_assignment.integrations.census_geocoder import _parse_candidates
from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.shared.geo import AddressCandidate, supports_suggestions


def test_mock_geocoder_advertises_and_ranks_candidates():
    g = MockGeocoder()
    assert supports_suggestions(g)
    # A typo still surfaces the matching demo address, ranked by token overlap.
    cands = g.suggest("1200 McKiney St, Houston")
    assert cands and isinstance(cands[0], AddressCandidate)
    assert cands[0].formatted == "1200 McKinney St, Houston, TX 77010"


def test_mock_geocoder_returns_empty_for_no_overlap_and_blank():
    g = MockGeocoder()
    assert g.suggest("zzzzz qqqqq wwwww") == []
    assert g.suggest("") == []


def test_mock_geocoder_respects_limit():
    g = MockGeocoder()
    # "Houston" overlaps several demo addresses; limit caps the returned set.
    assert len(g.suggest("Houston TX", limit=1)) == 1


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_census_parse_candidates_maps_matches():
    payload = {
        "result": {
            "addressMatches": [
                {
                    "matchedAddress": "1200 MCKINNEY ST, HOUSTON, TX, 77010",
                    "coordinates": {"x": -95.367, "y": 29.757},
                    "addressComponents": {"city": "HOUSTON", "state": "TX"},
                },
                {"matchedAddress": "bad one", "coordinates": {"x": "nope"}},  # skipped
            ]
        }
    }
    cands = _parse_candidates("1200 mckinney", _Resp(payload), limit=5)
    assert len(cands) == 1  # the malformed match is skipped, not fatal
    assert cands[0].formatted == "1200 MCKINNEY ST, HOUSTON, TX, 77010"
    assert cands[0].location.latitude == 29.757 and cands[0].location.longitude == -95.367
    assert cands[0].components == {"city": "HOUSTON", "state": "TX"}


def test_census_parse_candidates_empty_when_no_matches():
    assert _parse_candidates("x", _Resp({"result": {"addressMatches": []}}), limit=5) == []
