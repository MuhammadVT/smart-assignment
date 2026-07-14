"""
Resolver orchestration + geocoder feature-detection. Offline: every LLM
interaction is a fake choice_fn, and the geocoder is a small in-memory fake.
"""

from __future__ import annotations

from smart_assignment.address_resolve import (
    resolve_address,
    resolve_from_geocoder,
    suggest_addresses,
)
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate, GeocodingServiceError
from smart_assignment.shared.models import GeoPoint

MCKINNEY = AddressCandidate("1200 McKinney St, Houston, TX 77010", GeoPoint(29.7, -95.3))
WESTHEIMER = AddressCandidate("5085 Westheimer Rd, Houston, TX 77056", GeoPoint(29.7, -95.4))


def _pick(idx, value=None):
    """A fake choice_fn that picks candidate `idx`, citing its similarity."""

    def fn(config, prompt):
        cites = [] if value is None else [{"index": idx, "field": "similarity", "value": value}]
        return {"chosen_index": idx, "rationale": f"candidate {idx}", "citations": cites}

    return fn


def test_no_candidates_returns_none():
    assert resolve_address("q", [], Config(), choice_fn=_pick(0)) is None


def test_grounded_pick_is_used_and_alternatives_exclude_it():
    cands = [WESTHEIMER, MCKINNEY]
    # McKinney is index 1; the model picks it even though similarity ranks it top too.
    resolved = resolve_address("1200 McKinney St, Houston", cands, Config(), choice_fn=_pick(1))
    assert resolved.chosen.formatted == MCKINNEY.formatted
    assert resolved.provenance == "llm"
    assert [c.formatted for c in resolved.alternatives] == [WESTHEIMER.formatted]
    assert resolved.rationale == "candidate 1"


def test_falls_back_to_highest_similarity_on_bad_index():
    cands = [WESTHEIMER, MCKINNEY]  # McKinney (idx 1) has the higher similarity

    def bad(config, prompt):
        return {"chosen_index": 9, "rationale": "invalid", "citations": []}

    resolved = resolve_address("1200 McKinney St, Houston", cands, Config(), choice_fn=bad)
    assert resolved.provenance == "deterministic"
    assert resolved.chosen.formatted == MCKINNEY.formatted  # highest-similarity candidate
    assert resolved.rationale is None


def test_falls_back_when_choice_fn_raises():
    def boom(config, prompt):
        raise RuntimeError("no creds")

    resolved = resolve_address("1200 McKinney St, Houston", [MCKINNEY], Config(), choice_fn=boom)
    assert resolved.provenance == "deterministic"
    assert resolved.chosen.formatted == MCKINNEY.formatted


# --- geocoder feature-detection ---------------------------------------------


class _PlainGeocoder:
    def geocode(self, address):  # pragma: no cover - not exercised
        raise NotImplementedError


class _SuggestGeocoder:
    def __init__(self, candidates=None, error=None):
        self._candidates = candidates or []
        self._error = error

    def geocode(self, address):  # pragma: no cover
        raise NotImplementedError

    def suggest(self, address, *, limit=5):
        if self._error:
            raise self._error
        return self._candidates[:limit]


def test_suggest_addresses_empty_for_non_suggesting_geocoder():
    assert suggest_addresses("x", _PlainGeocoder()) == []


def test_suggest_addresses_swallows_service_errors():
    g = _SuggestGeocoder(error=GeocodingServiceError("x", "down"))
    assert suggest_addresses("x", g) == []


def test_resolve_from_geocoder_end_to_end():
    g = _SuggestGeocoder(candidates=[WESTHEIMER, MCKINNEY])
    resolved = resolve_from_geocoder(
        "1200 McKinney St, Houston", g, Config(), choice_fn=_pick(1)
    )
    assert resolved is not None
    assert resolved.chosen.formatted == MCKINNEY.formatted


def test_resolve_from_geocoder_none_when_no_candidates():
    g = _SuggestGeocoder(candidates=[])
    assert resolve_from_geocoder("x", g, Config(), choice_fn=_pick(0)) is None
