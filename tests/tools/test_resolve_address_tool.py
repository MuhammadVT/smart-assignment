"""
The resolve_address conversational tool: the needs-confirmation suggestion, the
no-suggestion fallback, and the flag-off path. The resolver itself is faked so
this stays offline and focused on the tool's contract.
"""

from __future__ import annotations

import smart_assignment.tools.slot_recommendation as sr
from smart_assignment.address_resolve import ResolvedAddress
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressCandidate
from smart_assignment.shared.models import GeoPoint


class _Ctx:
    def __init__(self, state):
        self.state = state


def _with_address():
    return _Ctx(
        {sr._STATE_PROFILE_KEY: {"address": "1200 McKiney St, Houston", "order_quantity_cases": 90}}
    )


def test_needs_confirmation_when_a_suggestion_is_found(monkeypatch):
    resolved = ResolvedAddress(
        chosen=AddressCandidate("1200 McKinney St, Houston, TX 77010", GeoPoint(29.7, -95.3)),
        alternatives=[
            AddressCandidate("5085 Westheimer Rd, Houston, TX 77056", GeoPoint(29.7, -95.4))
        ],
        provenance="llm",
        rationale="closest street + city match",
    )
    monkeypatch.setattr(sr, "resolve_from_geocoder", lambda *a, **k: resolved)

    out = sr.resolve_address(_with_address())
    assert out["ok"] is True and out["needs_confirmation"] is True
    assert out["original_address"] == "1200 McKiney St, Houston"
    assert out["suggested_address"] == "1200 McKinney St, Houston, TX 77010"
    assert out["alternatives"] == ["5085 Westheimer Rd, Houston, TX 77056"]
    assert "Did you mean" in out["message"]


def test_no_suggestions_falls_back_to_double_check(monkeypatch):
    monkeypatch.setattr(sr, "resolve_from_geocoder", lambda *a, **k: None)
    out = sr.resolve_address(_with_address())
    assert out["ok"] is False and out["no_suggestions"] is True
    assert "double-check" in out["error"]


def test_resolver_error_degrades_to_double_check(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("backend down")

    monkeypatch.setattr(sr, "resolve_from_geocoder", boom)
    out = sr.resolve_address(_with_address())
    assert out["ok"] is False and out["no_suggestions"] is True


def test_flag_off_never_calls_the_resolver(monkeypatch):
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return None

    monkeypatch.setattr(sr, "resolve_from_geocoder", spy)
    monkeypatch.setattr(sr, "DEFAULT_CONFIG", Config(use_address_resolution=False))
    out = sr.resolve_address(_with_address())
    assert out["ok"] is False and out["no_suggestions"] is True
    assert called["n"] == 0  # the resolver is never consulted when the flag is off


def test_no_address_on_file():
    out = sr.resolve_address(_Ctx({}))
    assert out["ok"] is False
    assert "no address on file" in out["error"]
