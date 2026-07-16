"""
The `SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT` flag must actually take effect in
`run_slot_recommendation` (the offline demo, page generator, and web app all go
through it) -- not only in the conversational tool.

Regression guard for the bug where the flag was silently ignored everywhere
except `recommend_or_escalate`. Offline: `default_judge` is monkeypatched to a
`GroundedJudge` driven by a fake `judgment_fn`, so no network/credentials.
"""

from __future__ import annotations

import json

from smart_assignment import judgment, pipeline
from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.judgment import GroundedJudge
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, Decision

_MARKER = "GROUNDED-PATH-RATIONALE-MARKER"


def _bayou() -> CustomerProfile:
    # Fresh instance each call -- run_slot_recommendation mutates .location.
    return CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
    )


def _fake_fn(config, prompt):
    """A grounded reply that recommends the first feasible route in the packet
    with a recognizable rationale, so we can prove the grounded path ran."""
    body = prompt.split("EVIDENCE PACKET:", 1)[1].split("Reply with a SINGLE", 1)[0]
    packet = json.loads(body.strip())
    first = packet["feasible_candidates"][0]
    return {
        "decision": "RECOMMEND",
        "confidence": "HIGH",
        "recommended_route_id": first["route_id"],
        "rationale": _MARKER,
        "citations": [
            {
                "route_id": first["route_id"],
                "field": "utilization_after",
                "value": first["facts"]["utilization_after"],
            }
        ],
    }


def test_flag_on_routes_run_slot_recommendation_through_grounded(monkeypatch):
    def fake_default_judge(config, reasoner=None):
        return GroundedJudge(judgment_fn=_fake_fn, fallback_reasoner=DeterministicReasoner())

    monkeypatch.setattr(judgment, "default_judge", fake_default_judge)

    result = pipeline.run_slot_recommendation(
        _bayou(), config=Config(use_grounded_judgment=True), geocoder=MockGeocoder()
    )
    # The grounded rationale reached the output -> the flag took effect.
    assert result.recommendation.reasoning == _MARKER
    assert result.recommendation.decision is Decision.RECOMMENDED


def test_flag_off_never_consults_default_judge(monkeypatch):
    calls = {"n": 0}

    def spy(config, reasoner=None):  # pragma: no cover - must not run
        calls["n"] += 1
        raise AssertionError("default_judge must not be called when the flag is off")

    monkeypatch.setattr(judgment, "default_judge", spy)

    result = pipeline.run_slot_recommendation(
        _bayou(),
        config=Config(use_grounded_judgment=False),
        geocoder=MockGeocoder(),
        reasoner=DeterministicReasoner(),
    )
    assert calls["n"] == 0  # weighted path only
    assert result.recommendation.decision is Decision.RECOMMENDED


def test_explicit_judge_overrides_the_flag(monkeypatch):
    # An explicitly-passed judge wins even if the flag is off; default_judge is
    # never consulted.
    def boom(config, reasoner=None):  # pragma: no cover
        raise AssertionError("default_judge must not be called when judge= is explicit")

    monkeypatch.setattr(judgment, "default_judge", boom)

    explicit = GroundedJudge(judgment_fn=_fake_fn, fallback_reasoner=DeterministicReasoner())
    result = pipeline.run_slot_recommendation(
        _bayou(),
        config=Config(use_grounded_judgment=False),
        geocoder=MockGeocoder(),
        judge=explicit,
    )
    assert result.recommendation.reasoning == _MARKER
