"""
`cached_decision_for` -- reusing the decision the agent already made, but ONLY
when it provably belongs to the prospect currently in session state.

Every "not sure" case must return None, because None simply means "decide
again", which is always correct. The dangerous outcome is the opposite: reusing
a decision computed for a different prospect.
"""

from __future__ import annotations

from smart_assignment.shared.models import Decision, SlotRecommendation
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_DECISION_KEY,
    cached_decision_for,
)

_PROFILE = {
    "name": "Test Prospect",
    "address": "1200 McKinney St, Houston, TX 77010",
    "order_quantity_cases": 90,
    "customer_number": None,
    "preferred_day": "TUE",
    "preferred_window_start": "07:00",
    "preferred_window_end": "10:00",
}


def _recommendation() -> SlotRecommendation:
    return SlotRecommendation(
        customer_name="Test Prospect",
        decision=Decision.RECOMMENDED,
        total_score=0.79,
        reasoning="RTE-4100 is the strongest route-slot overall.",
        recommended_route_id="RTE-4100",
    )


def _state(profile=None, recommendation=None) -> dict:
    return {
        _STATE_LAST_DECISION_KEY: {
            "profile": profile if profile is not None else dict(_PROFILE),
            "recommendation": (recommendation or _recommendation()).to_state_dict(),
        }
    }


def test_returns_the_decision_for_a_matching_profile():
    got = cached_decision_for(_state(), dict(_PROFILE))
    assert got == _recommendation()


def test_returns_none_when_there_is_no_snapshot():
    assert cached_decision_for({}, dict(_PROFILE)) is None


def test_returns_none_when_the_prospect_was_revised():
    # The order grew mid-conversation: the cached decision was computed for the
    # OLD quantity and must not be reused for the new one.
    revised = dict(_PROFILE, order_quantity_cases=400)
    assert cached_decision_for(_state(), revised) is None


def test_returns_none_when_the_preference_changed():
    revised = dict(_PROFILE, preferred_day="WED")
    assert cached_decision_for(_state(), revised) is None


def test_returns_none_when_the_address_changed():
    revised = dict(_PROFILE, address="5085 Westheimer Rd, Houston, TX 77056")
    assert cached_decision_for(_state(), revised) is None


def test_returns_none_on_a_corrupt_snapshot(caplog):
    state = _state()
    state[_STATE_LAST_DECISION_KEY]["recommendation"]["decision"] = "NOT_A_DECISION"
    assert cached_decision_for(state, dict(_PROFILE)) is None


def test_returns_none_on_a_malformed_snapshot():
    assert cached_decision_for({_STATE_LAST_DECISION_KEY: "nonsense"}, dict(_PROFILE)) is None
    assert cached_decision_for({_STATE_LAST_DECISION_KEY: {}}, dict(_PROFILE)) is None
