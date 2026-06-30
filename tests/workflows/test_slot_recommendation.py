"""
Tests for slot_recommendation's deterministic routing logic (the
conditional router functions), independent of the LLM node. These
verify the graph routes correctly without needing a live model call.
"""

from __future__ import annotations

from smart_assignment.workflows.slot_recommendation.nodes import (
    confidence_gate,
    route_on_feasibility,
)
from smart_assignment.workflows.slot_recommendation.prompts import SlotRecommendationOutput


def test_route_on_feasibility_no_options():
    event = route_on_feasibility([])
    assert event.actions.route == ["NO_OPTIONS"]


def test_route_on_feasibility_has_options(sample_customer, open_route):
    from smart_assignment.shared.tools import filter_feasible_slots

    feasible = filter_feasible_slots(sample_customer, [open_route])
    event = route_on_feasibility(feasible)
    assert event.actions.route == ["HAS_OPTIONS"]


class _FakeContext:
    """Minimal stand-in for ADK's Context to avoid spinning up a full session."""

    def __init__(self, state):
        self.state = state


def test_confidence_gate_low_confidence(sample_customer):
    rec = SlotRecommendationOutput(
        recommended_route_id="RTE-1",
        recommended_day="TUE",
        recommended_window_start="08:00",
        recommended_window_end="09:00",
        confidence=0.4,
        reasoning="ambiguous tradeoff",
        rejected_alternatives=[],
    )
    event = confidence_gate(rec, _FakeContext(state={"customer": sample_customer}))
    assert event.actions.route == ["LOW_CONFIDENCE"]


def test_confidence_gate_high_confidence(sample_customer):
    rec = SlotRecommendationOutput(
        recommended_route_id="RTE-1",
        recommended_day="TUE",
        recommended_window_start="08:00",
        recommended_window_end="09:00",
        confidence=0.95,
        reasoning="clear best option",
        rejected_alternatives=[],
    )
    event = confidence_gate(rec, _FakeContext(state={"customer": sample_customer}))
    assert event.actions.route == ["HIGH_CONFIDENCE"]
