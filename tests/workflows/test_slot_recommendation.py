"""
Tests for the slot_recommendation workflow: the end-to-end pipeline decisions
and the ADK graph's conditional routers. All deterministic — the LLM reasoning
layer is bypassed via the DeterministicReasoner so no API key/network is used.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, Decision
from smart_assignment.workflows.slot_recommendation.nodes import (
    confidence_gate,
    route_on_feasibility,
)
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation
from smart_assignment.workflows.slot_recommendation.reasoning import (
    DeterministicReasoner,
    compute_confidence,
)

_DETERMINISTIC = DeterministicReasoner()


def _run(customer, config=None):
    return run_slot_recommendation(customer, config=config or Config(), reasoner=_DETERMINISTIC)


def test_clear_case_is_recommended():
    customer = CustomerProfile(
        customer_id="C1",
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_window=(time(7, 0), time(10, 0)),
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.RECOMMENDED
    assert rec.recommended_route_id == "RTE-4100"
    assert rec.requires_human_review is False


def test_unserviceable_customer_escalates_no_feasible_slot():
    customer = CustomerProfile(
        customer_id="C3",
        name="Katy Prairie Steakhouse",
        address="24600 Katy Fwy, Katy, TX 77494",
        order_quantity_cases=260,
        preferred_window=(time(6, 0), time(8, 0)),
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    assert rec.requires_human_review is True


def test_near_tie_escalates_low_confidence():
    customer = CustomerProfile(
        customer_id="C2",
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=140,
        preferred_window=None,
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.ESCALATED_LOW_CONFIDENCE
    assert rec.recommended_route_id is not None  # a slot IS proposed for the human
    assert rec.requires_human_review is True


# --- confidence math --------------------------------------------------------


def test_confidence_low_for_near_tie():
    config = Config()

    class _Cand:
        def __init__(self, score):
            self.total_score = score

    assert compute_confidence([_Cand(0.55), _Cand(0.54)], config) < config.confidence_threshold
    assert compute_confidence([_Cand(0.9), _Cand(0.5)], config) >= config.confidence_threshold


# --- ADK conditional routers (no live model needed) -------------------------


class _Feasible:
    feasible = True


class _Infeasible:
    feasible = False


def test_route_on_feasibility_no_options():
    event = route_on_feasibility([_Infeasible(), _Infeasible()])
    assert event.actions.route == ["NO_OPTIONS"]


def test_route_on_feasibility_has_options():
    event = route_on_feasibility([_Infeasible(), _Feasible()])
    assert event.actions.route == ["HAS_OPTIONS"]


class _FakeContext:
    def __init__(self, state):
        self.state = state


def _rec(decision):
    from smart_assignment.shared.models import SlotRecommendation

    return SlotRecommendation(
        customer_id="C",
        customer_name="n",
        decision=decision,
        confidence=0.9,
        reasoning="",
    )


def test_confidence_gate_high_confidence():
    event = confidence_gate(_rec(Decision.RECOMMENDED), _FakeContext({}))
    assert event.actions.route == ["HIGH_CONFIDENCE"]


def test_confidence_gate_low_confidence():
    event = confidence_gate(_rec(Decision.ESCALATED_LOW_CONFIDENCE), _FakeContext({}))
    assert event.actions.route == ["LOW_CONFIDENCE"]
