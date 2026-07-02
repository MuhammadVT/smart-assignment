"""
Tests for the slot_recommendation workflow: the end-to-end pipeline decisions
and the ADK graph's conditional routers. All deterministic — the LLM reasoning
layer is bypassed via the DeterministicReasoner so no API key/network is used.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, Decision, PreferredSlot
from smart_assignment.workflows.slot_recommendation.nodes import (
    route_on_feasibility,
    total_score_gate,
)
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation
from smart_assignment.workflows.slot_recommendation.reasoning import (
    DeterministicReasoner,
    compute_total_score,
)

_DETERMINISTIC = DeterministicReasoner()


def _run(customer, config=None):
    return run_slot_recommendation(customer, config=config or Config(), reasoner=_DETERMINISTIC)


def test_clear_case_is_recommended():
    customer = CustomerProfile(
        customer_number="067-100001",
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.RECOMMENDED
    assert rec.recommended_route_id == "RTE-4100"
    assert rec.requires_human_review is False


def test_unserviceable_customer_escalates_no_feasible_slot():
    customer = CustomerProfile(
        customer_number="067-100003",
        name="Katy Prairie Steakhouse",
        address="24600 Katy Fwy, Katy, TX 77494",
        order_quantity_cases=260,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(6, 0), time(8, 0))),
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    assert rec.requires_human_review is True


def test_malformed_customer_number_is_rejected():
    import pytest

    customer = CustomerProfile(
        customer_number="CUST-NEW-9001",  # not the NNN-NNNNNN format
        name="Bad Number Diner",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
    )
    with pytest.raises(ValueError):
        _run(customer)


def test_large_order_escalates_low_total_score():
    # Large enough that only one nearby route can still take it, and even
    # that route ends up quite full -- its own total_score (not a tie with
    # some other option) is what trips the escalation.
    customer = CustomerProfile(
        customer_number="067-100002",
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        preferred_slot=None,
    )
    rec = _run(customer).recommendation
    assert rec.decision == Decision.ESCALATED_LOW_SCORE
    assert rec.total_score < Config().total_score_threshold
    assert rec.recommended_route_id is not None  # a slot IS proposed for the human


# --- total-score gating math -------------------------------------------------


def test_total_score_is_the_winners_own_score_untouched_by_the_runner_up():
    class _Cand:
        def __init__(self, score):
            self.total_score = score

    # A tie between two GOOD options is not penalized -- the winner's own
    # score stands on its own, regardless of how close the runner-up scored.
    assert compute_total_score([_Cand(0.75), _Cand(0.74)]) == 0.75
    # A tie between two MEDIOCRE options stays mediocre -- correctly still low.
    assert compute_total_score([_Cand(0.55), _Cand(0.54)]) == 0.55
    # No feasible candidates at all.
    assert compute_total_score([]) == 0.0


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
        customer_number="067-100000",
        customer_name="n",
        decision=decision,
        total_score=0.9,
        reasoning="",
    )


def test_total_score_gate_high_score():
    event = total_score_gate(_rec(Decision.RECOMMENDED), _FakeContext({}))
    assert event.actions.route == ["HIGH_SCORE"]


def test_total_score_gate_low_score():
    event = total_score_gate(_rec(Decision.ESCALATED_LOW_SCORE), _FakeContext({}))
    assert event.actions.route == ["LOW_SCORE"]
