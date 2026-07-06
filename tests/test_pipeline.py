"""
Tests for the slot_recommendation pipeline's end-to-end decisions and the
total-score gating math. All deterministic -- the LLM reasoning layer is
bypassed via the DeterministicReasoner so no API key/network is used.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner, compute_total_score
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, Decision, PreferredSlot

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


def test_prospect_with_no_customer_number_runs_by_address():
    # The default path: a new prospect from Salesforce with an address but no
    # Sysco customer number yet. customer_number is left as the placeholder
    # default (None); intake must not require it.
    customer = CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
    )
    result = _run(customer)
    assert result.customer.customer_number is None
    assert result.customer.lookup_key == customer.address
    assert result.recommendation.decision == Decision.RECOMMENDED
    assert result.recommendation.customer_number is None
    assert result.recommendation.customer_address == customer.address


def test_missing_address_is_rejected():
    import pytest

    customer = CustomerProfile(
        name="No Address Diner",
        address="",
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
