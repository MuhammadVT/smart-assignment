"""
Evidence-packet construction -- the raw per-route facts a triage brief is
grounded on (`triage/evidence.py`).

Everything here is offline: real `CandidateEvaluation`s are produced by running
the deterministic front of the pipeline (intake -> geo_lookup -> evaluate) over
the mock customers. No LLM, no network, no credentials.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import evaluate_candidates, geo_lookup, intake
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, PreferredSlot
from smart_assignment.triage.evidence import NUMERIC_FACT_KEYS, build_evidence_packet

# The canonical mock scenarios (mirrors tests/test_pipeline.py).
CLEAR_RECOMMEND = CustomerProfile(
    name="Bayou City Bistro",
    address="1200 McKinney St, Houston, TX 77010",
    order_quantity_cases=90,
    preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
)
LOW_SCORE = CustomerProfile(
    name="Galleria Grill & Catering",
    address="5085 Westheimer Rd, Houston, TX 77056",
    order_quantity_cases=400,
    preferred_slot=None,
)
NO_FEASIBLE = CustomerProfile(
    name="Katy Prairie Steakhouse",
    address="5000 Katy Mills Cir, Katy, TX 77494",
    order_quantity_cases=260,
    preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(6, 0), time(8, 0))),
)


def evaluations_for(customer: CustomerProfile, config: Config | None = None):
    """Run the deterministic front of the pipeline; return (customer, evals)."""
    config = config or Config()
    customer = intake(customer)
    candidates = geo_lookup(customer, fetch_candidate_routes(), MockGeocoder(), config)
    return customer, evaluate_candidates(customer, candidates, config)


def feasible_ids(evaluations) -> list[str]:
    return [e.route.route_id for e in evaluations if e.feasible]


def test_packet_splits_feasible_and_infeasible_and_exposes_raw_facts():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)

    assert packet.feasible_candidates, "Galleria should have at least one feasible route"
    assert packet.infeasible_candidates, "Galleria should have infeasible routes too"

    for cand in packet.feasible_candidates:
        facts = cand["facts"]
        for key in NUMERIC_FACT_KEYS:
            assert key in facts
        # The weighted score is present but explicitly labelled reference-only.
        assert facts["reference_weighted_score"] is not None
        assert "context only" in cand["reference_only_note"].lower()

    for cand in packet.infeasible_candidates:
        assert cand["failed_constraints"], "an infeasible candidate must say why it failed"


def test_packet_facts_match_the_evaluation_numbers():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)
    winner_id = feasible_ids(evals)[0]
    ev = packet.evaluation_for(winner_id)
    facts = packet.candidate_dict(winner_id)["facts"]
    assert facts["utilization_after"] == round(ev.utilization_after, 4)
    assert facts["order_quantity_cases"] == customer.order_quantity_cases


def test_no_feasible_packet_has_empty_feasible_list():
    config = Config()
    customer, evals = evaluations_for(NO_FEASIBLE, config)
    packet = build_evidence_packet(customer, evals, config)
    assert packet.feasible_candidates == []
    assert packet.feasible_route_ids == []


def test_clear_recommend_customer_has_a_feasible_winner():
    # Sanity: the "clear" scenario really does produce a feasible route to pick.
    config = Config()
    _, evals = evaluations_for(CLEAR_RECOMMEND, config)
    assert feasible_ids(evals), "Bayou City Bistro should have a feasible route"
