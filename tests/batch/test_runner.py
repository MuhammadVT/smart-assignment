"""
Tests for the deterministic per-prospect batch engine (batch/runner.run_one),
fully offline.

``run_one`` is the deterministic floor the agent batch runner falls back to. Uses
MockGeocoder, the mock routes source, and a deterministic config (grounded
reasoning + triage off) so every outcome is reproducible with no LLM call. The
grounded triage brief is covered separately in tests/triage/test_compose_brief.py.
"""

from __future__ import annotations

from dataclasses import replace

from smart_assignment.batch.runner import run_one
from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
)
from smart_assignment.batch.source import MockProspectSource, Prospect
from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.geo import AddressNotFoundError
from smart_assignment.shared.models import CustomerProfile


def _config():
    """Deterministic: no grounded sampling, no LLM triage call."""
    return replace(
        DEFAULT_CONFIG,
        use_grounded_route_slot_escalation=False,
        use_grounded_route_slot_pick=False,
        use_escalation_triage=False,
    )


def _run(prospects):
    """Drive run_one over a list of prospects (the deterministic floor), the way
    the agent runner falls back to it -- returning the per-id records."""
    config = _config()
    geocoder = MockGeocoder()
    routes = fetch_candidate_routes()
    return {p.prospect_id: run_one(p, config, geocoder, routes, "t0") for p in prospects}


def test_runs_every_sample_and_classifies_outcomes():
    by_id = _run(list(MockProspectSource.from_samples().prospects()))

    assert len(by_id) == 4
    # SAMPLE_CUSTOMERS order: Bayou (recommend), Galleria (escalate),
    # Katy (escalate), Woodlands (recommend).
    assert by_id["MOCK-001"].outcome == OUTCOME_RECOMMEND
    assert by_id["MOCK-002"].outcome == OUTCOME_ESCALATE
    assert by_id["MOCK-003"].outcome == OUTCOME_ESCALATE
    assert by_id["MOCK-004"].outcome == OUTCOME_RECOMMEND


def test_recommend_records_carry_the_customer_view_payload():
    by_id = _run(list(MockProspectSource.from_samples().prospects()))
    rec = by_id["MOCK-001"]
    assert rec.payload is not None
    assert rec.payload.get("frontendHtml")  # Customer View renders this unchanged
    assert rec.error is None


def test_escalate_records_carry_a_brief_and_reason():
    by_id = _run(list(MockProspectSource.from_samples().prospects()))
    esc = by_id["MOCK-002"]
    assert esc.payload is not None
    assert esc.review_reason  # why it escalated
    assert esc.triage_brief and "SITUATION" in esc.triage_brief  # deterministic floor brief


def test_invalid_intake_becomes_needs_attention():
    bad = Prospect(
        prospect_id="BAD-1",
        profile=CustomerProfile(
            name="Zero Order", address="1200 McKinney St, Houston, TX 77010",
            order_quantity_cases=0, preferred_slot=None,
        ),
    )
    good = list(MockProspectSource.from_samples().prospects())[0]  # a clean recommend
    by_id = _run([bad, good])

    assert by_id["BAD-1"].outcome == OUTCOME_NEEDS_ATTENTION
    assert by_id["BAD-1"].error and by_id["BAD-1"].payload is None
    # An independent per-prospect call: the bad record never affects the good one.
    assert by_id[good.prospect_id].outcome == OUTCOME_RECOMMEND


def test_unresolvable_address_becomes_needs_attention():
    """The 'trust Salesforce as-is' policy: a geocode miss is surfaced for a human,
    never auto-resolved. MockGeocoder never misses, so a raising stub stands in for
    a real geocoder rejecting a bad Salesforce address."""

    class _RaisingGeocoder:
        def geocode(self, address):
            raise AddressNotFoundError(address, "no match")

    prospect = Prospect(
        prospect_id="SF-BAD-ADDR",
        profile=CustomerProfile(
            name="Bad Address", address="nowhere at all",
            order_quantity_cases=100, preferred_slot=None,
        ),
    )
    record = run_one(prospect, _config(), _RaisingGeocoder(), fetch_candidate_routes(), "t0")

    assert record.outcome == OUTCOME_NEEDS_ATTENTION
    assert "address not found" in record.error
    assert record.payload is None
