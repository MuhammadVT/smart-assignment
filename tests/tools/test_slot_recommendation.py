"""
Unit tests for the conversational tool wrappers
(tools/slot_recommendation.py). These call the plain-Python wrapper
functions directly with a fake tool context -- no LLM, no ADK runtime
needed.
"""

from __future__ import annotations

from smart_assignment.tools.slot_recommendation import (
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
)


class _FakeToolContext:
    def __init__(self):
        self.state = {}


def test_intake_customer_requires_address_and_cases():
    ctx = _FakeToolContext()
    result = intake_customer(tool_context=ctx)
    assert result["ok"] is False
    assert "address" in result["error"]


def test_intake_customer_succeeds_and_persists_profile():
    ctx = _FakeToolContext()
    result = intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        tool_context=ctx,
    )
    assert result["ok"] is True
    assert ctx.state["sa_profile"]["address"] == "1200 McKinney St, Houston, TX 77010"
    assert ctx.state["sa_profile"]["customer_number"] is None


def test_intake_customer_merges_partial_updates_without_losing_prior_fields():
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        tool_context=ctx,
    )
    # A revision only supplies the changed fields -- address/cases must survive.
    result = intake_customer(
        preferred_day="tue",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=ctx,
    )
    assert result["ok"] is True
    profile = result["profile"]
    assert profile["address"] == "1200 McKinney St, Houston, TX 77010"
    assert profile["order_quantity_cases"] == 90
    assert profile["preferred_day"] == "TUE"  # normalized to upper case


def test_intake_customer_rejects_half_specified_slot():
    ctx = _FakeToolContext()
    result = intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_day="TUE",  # window omitted
        tool_context=ctx,
    )
    assert result["ok"] is False


def test_intake_customer_rejects_malformed_customer_number():
    ctx = _FakeToolContext()
    result = intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        customer_number="BAD-NUMBER",
        tool_context=ctx,
    )
    assert result["ok"] is False


def test_clear_preferred_slot_removes_a_previously_recorded_one():
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=ctx,
    )
    result = intake_customer(clear_preferred_slot=True, tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["preferred_day"] is None


def test_downstream_tools_require_intake_first():
    ctx = _FakeToolContext()
    assert find_candidate_routes(tool_context=ctx)["ok"] is False
    assert evaluate_and_score_routes(tool_context=ctx)["ok"] is False
    assert recommend_or_escalate(tool_context=ctx)["ok"] is False


def test_full_conversational_flow_matches_batch_pipeline():
    # Bayou City Bistro's known-good scenario (see test_slot_recommendation.py).
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=ctx,
    )

    routes = find_candidate_routes(tool_context=ctx)
    assert routes["ok"] is True
    assert routes["candidate_routes"]  # at least one nearby route found

    scored = evaluate_and_score_routes(tool_context=ctx)
    assert scored["ok"] is True
    feasible = [r for r in scored["routes"] if r["feasible"]]
    assert feasible and feasible[0]["route_id"] == "RTE-4100"

    rec = recommend_or_escalate(tool_context=ctx)
    assert rec["ok"] is True
    assert rec["decision"] == "RECOMMENDED"
    assert rec["recommended_route_id"] == "RTE-4100"
    assert rec["requires_human_review"] is False
    assert ctx.state["sa_last_recommendation"]["decision"] == "RECOMMENDED"


def test_revision_flows_through_to_a_new_recommendation():
    # Galleria's large order escalates on low score; shrinking it back down
    # via a follow-up intake_customer call (no address/slot restated) must
    # flow through to a better outcome without losing anything on file.
    ctx = _FakeToolContext()
    intake_customer(
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        tool_context=ctx,
    )
    first = recommend_or_escalate(tool_context=ctx)
    assert first["decision"] == "ESCALATED_LOW_SCORE"
    assert first["requires_human_review"] is True

    intake_customer(order_quantity_cases=140, tool_context=ctx)
    second = recommend_or_escalate(tool_context=ctx)
    assert second["decision"] == "RECOMMENDED"
    assert second["total_score"] > first["total_score"]
    assert ctx.state["sa_profile"]["address"] == "5085 Westheimer Rd, Houston, TX 77056"
