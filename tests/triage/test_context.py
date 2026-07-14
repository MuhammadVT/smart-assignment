"""
Offline tests for the escalation-triage data tool (triage/context.py).

Like the conversational-tool tests, these call the plain-Python function with a
fake tool context and the deterministic MockGeocoder -- no LLM, no ADK runtime.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.tools import slot_recommendation as tools_module
from smart_assignment.tools.slot_recommendation import intake_customer, recommend_or_escalate
from smart_assignment.triage.context import (
    _STATE_TRIAGE_GROUNDING_KEY,
    check_brief_grounding,
    get_escalation_context,
)


@pytest.fixture(autouse=True)
def _use_mock_geocoder():
    with patch.object(tools_module, "_GEOCODER", MockGeocoder()):
        yield


class _FakeToolContext:
    def __init__(self):
        self.state = {}


def _escalated_ctx():
    """A context whose last recommendation escalated (Galleria's 400-case order)."""
    ctx = _FakeToolContext()
    intake_customer(
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        tool_context=ctx,
    )
    rec = recommend_or_escalate(tool_context=ctx)
    assert rec["requires_human_review"] is True  # precondition
    return ctx


def test_returns_grounded_facts_for_an_escalation():
    out = get_escalation_context(_escalated_ctx())

    assert out["ok"] is True
    assert out["decision"] == "ESCALATED_LOW_SCORE"
    assert out["review_reason"]  # why it escalated
    assert out["proposed_route_id"] == "RTE-4200"  # a slot is still proposed
    # Both a feasible (the proposed) route and the infeasible ones are exposed,
    # each with raw per-route facts to ground the brief.
    assert out["feasible_candidates"] and out["infeasible_candidates"]
    facts = out["feasible_candidates"][0]["facts"]
    for key in ("utilization_after", "remaining_capacity_after", "distance_miles"):
        assert key in facts
    # Infeasible routes carry their failure reasons.
    assert out["infeasible_candidates"][0]["failed_constraints"]


def test_auto_approved_recommendation_has_nothing_to_triage():
    # Bayou City Bistro is a clean auto-assign -> no human review -> no triage.
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        tool_context=ctx,
    )
    rec = recommend_or_escalate(tool_context=ctx)
    assert rec["requires_human_review"] is False

    out = get_escalation_context(ctx)
    assert out["ok"] is False
    assert "nothing to triage" in out["error"].lower()


def test_no_profile_yet():
    out = get_escalation_context(_FakeToolContext())
    assert out["ok"] is False
    assert "profile" in out["error"].lower()


def test_no_recommendation_yet():
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        tool_context=ctx,
    )
    out = get_escalation_context(ctx)
    assert out["ok"] is False
    assert "recommend_or_escalate" in out["error"]


def test_recommend_or_escalate_now_exposes_alternative_takes():
    # The tool result must carry alternative_takes (empty on the weighted path)
    # so triage can surface split opinions.
    ctx = _escalated_ctx()
    assert "alternative_takes" in ctx.state["sa_last_recommendation"]


# --- brief groundedness self-check tool --------------------------------------


def test_get_escalation_context_stashes_grounding_for_the_verifier():
    ctx = _escalated_ctx()
    get_escalation_context(ctx)
    grounding = ctx.state.get(_STATE_TRIAGE_GROUNDING_KEY)
    assert grounding and grounding["numbers"] and grounding["route_ids"]


def test_check_brief_grounding_passes_a_faithful_brief():
    ctx = _escalated_ctx()
    context = get_escalation_context(ctx)
    facts = context["feasible_candidates"][0]["facts"]
    pct = round(float(facts["utilization_after"]) * 100)
    rid = context["feasible_candidates"][0]["route_id"]
    brief = f"Route {rid} sits at about {pct}% utilization -- a thin margin, hence the review."

    result = check_brief_grounding(ctx, brief)
    assert result["ok"] is True


def test_check_brief_grounding_flags_a_hallucinated_figure():
    ctx = _escalated_ctx()
    get_escalation_context(ctx)

    result = check_brief_grounding(ctx, "This route has 9999 cases of headroom, plenty of room.")
    assert result["ok"] is False
    assert "9999" in result["ungrounded_numbers"]
    assert "caution" in result["message"].lower()


def test_check_brief_grounding_requires_context_first():
    ctx = _FakeToolContext()
    result = check_brief_grounding(ctx, "some brief")
    assert result["ok"] is False
    assert "get_escalation_context" in result["error"]
