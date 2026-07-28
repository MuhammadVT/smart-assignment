"""
Tests for the *pure* escalation-context builders extracted from the ADK tool
(triage/context.py): ``build_escalation_context`` and
``escalation_context_from_recommendation``.

The first test LOCKS the refactor: the conversational tool
(``get_escalation_context``) must delegate to ``build_escalation_context`` and
produce a byte-identical dict, so chat behavior is provably unchanged. The others
exercise the batch-facing entry point over the real deterministic pipeline. All
offline: MockGeocoder, no LLM, no ADK runtime.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.pipeline import evaluate_candidates, run_slot_recommendation
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import CustomerProfile
from smart_assignment.tools import slot_recommendation as tools_module
from smart_assignment.tools.slot_recommendation import intake_customer
from smart_assignment.triage.context import (
    build_escalation_context,
    escalation_context_from_recommendation,
    get_escalation_context,
)

_GALLERIA_ADDR = "5085 Westheimer Rd, Houston, TX 77056"


@pytest.fixture(autouse=True)
def _use_mock_geocoder():
    with patch.object(tools_module, "_GEOCODER", MockGeocoder()):
        yield


class _FakeToolContext:
    def __init__(self):
        self.state = {}


def _deterministic_config():
    """Grounded reasoning off, so step 5 is fully reproducible (no sampling)."""
    return replace(
        DEFAULT_CONFIG,
        use_grounded_route_slot_escalation=False,
        use_grounded_route_slot_pick=False,
    )


def test_builder_matches_the_tool_context_exactly():
    """The refactor lock: with the SAME profile + last-recommendation facts, the
    tool's ``get_escalation_context`` and a direct ``build_escalation_context`` call
    must yield an identical dict. Uses a synthetic ``last`` so no step-5 sampling can
    make the two paths diverge."""
    ctx = _FakeToolContext()
    intake_customer(address=_GALLERIA_ADDR, order_quantity_cases=400, tool_context=ctx)
    profile = ctx.state[tools_module._STATE_PROFILE_KEY]

    last = {
        "requires_human_review": True,
        "decision": "ESCALATED_LOW_SCORE",
        "review_reason": "the proposed route's margin is thin",
        "recommended_route_id": "RTE-4200",
        "total_score": 0.5,
        "alternative_takes": [],
    }
    ctx.state[tools_module._STATE_LAST_RECOMMENDATION_KEY] = last
    context_tool = get_escalation_context(ctx)

    # Direct builder path: same customer + evaluations, same decision facts.
    customer = tools_module._profile_from_state_dict(profile)
    candidates = tools_module._find_candidates(customer)
    evaluations = evaluate_candidates(customer, candidates, DEFAULT_CONFIG)
    context_direct = build_escalation_context(
        customer,
        evaluations,
        {
            "decision": last["decision"],
            "review_reason": last["review_reason"],
            "proposed_route_id": last["recommended_route_id"],
            "total_score": last["total_score"],
            "alternative_takes": last["alternative_takes"],
        },
        DEFAULT_CONFIG,
    )

    assert context_direct == context_tool


def test_from_recommendation_maps_the_decision_fields():
    cfg = _deterministic_config()
    profile = CustomerProfile(
        name="Galleria Grill & Catering",
        address=_GALLERIA_ADDR,
        order_quantity_cases=400,
        preferred_slot=None,
    )
    result = run_slot_recommendation(profile, config=cfg, geocoder=MockGeocoder())
    rec = result.recommendation
    assert rec.requires_human_review  # precondition: this prospect escalates

    context = escalation_context_from_recommendation(
        result.customer, result.candidates_considered, rec, cfg
    )

    assert context["ok"] is True
    assert context["decision"] == rec.decision.value
    assert context["proposed_route_id"] == rec.recommended_route_id
    assert context["total_score"] == rec.total_score
    assert context["alternative_takes"] == rec.alternative_takes
    # The raw candidate facts a brief cites are present.
    assert context["feasible_candidates"] or context["infeasible_candidates"]
    assert context["customer"]["order_quantity_cases"] == 400
