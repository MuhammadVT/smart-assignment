"""
Tests for the consolidated orchestration mode
(``Config.use_consolidated_pipeline_tool``).

The mode swaps the agent's four deterministic step tools for one that runs the
whole chain. Two things must hold for that to be a safe trade:

* **Same answer.** ``assign_delivery_slot`` must produce exactly what the stepwise
  ``intake_customer`` -> ``recommend_or_escalate`` sequence produces. If the two
  shapes could disagree, the flag would be a behavior change rather than a cost
  optimization.
* **Flag-off changes nothing.** With the flag off, the registered tools and the
  built instruction must be byte-identical to before.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.prompts import build_instruction
from smart_assignment.shared.geo import AddressNotFoundError
from smart_assignment.tools import slot_recommendation as tools_module
from smart_assignment.tools.slot_recommendation import (
    assign_delivery_slot,
    intake_customer,
    recommend_or_escalate,
)

_ADDRESS = "1200 McKinney St, Houston, TX 77010"


@pytest.fixture(autouse=True)
def _use_mock_geocoder():
    with patch.object(tools_module, "_GEOCODER", MockGeocoder()):
        yield


class _FakeToolContext:
    def __init__(self):
        self.state = {}


# --- equivalence with the stepwise sequence --------------------------------


def test_matches_the_stepwise_sequence_exactly():
    stepwise_ctx = _FakeToolContext()
    intake_customer(
        tool_context=stepwise_ctx,
        address=_ADDRESS,
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
    )
    stepwise = recommend_or_escalate(tool_context=stepwise_ctx)

    consolidated_ctx = _FakeToolContext()
    consolidated = assign_delivery_slot(
        tool_context=consolidated_ctx,
        address=_ADDRESS,
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
    )

    # `workflow_steps` is the only addition; every decision field must match.
    assert {k: v for k, v in consolidated.items() if k != "workflow_steps"} == stepwise


def test_session_state_is_identical_to_stepwise_mode():
    """The escalation-triage sub-agent reads the stashed recommendation, so what
    lands in state must not gain the extra narration facts."""
    stepwise_ctx = _FakeToolContext()
    intake_customer(tool_context=stepwise_ctx, address=_ADDRESS, order_quantity_cases=90)
    recommend_or_escalate(tool_context=stepwise_ctx)

    consolidated_ctx = _FakeToolContext()
    assign_delivery_slot(tool_context=consolidated_ctx, address=_ADDRESS, order_quantity_cases=90)

    assert consolidated_ctx.state["sa_profile"] == stepwise_ctx.state["sa_profile"]
    assert (
        consolidated_ctx.state["sa_last_recommendation"]
        == stepwise_ctx.state["sa_last_recommendation"]
    )
    assert "workflow_steps" not in consolidated_ctx.state["sa_last_recommendation"]


# --- the narration facts the middle tools used to supply --------------------


def test_carries_the_geo_and_feasibility_facts_for_narration():
    ctx = _FakeToolContext()
    result = assign_delivery_slot(tool_context=ctx, address=_ADDRESS, order_quantity_cases=90)
    steps = result["workflow_steps"]
    assert steps["geocoded_location"]["latitude"] is not None
    assert steps["candidate_routes"]
    for route in steps["candidate_routes"]:
        assert route["route_id"] and route["name"] and route["day"]
        assert isinstance(route["feasible"], bool)
        # An infeasible route says WHY, so the agent can narrate the rule-out.
        if not route["feasible"]:
            assert route["failed_constraints"]


# --- intake still pauses, revisions still merge ----------------------------


def test_incomplete_intake_returns_the_same_clarification_without_geocoding():
    ctx = _FakeToolContext()
    result = assign_delivery_slot(tool_context=ctx, address=_ADDRESS)
    assert result["ok"] is False
    assert "order quantity" in result["error"]
    # Intake validates before any geocode, so the partial profile is still saved
    # and the user is never asked to repeat what they already gave.
    assert ctx.state["sa_profile"]["address"] == _ADDRESS


def test_revision_merges_only_the_changed_field():
    ctx = _FakeToolContext()
    first = assign_delivery_slot(tool_context=ctx, address=_ADDRESS, order_quantity_cases=90)
    assert first["ok"] is True

    revised = assign_delivery_slot(tool_context=ctx, order_quantity_cases=20)
    assert revised["ok"] is True
    assert ctx.state["sa_profile"]["address"] == _ADDRESS
    assert ctx.state["sa_profile"]["order_quantity_cases"] == 20


def test_geocoding_failure_is_relayed_so_address_resolution_can_run():
    class _NotFound:
        def geocode(self, address):
            raise AddressNotFoundError(address, "no match")

    ctx = _FakeToolContext()
    with patch.object(tools_module, "_GEOCODER", _NotFound()):
        result = assign_delivery_slot(
            tool_context=ctx, address="nowhere, XX", order_quantity_cases=10
        )
    assert result["ok"] is False
    # The message the agent needs in order to reach for resolve_address.
    assert "couldn't find a location" in result["error"]


# --- flag-off parity + wiring ----------------------------------------------


def test_flag_off_instruction_is_unchanged():
    """The stepwise instruction must be reproduced exactly when the flag is off."""
    from smart_assignment.prompts import (
        ADDRESS_RESOLUTION_GUIDANCE,
        ESCALATION_TRIAGE_GUIDANCE,
        INSTRUCTION,
    )

    assert build_instruction() == INSTRUCTION
    assert (
        build_instruction(include_triage=True, include_address_resolution=True)
        == INSTRUCTION + ADDRESS_RESOLUTION_GUIDANCE + ESCALATION_TRIAGE_GUIDANCE
    )


def test_consolidated_instruction_names_only_tools_that_exist():
    instruction = build_instruction(
        consolidated=True, include_triage=True, include_address_resolution=True
    )
    assert "assign_delivery_slot" in instruction
    # The step tools are not registered in this mode, so the prompt must never
    # tell the model to call them.
    for absent in ("find_candidate_routes", "evaluate_and_score_routes", "intake_customer"):
        assert absent not in instruction
    # The tools that DO remain must still be named.
    assert "resolve_address" in instruction
    assert "escalation_triage" in instruction
    assert "request_input" in instruction


def test_consolidated_instruction_is_shorter_than_the_stepwise_one():
    """The 'don't stop between steps' scaffolding is unnecessary when there are no
    intermediate steps -- if this ever inverts, the mode has lost its point."""
    assert len(build_instruction(consolidated=True)) < len(build_instruction())
