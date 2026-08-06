"""
Unit tests for the conversational tool wrappers
(tools/slot_recommendation.py). These call the plain-Python wrapper
functions directly with a fake tool context -- no LLM, no ADK runtime
needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.shared.geo import AddressNotFoundError, GeocodingServiceError
from smart_assignment.tools import slot_recommendation as tools_module
from smart_assignment.tools.slot_recommendation import (
    _STATE_PROFILE_KEY,
    assign_prospect,
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
    start_new_prospect,
)


@pytest.fixture(autouse=True)
def _use_mock_geocoder():
    # The tools default to the real CensusGeocoder (network calls). Swap in
    # the deterministic MockGeocoder for this whole file so these tests stay
    # offline -- the geocoding-failure tests below patch it again per-test to
    # something that raises instead.
    with patch.object(tools_module, "_GEOCODER", MockGeocoder()):
        yield


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


def test_intake_customer_persists_fields_from_an_incomplete_first_call():
    # Conversational intake often arrives one field at a time: the address this
    # turn, the order quantity the next. The first call is incomplete and returns
    # an error, but the field it carried must survive so the customer is not asked
    # to repeat it (regression: partial intake used to be discarded on the error).
    ctx = _FakeToolContext()
    first = intake_customer(address="1200 McKinney St, Houston", tool_context=ctx)
    assert first["ok"] is False  # still needs the order quantity
    assert ctx.state["sa_profile"]["address"] == "1200 McKinney St, Houston"

    second = intake_customer(order_quantity_cases=90, tool_context=ctx)
    assert second["ok"] is True  # the earlier address is still on file
    assert second["profile"]["address"] == "1200 McKinney St, Houston"
    assert second["profile"]["order_quantity_cases"] == 90


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


# --- Consolidated one-shot tool: assign_prospect -----------------------------


def test_assign_prospect_matches_the_step_by_step_flow():
    """One assign_prospect call must yield the IDENTICAL decision the four-tool
    sequence produces for the same prospect -- it changes no logic, only round
    trips. Compared field-by-field against intake -> recommend_or_escalate."""
    stepwise_ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=stepwise_ctx,
    )
    stepwise = recommend_or_escalate(tool_context=stepwise_ctx)

    oneshot_ctx = _FakeToolContext()
    oneshot = assign_prospect(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=oneshot_ctx,
    )

    assert oneshot == stepwise
    assert oneshot["decision"] == "RECOMMENDED"
    assert oneshot["recommended_route_id"] == "RTE-4100"


def test_assign_prospect_writes_the_same_state_as_recommend_or_escalate():
    """It must populate BOTH the agent-facing summary and the per-turn decision
    snapshot (bound to the profile), so a rendering surface reuses the decision
    exactly as it does after recommend_or_escalate."""
    ctx = _FakeToolContext()
    assign_prospect(
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        tool_context=ctx,
    )
    assert ctx.state["sa_last_recommendation"]["decision"] == "RECOMMENDED"
    snapshot = ctx.state["sa_last_decision"]
    assert snapshot["profile"] == ctx.state[_STATE_PROFILE_KEY]
    assert snapshot["recommendation"]  # a serialized SlotRecommendation


def test_assign_prospect_decides_from_a_pre_seeded_profile_without_intake_fields():
    """The batch path seeds the profile into session state, then calls
    assign_prospect with NO intake fields -- it must decide from what's on file."""
    ctx = _FakeToolContext()
    ctx.state[_STATE_PROFILE_KEY] = {
        "name": "Bayou City Bistro",
        "address": "1200 McKinney St, Houston, TX 77010",
        "order_quantity_cases": 90,
        "customer_number": None,
        "preferred_day": "TUE",
        "preferred_window_start": "07:00",
        "preferred_window_end": "10:00",
    }
    result = assign_prospect(tool_context=ctx)
    assert result["ok"] is True
    assert result["decision"] == "RECOMMENDED"
    assert result["recommended_route_id"] == "RTE-4100"


def test_assign_prospect_relays_intake_failure_without_deciding():
    """An incomplete intake must come back as intake_customer's error, and no
    decision may be written."""
    ctx = _FakeToolContext()
    result = assign_prospect(address="1200 McKinney St, Houston, TX 77010", tool_context=ctx)
    assert result["ok"] is False
    assert "order quantity" in result["error"]
    assert "sa_last_recommendation" not in ctx.state
    assert "sa_last_decision" not in ctx.state


def test_assign_prospect_escalates_like_the_step_by_step_flow():
    """Galleria's large order escalates; the one-shot tool must reach the same
    escalation with requires_human_review set."""
    ctx = _FakeToolContext()
    result = assign_prospect(
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        tool_context=ctx,
    )
    assert result["decision"] == "ESCALATED_LOW_SCORE"
    assert result["requires_human_review"] is True


def test_assign_prospect_relays_geocoding_failure_without_half_writing_state():
    ctx = _FakeToolContext()
    with patch.object(
        tools_module._GEOCODER,
        "geocode",
        side_effect=GeocodingServiceError("1200 McKinney St, Houston, TX 77010", "boom"),
    ):
        result = assign_prospect(
            address="1200 McKinney St, Houston, TX 77010",
            order_quantity_cases=90,
            tool_context=ctx,
        )
    assert result["ok"] is False
    assert "temporarily unavailable" in result["error"]
    assert "sa_last_recommendation" not in ctx.state


# --- Geocoding failure handling ----------------------------------------------


def test_find_candidate_routes_relays_address_not_found():
    ctx = _FakeToolContext()
    intake_customer(address="not a real place", order_quantity_cases=90, tool_context=ctx)
    with patch.object(
        tools_module._GEOCODER,
        "geocode",
        side_effect=AddressNotFoundError("not a real place", "no match"),
    ):
        result = find_candidate_routes(tool_context=ctx)
    assert result["ok"] is False
    assert "not a real place" in result["error"]


def test_evaluate_and_score_routes_relays_service_error():
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010", order_quantity_cases=90, tool_context=ctx
    )
    with patch.object(
        tools_module._GEOCODER,
        "geocode",
        side_effect=GeocodingServiceError("1200 McKinney St, Houston, TX 77010", "timed out"),
    ):
        result = evaluate_and_score_routes(tool_context=ctx)
    assert result["ok"] is False
    assert "temporarily unavailable" in result["error"]


def test_recommend_or_escalate_relays_geocoding_failure_instead_of_crashing():
    ctx = _FakeToolContext()
    intake_customer(
        address="1200 McKinney St, Houston, TX 77010", order_quantity_cases=90, tool_context=ctx
    )
    with patch.object(
        tools_module._GEOCODER,
        "geocode",
        side_effect=GeocodingServiceError("1200 McKinney St, Houston, TX 77010", "boom"),
    ):
        result = recommend_or_escalate(tool_context=ctx)
    assert result["ok"] is False
    assert "sa_last_recommendation" not in ctx.state  # nothing half-written on failure


# --- New-prospect isolation: the profile belongs to its address ---------------
#
# The leak these pin down (observed live): a browser/adk-web conversation runs
# prospect A to a decision, then starts prospect B in the SAME session. intake's
# merge -- correct for revisions -- carried A's unstated fields (order size,
# preferred slot) into B, so B was decided against a delivery preference its
# customer never expressed. The guard: after a decision, a DIFFERENT address
# starts a fresh prospect and clears every prospect-scoped state key. Before a
# decision, merging is unchanged -- that is what the resolve_address correction
# flow and one-field-at-a-time intake rely on.


_ADDR_A = "1200 McKinney St, Houston, TX 77010"
_ADDR_B = "5000 Katy Mills Cir, Katy, TX 77494"


def _concluded_prospect_a(ctx):
    """Prospect A on file WITH a preferred slot, decided (escalate or recommend)."""
    intake_customer(
        address=_ADDR_A,
        order_quantity_cases=90,
        preferred_day="TUE",
        preferred_window_start="07:00",
        preferred_window_end="10:00",
        tool_context=ctx,
    )
    result = recommend_or_escalate(tool_context=ctx)
    assert result["ok"] is True  # a decision (either way) is now on file
    assert ctx.state["sa_last_recommendation"]


def test_new_address_after_decision_starts_a_fresh_profile():
    """THE captured bug: prospect B states no preference and must not inherit A's."""
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)

    result = intake_customer(
        address=_ADDR_B, order_quantity_cases=260, tool_context=ctx
    )
    assert result["ok"] is True
    profile = result["profile"]
    assert profile["address"] == _ADDR_B
    assert profile["order_quantity_cases"] == 260
    # The heart of the bug: these were 'TUE'/'07:00'/'10:00' before the guard.
    assert profile["preferred_day"] is None
    assert profile["preferred_window_start"] is None
    assert profile["preferred_window_end"] is None


def test_new_address_after_decision_clears_the_decision_and_triage_state():
    """A stale snapshot would let cached_decision_for re-render -- and the triage
    tool ground a specialist brief on -- the PREVIOUS customer's outcome."""
    from smart_assignment.tools.slot_recommendation import _PROSPECT_SCOPED_STATE_KEYS

    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)
    ctx.state["sa_triage_grounding"] = {"figures": [90]}  # as the triage tool would

    intake_customer(address=_ADDR_B, order_quantity_cases=260, tool_context=ctx)

    for key in _PROSPECT_SCOPED_STATE_KEYS:
        assert not ctx.state.get(key), f"{key} survived into the new prospect"


def test_new_address_after_decision_requires_cases_again():
    """The accepted re-ask: a fresh prospect must not inherit A's order size, so
    intake without cases fails with the standard ask instead of proceeding."""
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)

    result = intake_customer(address=_ADDR_B, tool_context=ctx)
    assert result["ok"] is False
    assert "order quantity" in result["error"]
    # And nothing of prospect A is left to leak if the model retries.
    assert ctx.state["sa_profile"].get("preferred_day") is None


def test_address_change_before_any_decision_still_merges():
    """The resolve_address confirmation flow: a corrected address arrives BEFORE a
    decision exists and must keep the cases already collected."""
    ctx = _FakeToolContext()
    intake_customer(address="1200 McKiney St, Houston, TX", order_quantity_cases=90,
                    tool_context=ctx)
    result = intake_customer(address=_ADDR_A, tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["order_quantity_cases"] == 90  # kept, not reset


def test_same_address_reformatted_after_decision_is_not_a_new_prospect():
    """Case/punctuation/whitespace differences are formatting, not a new customer:
    a spurious reset would drop fields over 'St.' vs 'St'."""
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)

    result = intake_customer(
        address="1200  mckinney st., houston, tx 77010", tool_context=ctx
    )
    assert result["ok"] is True
    assert result["profile"]["order_quantity_cases"] == 90  # merged, not reset


def test_revision_without_address_after_decision_still_merges():
    """'try 20 cases' after a recommendation is a supported revision."""
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)

    result = intake_customer(order_quantity_cases=20, tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["address"] == _ADDR_A
    assert result["profile"]["order_quantity_cases"] == 20
    assert result["profile"]["preferred_day"] == "TUE"


def test_prospect_scoped_keys_stay_in_sync_with_triage():
    """The triage keys are spelled as literals in _PROSPECT_SCOPED_STATE_KEYS
    (importing them back would be a cycle); fail loudly if either side renames."""
    from smart_assignment.tools.slot_recommendation import _PROSPECT_SCOPED_STATE_KEYS
    from smart_assignment.triage.context import (
        _STATE_TRIAGE_CHECK_COUNT_KEY,
        _STATE_TRIAGE_GROUNDING_KEY,
    )

    assert _STATE_TRIAGE_GROUNDING_KEY in _PROSPECT_SCOPED_STATE_KEYS
    assert _STATE_TRIAGE_CHECK_COUNT_KEY in _PROSPECT_SCOPED_STATE_KEYS


def test_batch_shape_cannot_trigger_the_reset():
    """Batch seeds a FRESH session per prospect and intake runs before any
    decision, so the guard's decision-exists condition can never hold there --
    even if the model re-passes the seeded address reformatted. Pins the property
    the batch runner's isolation now explicitly relies on."""
    ctx = _FakeToolContext()
    # Exactly what batch/agent_runner.create_session seeds: profile, no decision.
    ctx.state["sa_profile"] = {
        "name": "Katy Mills",
        "address": _ADDR_B,
        "order_quantity_cases": 260,
        "customer_number": None,
        "preferred_day": None,
        "preferred_window_start": None,
        "preferred_window_end": None,
    }
    # The model echoes the address back, reformatted -- and even a DIFFERENT
    # address must merge here, because no decision exists yet.
    result = intake_customer(address="5000 katy mills cir, katy tx 77494",
                             tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["order_quantity_cases"] == 260


# --- The explicit start_new_prospect boundary (phase 2) -----------------------
#
# The deterministic guard above cannot fire when no address is passed, or before
# a decision exists -- "another customer, 40 cases" is byte-identical at the tool
# level to the revision "try 40 cases". Only the model sees the words, so it
# declares the switch by calling start_new_prospect. A dedicated NO-ARG tool, not
# an intake_customer parameter, deliberately: the golden eval pins intake's
# argument dict exactly (an extra argument flaked it, measured live), while the
# IN_ORDER trajectory matcher tolerates extra tool CALLS -- so even a spurious
# boundary call cannot flake the eval. And it is safe in exactly one direction:
# it only DISCARDS state (worst case: a re-ask), never carries state over.


def test_start_new_prospect_resets_even_before_any_decision():
    """Mid-intake switch: prospect A never concluded, so the deterministic guard
    stays out of it -- the boundary tool is the only thing that can reset here."""
    ctx = _FakeToolContext()
    intake_customer(
        address=_ADDR_A, order_quantity_cases=90,
        preferred_day="TUE", preferred_window_start="07:00",
        preferred_window_end="10:00", tool_context=ctx,
    )
    assert start_new_prospect(tool_context=ctx)["ok"] is True
    result = intake_customer(address=_ADDR_B, order_quantity_cases=260,
                             tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["preferred_day"] is None


def test_start_new_prospect_then_no_address_asks_for_one():
    """'Another customer, 40 cases' -- no address given. The boundary discards A's
    profile, so intake asks for the address instead of silently deciding the NEW
    customer's order against the OLD customer's address."""
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)

    start_new_prospect(tool_context=ctx)
    result = intake_customer(order_quantity_cases=40, tool_context=ctx)
    assert result["ok"] is False
    assert "address" in result["error"]
    assert ctx.state["sa_profile"].get("address") is None  # A is gone
    assert not ctx.state.get("sa_last_recommendation")  # A's decision too


def test_start_new_prospect_clears_triage_state_too():
    ctx = _FakeToolContext()
    _concluded_prospect_a(ctx)
    ctx.state["sa_triage_grounding"] = {"figures": [90]}

    start_new_prospect(tool_context=ctx)
    from smart_assignment.tools.slot_recommendation import _PROSPECT_SCOPED_STATE_KEYS

    for key in _PROSPECT_SCOPED_STATE_KEYS:
        assert not ctx.state.get(key)
    assert not ctx.state.get("sa_profile")


def test_start_new_prospect_on_empty_state_is_a_harmless_no_op():
    """A spurious call on the first customer of a conversation (the model
    following the words "new prospect") must cost nothing."""
    ctx = _FakeToolContext()
    assert start_new_prospect(tool_context=ctx)["ok"] is True
    result = intake_customer(address=_ADDR_A, order_quantity_cases=90,
                             tool_context=ctx)
    assert result["ok"] is True
    assert result["profile"]["address"] == _ADDR_A


def test_intake_customer_signature_matches_the_golden_eval_pin():
    """The golden eval pins intake's argument dict exactly, so intake must not
    grow model-visible parameters (that is what start_new_prospect is for -- an
    extra CALL is tolerated by the IN_ORDER matcher; an extra ARGUMENT is not)."""
    import inspect

    params = set(inspect.signature(intake_customer).parameters)
    assert params == {
        "tool_context", "address", "order_quantity_cases", "preferred_day",
        "preferred_window_start", "preferred_window_end", "customer_number",
        "name", "clear_preferred_slot",
    }


def test_batch_agent_does_not_get_the_boundary_tool():
    """Batch seeds a fresh session per prospect; a boundary tool there could only
    spuriously discard the CRM-seeded profile. Its tool surface stays identical."""
    from smart_assignment.agent import _batch_agent_tools
    from smart_assignment.shared.config import Config

    tools = _batch_agent_tools(Config(use_escalation_triage=False))
    names = {getattr(t, "name", None) or getattr(t.func, "__name__", None) for t in tools}
    assert "start_new_prospect" not in names


def test_interactive_agent_wires_the_boundary_tool():
    """The tool must actually be offered on the conversational surfaces -- adk
    web/run and the webapp all build _build_root_agent's tool list."""
    import inspect

    from smart_assignment import agent as agent_module

    source = inspect.getsource(agent_module._build_root_agent)
    assert "start_new_prospect" in source


def test_instruction_scopes_the_boundary_to_switching_customers():
    """The guidance must exist and must scope the call to a SWITCH -- the golden
    eval cases are single-prospect, so the model has no reason to call it there
    (and a spurious call is tolerated by the matcher anyway)."""
    from smart_assignment.prompts import build_instruction

    text = " ".join(build_instruction().split())  # collapse line breaks
    assert "call start_new_prospect FIRST" in text
    assert "DIFFERENT customer" in text
    assert "Never call it for a correction or revision" in text
