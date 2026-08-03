"""
Tests for the live-step narration copy (smart_assignment/webapp/narration.py).

The narration is descriptive breadcrumbs only -- these lock in the labels, the
generic per-step descriptions, and the grounded Intake read-back (which must
echo the customer's own inputs and never invent one).
"""

from __future__ import annotations

from smart_assignment.webapp.narration import (
    HANDOFF_STEPS,
    STEP_LABELS,
    step_detail,
    step_label,
    step_phase,
    tool_steps,
)


def test_step_label_maps_pipeline_tools():
    assert step_label("intake_customer") == "Intake"
    assert step_label("find_candidate_routes") == "Geo-Lookup"
    assert step_label("evaluate_and_score_routes") == "Score & Rank"
    assert step_label("recommend_or_escalate") == "Recommend / Decide"


def test_step_label_none_for_non_step_tools():
    assert step_label("resolve_address") is None
    assert step_label("some_other_tool") is None


def test_step_detail_generic_lines_present_for_every_step():
    for name in STEP_LABELS:
        assert step_detail(name), f"missing narration for {name}"


def test_intake_detail_reads_back_the_customers_inputs():
    detail = step_detail(
        "intake_customer",
        {"order_quantity_cases": 150, "preferred_day": "thu"},
    )
    assert "150 cases" in detail
    assert "THU" in detail  # normalised to upper-case


def test_intake_detail_handles_partial_call_gracefully():
    # An address-only first call has nothing to echo -> fall back to the generic
    # description rather than inventing an order size.
    detail = step_detail("intake_customer", {"address": "1200 McKinney St"})
    assert detail == step_detail("intake_customer", None)
    assert "cases" not in detail


def test_intake_detail_ignores_bogus_values():
    # Booleans and non-positive quantities are not real order sizes.
    assert step_detail("intake_customer", {"order_quantity_cases": True}) == step_detail(
        "intake_customer", None
    )
    assert step_detail("intake_customer", {"order_quantity_cases": 0}) == step_detail(
        "intake_customer", None
    )


def test_step_detail_none_for_non_step_tool():
    assert step_detail("resolve_address", {}) is None


# --- tool -> pipeline steps (breadcrumbs decoupled from tool count) -----------


def test_tool_steps_single_step_tools_map_to_themselves():
    assert tool_steps("intake_customer") == ["intake_customer"]
    assert tool_steps("find_candidate_routes") == ["find_candidate_routes"]


def test_recommend_or_escalate_covers_geo_score_and_decide():
    # One tool call that internally runs geo + score + decide lights up all three.
    assert tool_steps("recommend_or_escalate") == [
        "find_candidate_routes",
        "evaluate_and_score_routes",
        "recommend_or_escalate",
    ]


def test_every_tool_step_has_a_label_and_detail():
    for tool in ("intake_customer", "recommend_or_escalate", "assign_prospect"):
        for step in tool_steps(tool):
            assert step_label(step), f"no label for step {step}"
            assert step_detail(step), f"no detail for step {step}"


def test_tool_steps_empty_for_non_pipeline_tool():
    assert tool_steps("resolve_address") == []
    assert tool_steps("some_other_tool") == []


# --- the handoff phase (escalation) ------------------------------------------
#
# Composing the specialist brief is the longest call in an escalation turn. It is
# a step so the stepper isn't sitting fully ticked while it runs, and it carries a
# phase so the UI can show it as a change of hands, not a fifth pipeline step.


def test_escalation_triage_is_a_narrated_step():
    assert tool_steps("escalation_triage") == ["escalation_triage"]
    assert step_label("escalation_triage") == "Briefing a specialist"
    assert step_detail("escalation_triage")


def test_only_the_handoff_steps_carry_the_handoff_phase():
    assert step_phase("escalation_triage") == "handoff"
    for step in tool_steps("assign_prospect"):  # every assignment step
        assert step_phase(step) is None
    assert step_phase("some_other_tool") is None


def test_handoff_steps_are_narrated_like_any_other_step():
    # The phase changes only how a step is DISPLAYED -- it still needs the same
    # label + description every breadcrumb has.
    for step in HANDOFF_STEPS:
        assert step_label(step), f"no label for handoff step {step}"
        assert step_detail(step), f"no detail for handoff step {step}"
