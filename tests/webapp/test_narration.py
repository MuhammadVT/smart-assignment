"""
Tests for the live-step narration copy (smart_assignment/webapp/narration.py).

The narration is descriptive breadcrumbs only -- these lock in the labels, the
generic per-step descriptions, and the grounded Intake read-back (which must
echo the customer's own inputs and never invent one).
"""

from __future__ import annotations

from smart_assignment.webapp.narration import STEP_LABELS, step_detail, step_label


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
