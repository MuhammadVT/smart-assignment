"""
Round-tripping a decision through JSON-able session state
(SlotRecommendation.to_state_dict / from_state_dict).

This exists because step 5 is non-deterministic once grounded reasoning is on:
a surface that has to re-render the SAME decision must carry it rather than
re-decide. If the round-trip silently drops a field, the card would quietly lose
part of the explanation, so these tests pin every field.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from smart_assignment.shared.models import Decision, FactorScore, SlotRecommendation


def _full_recommendation() -> SlotRecommendation:
    """Every field populated with a distinct value, so a dropped one shows up."""
    return SlotRecommendation(
        customer_name="Bayou City Bistro",
        decision=Decision.ESCALATED_LOW_SCORE,
        total_score=0.54,
        reasoning="RTE-4100 is the strongest route-slot overall.",
        customer_number="067-100001",
        customer_address="1200 McKinney St, Houston, TX 77010",
        recommended_route_id="RTE-4100",
        recommended_route_name="Central Houston",
        recommended_day="TUE",
        recommended_window="07:20-10:20",
        recommended_window_basis="between_adjacent_stops",
        recommended_window_rationale="Sits between two nearby morning stops.",
        decision_summary="Assign RTE-4100 - Central Houston.",
        primary_reasons=["Geographic fit: avg 1.6 mi.", "Capacity headroom: 440 cases."],
        key_tradeoff="Gives up slot openness for clustering.",
        runner_up="RTE-4110 (WED) 08:30-11:30 - scored 0.59",
        default_comparison="Agreed with the weighted-heuristic default.",
        factor_breakdown=[
            FactorScore(name="geographic_clustering", weight=0.35, value=0.89, detail="avg 1.6 mi"),
            FactorScore(name="slot_availability", weight=0.20, value=0.26, detail="contention 2.8"),
        ],
        rejected_alternatives=["RTE-4200 (WED): infeasible - service area"],
        review_reason="Below the 55% auto-assign bar.",
        alternative_takes=["[RECOMMEND/LOW] RTE-4100: marginal but workable."],
        grounded_fallback=True,
        grounded_fallback_reason="Grounded reasoning was unavailable.",
    )


def test_every_declared_field_is_carried():
    # A guard against silent drift: adding a field to SlotRecommendation without
    # adding it to to_state_dict would lose it on the round-trip.
    covered = set(_full_recommendation().to_state_dict())
    declared = {f.name for f in fields(SlotRecommendation)}
    assert declared == covered


def test_round_trip_is_lossless():
    original = _full_recommendation()
    assert SlotRecommendation.from_state_dict(original.to_state_dict()) == original


def test_round_trip_survives_the_empty_defaults():
    minimal = SlotRecommendation(
        customer_name="No Feasible Co",
        decision=Decision.ESCALATED_NO_FEASIBLE_SLOT,
        total_score=0.0,
        reasoning="No candidate route satisfied all hard constraints.",
    )
    assert SlotRecommendation.from_state_dict(minimal.to_state_dict()) == minimal


def test_state_dict_is_json_safe():
    import json

    payload = json.dumps(_full_recommendation().to_state_dict())
    assert SlotRecommendation.from_state_dict(json.loads(payload)) == _full_recommendation()


def test_unknown_decision_value_raises():
    # A corrupt snapshot must fail loudly so the caller recomputes rather than
    # rendering something wrong.
    data = _full_recommendation().to_state_dict()
    data["decision"] = "NOT_A_DECISION"
    with pytest.raises(ValueError):
        SlotRecommendation.from_state_dict(data)
