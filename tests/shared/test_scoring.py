"""
Unit tests for the route-level factors in the scoring layer (shared/scoring.py).

The (route, slot) composition itself -- slot openness, the window_match day
gate, and which factors are active -- is covered in tests/routeslot/test_scoring.
"""

from __future__ import annotations

from smart_assignment.shared.constraints import build_context
from smart_assignment.shared.scoring import capacity_buffer, score_route_slot
from smart_assignment.shared.slot_selection import identify_available_slots


def _first_slot(customer, route, config):
    slots = identify_available_slots(customer.location, route, config)
    assert slots, "the fixture route should yield at least one candidate slot"
    return slots[0]


def test_score_is_normalized(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    slot = _first_slot(sample_customer, open_route, config)
    breakdown, total = score_route_slot(sample_customer, open_route, ctx, slot, config)
    assert 0.0 <= total <= 1.0
    # sample_customer states a preference, so all four factors are active.
    assert {f.name for f in breakdown} == {
        "geographic_clustering",
        "capacity_buffer",
        "window_match",
        "slot_availability",
    }


def test_capacity_buffer_flat_within_safe_zone(sample_customer, open_route, config):
    # Below the safety margin (default 15pp under the 90% ceiling, i.e. <=75%
    # utilized), the score is flat at 1.0 -- an almost-empty route and a
    # fairly busy-but-still-safe route score identically. This is the bias
    # fix: extra headroom beyond "safe" no longer buys extra score.
    open_route.avg_load_cases = 10
    ctx_empty = build_context(sample_customer, open_route)
    f_empty = capacity_buffer(sample_customer, open_route, ctx_empty, config)

    open_route.avg_load_cases = 600
    ctx_busier = build_context(sample_customer, open_route)
    f_busier = capacity_buffer(sample_customer, open_route, ctx_busier, config)

    assert ctx_busier.utilization_after > ctx_empty.utilization_after  # sanity: busier is busier
    assert f_empty.value == 1.0
    assert f_busier.value == 1.0


def test_capacity_buffer_decays_between_safe_line_and_ceiling(sample_customer, open_route, config):
    # 700 committed + 90 new = 79% utilized, inside the 75-90% decay band.
    open_route.avg_load_cases = 700
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert 0.0 < f.value < 1.0


def test_capacity_buffer_reaches_zero_at_the_ceiling(sample_customer, open_route, config):
    # 810 committed + 90 new = exactly 90% utilized -- right at the hard ceiling.
    open_route.avg_load_cases = 810
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert f.value == 0.0


def test_factor_weights_respect_config(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    slot = _first_slot(sample_customer, open_route, config)
    breakdown, _ = score_route_slot(sample_customer, open_route, ctx, slot, config)
    weights = {f.name: f.weight for f in breakdown}
    # Priority order: clustering > capacity buffer > the slot-level factors.
    assert weights["geographic_clustering"] > weights["capacity_buffer"]
    assert weights["capacity_buffer"] > weights["window_match"]
    assert weights["window_match"] == weights["slot_availability"]
