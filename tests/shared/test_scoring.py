"""
Unit tests for the weighted scoring layer (shared/scoring.py).
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.constraints import build_context
from smart_assignment.shared.models import DayOfWeek
from smart_assignment.shared.scoring import capacity_buffer, score_candidate, window_match


def test_score_is_normalized(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, total = score_candidate(sample_customer, open_route, ctx, config)
    assert 0.0 <= total <= 1.0
    assert {f.name for f in breakdown} == {
        "geographic_clustering",
        "capacity_buffer",
        "window_match",
    }


def test_capacity_buffer_flat_within_safe_zone(sample_customer, open_route, config):
    # Below the safety margin (default 15pp under the 90% ceiling, i.e. <=75%
    # utilized), the score is flat at 1.0 -- an almost-empty route and a
    # fairly busy-but-still-safe route score identically. This is the bias
    # fix: extra headroom beyond "safe" no longer buys extra score.
    open_route.committed_stops[0].case_volume = 10
    ctx_empty = build_context(sample_customer, open_route)
    f_empty = capacity_buffer(sample_customer, open_route, ctx_empty, config)

    open_route.committed_stops[0].case_volume = 600
    ctx_busier = build_context(sample_customer, open_route)
    f_busier = capacity_buffer(sample_customer, open_route, ctx_busier, config)

    assert ctx_busier.utilization_after > ctx_empty.utilization_after  # sanity: busier is busier
    assert f_empty.value == 1.0
    assert f_busier.value == 1.0


def test_capacity_buffer_decays_between_safe_line_and_ceiling(sample_customer, open_route, config):
    # 700 committed + 90 new = 79% utilized, inside the 75-90% decay band.
    open_route.committed_stops[0].case_volume = 700
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert 0.0 < f.value < 1.0


def test_capacity_buffer_reaches_zero_at_the_ceiling(sample_customer, open_route, config):
    # 810 committed + 90 new = exactly 90% utilized -- right at the hard ceiling.
    open_route.committed_stops[0].case_volume = 810
    ctx = build_context(sample_customer, open_route)
    f = capacity_buffer(sample_customer, open_route, ctx, config)
    assert f.value == 0.0


def test_factor_weights_respect_config(sample_customer, open_route, config):
    ctx = build_context(sample_customer, open_route)
    breakdown, _ = score_candidate(sample_customer, open_route, ctx, config)
    weights = {f.name: f.weight for f in breakdown}
    # Priority order from spec: clustering > capacity buffer > window match.
    assert weights["geographic_clustering"] > weights["capacity_buffer"]
    assert weights["capacity_buffer"] > weights["window_match"]


# --- window_match: day is a gate, not partial credit --------------------


def test_window_match_full_day_and_time_overlap_scores_one(sample_customer, open_route, config):
    # sample_customer prefers TUE 07:00-10:00; open_route runs TUE 07:00-10:00.
    ctx = build_context(sample_customer, open_route)
    f = window_match(sample_customer, open_route, ctx, config)
    assert f.value == 1.0


def test_window_match_wrong_day_scores_zero_despite_full_time_overlap(
    sample_customer, open_route, config
):
    # Same time-of-day window (07:00-10:00), but the route runs on WED, not
    # the TUE the customer asked for. A numerically-overlapping clock time on
    # the wrong day is not a real match.
    open_route.day = DayOfWeek.WED
    ctx = build_context(sample_customer, open_route)
    f = window_match(sample_customer, open_route, ctx, config)
    assert f.value == 0.0


def test_window_match_right_day_zero_time_overlap_scores_zero(
    sample_customer, open_route, config
):
    # Right day, but a window nowhere near the customer's preferred hours.
    open_route.available_windows = [(time(13, 0), time(15, 0))]
    ctx = build_context(sample_customer, open_route)
    f = window_match(sample_customer, open_route, ctx, config)
    assert f.value == 0.0


def test_window_match_right_day_partial_time_overlap_is_proportional(
    sample_customer, open_route, config
):
    # Right day, and a window that covers 2 of the preferred 3 hours (120 of
    # 180 minutes) -- partial credit is still fine once the day itself is
    # right; it's only a day mismatch (or zero overlap) that gates to 0.
    open_route.available_windows = [(time(8, 0), time(11, 0))]
    ctx = build_context(sample_customer, open_route)
    f = window_match(sample_customer, open_route, ctx, config)
    assert f.value == 120 / 180


def test_window_match_neutral_score_when_no_preference(open_route, config):
    from smart_assignment.shared.models import CustomerProfile, GeoPoint

    customer = CustomerProfile(
        customer_number="067-100099",
        name="No Preference Cafe",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=50,
        preferred_slot=None,
        location=GeoPoint(29.7570, -95.3670),
    )
    ctx = build_context(customer, open_route)
    f = window_match(customer, open_route, ctx, config)
    assert f.value == config.window_neutral_score
