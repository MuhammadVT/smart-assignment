"""
Unit tests for the hard-constraint layer (shared/constraints.py) and the geo
helpers it relies on. Deterministic, fast, no LLM or network.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.constraints import (
    HARD_CONSTRAINTS,
    applicable_preferred_window,
    build_context,
    evaluate_constraints,
    geographic_serviceability,
    route_capacity,
    service_distance_limit,
)
from smart_assignment.shared.geo import haversine_miles
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, GeoPoint


def _all_pass(customer, route, config) -> bool:
    ctx = build_context(customer, route)
    return all(o.passed for o in evaluate_constraints(customer, route, ctx, config))


def test_haversine_known_distance():
    # Downtown Houston -> The Woodlands is roughly 28-30 miles.
    d = haversine_miles(GeoPoint(29.7570, -95.3670), GeoPoint(30.1620, -95.4590))
    assert 26 < d < 32


def test_nearby_open_route_is_feasible(sample_customer, open_route, config):
    assert _all_pass(sample_customer, open_route, config)


def test_full_route_fails_capacity(sample_customer, full_route, config):
    ctx = build_context(sample_customer, full_route)
    outcome = route_capacity(sample_customer, full_route, ctx, config)
    assert outcome.passed is False


def test_far_route_fails_serviceability(sample_customer, far_route, config):
    ctx = build_context(sample_customer, far_route)
    outcome = geographic_serviceability(sample_customer, far_route, ctx, config)
    assert outcome.passed is False


# --- the shared service-distance limit ----------------------------------------


def test_service_distance_limit_is_the_global_ceiling_without_a_route_radius(
    open_route, config
):
    open_route.service_radius_miles = None
    assert service_distance_limit(open_route, config) == config.max_service_distance_miles


def test_service_distance_limit_takes_the_tighter_of_the_two(open_route, config):
    # A route's own radius only ever NARROWS the global ceiling...
    open_route.service_radius_miles = 12.0
    assert service_distance_limit(open_route, config) == 12.0

    # ...it can never widen past it, so nothing is ever admitted beyond
    # max_service_distance_miles (SMART_ASSIGNMENT_MAX_SERVICE_MILES).
    open_route.service_radius_miles = 100.0
    assert service_distance_limit(open_route, config) == config.max_service_distance_miles


def test_geographic_serviceability_gates_on_that_same_limit(sample_customer, open_route, config):
    """The constraint and `pipeline.geo_lookup`'s preferred-day filter share this
    one definition, so they can never disagree about what "in range" means."""
    ctx = build_context(sample_customer, open_route)
    limit = service_distance_limit(open_route, config)

    assert geographic_serviceability(sample_customer, open_route, ctx, config).passed is (
        ctx.distance_miles <= limit
    )


def test_hard_constraints_exclude_delivery_window():
    # The preferred delivery window is a soft (scoring) preference, not a hard
    # constraint — only serviceability and capacity gate feasibility.
    names = {fn.__name__ for fn in HARD_CONSTRAINTS}
    assert names == {"geographic_serviceability", "route_capacity"}


def test_window_mismatch_does_not_make_route_infeasible(sample_customer, open_route, config):
    # Route only offers an afternoon window that misses the morning preference,
    # yet the route stays feasible (window is not a hard rule).
    from datetime import time

    open_route.available_windows = [(time(13, 0), time(15, 0))]
    assert _all_pass(sample_customer, open_route, config)


# --- the preference day gate --------------------------------------------------


def test_applicable_preferred_window_is_none_on_a_different_day(sample_customer, open_route):
    # sample_customer prefers TUE 07:00-10:00; open_route runs TUE.
    assert applicable_preferred_window(sample_customer, open_route) == (time(7, 0), time(10, 0))

    # Same clock window, wrong day -> the preference simply doesn't apply.
    open_route.day = DayOfWeek.WED
    assert applicable_preferred_window(sample_customer, open_route) is None


def test_applicable_preferred_window_is_none_without_a_preference(open_route):
    customer = CustomerProfile(
        name="No Preference Cafe",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=50,
        preferred_slot=None,
        location=GeoPoint(29.7570, -95.3670),
    )
    assert applicable_preferred_window(customer, open_route) is None


def test_window_overlap_is_zero_on_a_wrong_day_route(sample_customer, open_route):
    # Right day: the route's slot overlaps the preferred hours, so the reference
    # fact reports real overlap...
    ctx_same_day = build_context(sample_customer, open_route)
    assert ctx_same_day.window_overlap_minutes > 0

    # ...but the identical clock window on WED earns nothing. A customer who
    # asked for Tuesday cannot receive on Wednesday, so crediting a time-of-day
    # match there would overstate the fit (mirrors scoring._slot_window_match).
    open_route.day = DayOfWeek.WED
    ctx_wrong_day = build_context(sample_customer, open_route)
    assert ctx_wrong_day.window_overlap_minutes == 0


def test_wrong_day_route_does_not_keep_extra_preference_candidates(sample_customer, open_route):
    # The always-keep rule is preference-driven, so on a day the customer didn't
    # ask for it must not pad the menu beyond the quality top-N.
    same_day = build_context(sample_customer, open_route).available_slots
    open_route.day = DayOfWeek.WED
    wrong_day = build_context(sample_customer, open_route).available_slots
    assert len(wrong_day) <= len(same_day)
