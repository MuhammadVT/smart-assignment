"""
Unit tests for the hard-constraint layer (shared/constraints.py) and the geo
helpers it relies on. Deterministic, fast, no LLM or network.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.constraints import (
    build_context,
    delivery_window_compatibility,
    evaluate_constraints,
    geographic_serviceability,
    route_capacity,
)
from smart_assignment.shared.geo import haversine_miles
from smart_assignment.shared.models import GeoPoint


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


def test_window_incompatible_is_rejected(sample_customer, open_route, config):
    # Customer prefers 07:00-10:00; route only offers an afternoon window.
    open_route.available_windows = [(time(13, 0), time(15, 0))]
    ctx = build_context(sample_customer, open_route)
    outcome = delivery_window_compatibility(sample_customer, open_route, ctx, config)
    assert outcome.passed is False


def test_no_preference_passes_window_check(sample_customer, open_route, config):
    sample_customer.preferred_window = None
    ctx = build_context(sample_customer, open_route)
    outcome = delivery_window_compatibility(sample_customer, open_route, ctx, config)
    assert outcome.passed is True
