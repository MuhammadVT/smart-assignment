"""
Step 2 (`pipeline.geo_lookup`): the Top-N proximity cut, and the preferred-day
candidate that rides alongside it.

A purely distance-based cut is day-blind, so it can drop every route running on
the day the customer asked for -- before scoring, before any decision layer sees
anything. `Config.use_preferred_day_candidate` (default ON) keeps the nearest
preferred-day route as one ADDITIONAL candidate in that case, capped at the
service-area limit so a provably-unserviceable route is never added. These tests
pin the rule, the cap, and the flag-off equivalence.

Routes are built inline with explicit coordinates rather than drawn from the mock
dataset, so the proximity ordering under test is exact and independent of it.
"""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.pipeline import geo_lookup
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import Geocoder
from smart_assignment.shared.models import (
    CustomerProfile,
    DayOfWeek,
    GeoPoint,
    PreferredSlot,
    Route,
)

# The prospect sits at the origin below; every route's distance is a function of
# how far east its service center is, so the proximity ranking is by construction.
_CUSTOMER_POINT = GeoPoint(29.7600, -95.3700)


class _FixedGeocoder(Geocoder):
    """Resolves any address to one point -- the routes carry the geometry."""

    def geocode(self, address: str) -> GeoPoint:
        return _CUSTOMER_POINT


def _route(
    route_id: str,
    day: DayOfWeek,
    east_degrees: float,
    service_radius_miles: float | None = None,
    avg_load_cases: float = 100,
) -> Route:
    """A route whose only distinguishing features are its day and its distance
    (east of the prospect, so larger `east_degrees` == farther). One degree of
    longitude at this latitude is ~60 mi."""
    return Route(
        route_id=route_id,
        name=f"Route {route_id}",
        day=day,
        service_center=GeoPoint(_CUSTOMER_POINT.latitude, _CUSTOMER_POINT.longitude + east_degrees),
        service_radius_miles=service_radius_miles,
        vehicle_capacity_cases=1000,
        avg_load_cases=avg_load_cases,
        available_windows=[(time(7, 0), time(10, 0))],
    )


# Nearest first: MON, WED, FRI, TUE, THU, THU. The Top-3 (MON/WED/FRI) contains
# neither Tuesday nor Thursday -- exactly the shape that motivated the rule.
def _routes() -> list[Route]:
    return [
        _route("R-MON", DayOfWeek.MON, 0.01),
        _route("R-WED", DayOfWeek.WED, 0.02),
        _route("R-FRI", DayOfWeek.FRI, 0.03),
        _route("R-TUE", DayOfWeek.TUE, 0.04),
        _route("R-THU-NEAR", DayOfWeek.THU, 0.05),
        _route("R-THU-FAR", DayOfWeek.THU, 0.06),
    ]


def _customer(preferred_day: DayOfWeek | None = DayOfWeek.THU) -> CustomerProfile:
    slot = (
        PreferredSlot(preferred_day, (time(9, 0), time(12, 0)))
        if preferred_day is not None
        else None
    )
    return CustomerProfile(
        name="Preference Prospect",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=150,
        preferred_slot=slot,
    )


def _ids(routes: list[Route]) -> list[str]:
    return [r.route_id for r in routes]


def _lookup(customer: CustomerProfile, config: Config, routes=None) -> list[Route]:
    all_routes = routes if routes is not None else _routes()
    return geo_lookup(customer, all_routes, _FixedGeocoder(), config)


def test_nearest_preferred_day_route_is_added_beyond_the_top_n():
    """The Top-3 nearest run MON/WED/FRI; the customer asked for Thursday. The
    nearest Thursday route joins as a FOURTH candidate -- the Top-N is kept
    intact, not displaced."""
    candidates = _lookup(_customer(DayOfWeek.THU), Config(use_preferred_day_candidate=True))

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI", "R-THU-NEAR"]


def test_the_added_route_is_the_nearest_of_the_preferred_day_routes():
    """Two Thursday routes qualify; only the nearer one is added -- exactly one
    extra candidate, never a sweep of the day."""
    candidates = _lookup(_customer(DayOfWeek.THU), Config(use_preferred_day_candidate=True))

    assert "R-THU-NEAR" in _ids(candidates)
    assert "R-THU-FAR" not in _ids(candidates)
    assert len(candidates) == Config().top_n_candidate_routes + 1


def test_flag_off_reproduces_the_day_blind_top_n_exactly():
    candidates = _lookup(_customer(DayOfWeek.THU), Config(use_preferred_day_candidate=False))

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


def test_no_stated_preference_adds_nothing():
    """With no preferred slot there is no day to honour, so the candidate set is
    the plain Top-N even with the flag on."""
    candidates = _lookup(_customer(preferred_day=None), Config(use_preferred_day_candidate=True))

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


def test_preferred_day_already_in_the_top_n_is_not_duplicated():
    """A Monday preference is already served by the nearest route, so nothing is
    added -- and R-MON appears once."""
    candidates = _lookup(_customer(DayOfWeek.MON), Config(use_preferred_day_candidate=True))

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


def test_preferred_day_absent_from_every_route_adds_nothing():
    """Nothing runs on Saturday: the rule finds no route to add and the candidate
    set is unchanged (no error, no empty placeholder)."""
    candidates = _lookup(_customer(DayOfWeek.SAT), Config(use_preferred_day_candidate=True))

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


# --- the service-area cap ----------------------------------------------------


def _near_top_three() -> list[Route]:
    return [
        _route("R-MON", DayOfWeek.MON, 0.01),
        _route("R-WED", DayOfWeek.WED, 0.02),
        _route("R-FRI", DayOfWeek.FRI, 0.03),
    ]


def test_an_out_of_range_preferred_day_route_is_not_added():
    """A route beyond the service-area limit would provably fail
    `geographic_serviceability`, so it is never added -- a guaranteed-rejected
    candidate in front of a specialist is noise, not a diagnostic."""
    routes = _near_top_three() + [
        _route("R-THU-REMOTE", DayOfWeek.THU, 2.0),  # ~120 mi east, beyond the 25 mi cap
    ]

    candidates = _lookup(
        _customer(DayOfWeek.THU), Config(use_preferred_day_candidate=True), routes=routes
    )

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


def test_the_cap_honours_a_routes_own_radius_not_just_the_global_ceiling():
    """`service_distance_limit` is the global cap TIGHTENED by a route's own
    radius. A Thursday route 18 mi out is inside the 25 mi ceiling but outside
    its own 12 mi radius, so it still cannot serve this customer and is not
    added."""
    routes = _near_top_three() + [
        _route("R-THU-NARROW", DayOfWeek.THU, 0.30, service_radius_miles=12.0),  # ~18 mi
    ]
    config = Config(use_preferred_day_candidate=True, max_service_distance_miles=25.0)

    candidates = _lookup(_customer(DayOfWeek.THU), config, routes=routes)

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI"]


def test_the_scan_continues_past_an_out_of_range_route_to_a_serviceable_one():
    """Giving up at the first out-of-range preferred-day route would miss a
    farther one that declares a wider radius. The nearest SERVICEABLE route on
    the day is what gets added."""
    routes = _near_top_three() + [
        _route("R-THU-NARROW", DayOfWeek.THU, 0.30, service_radius_miles=12.0),  # ~18 mi, out
        _route("R-THU-WIDE", DayOfWeek.THU, 0.35, service_radius_miles=40.0),  # ~21 mi, in
    ]
    config = Config(use_preferred_day_candidate=True, max_service_distance_miles=25.0)

    candidates = _lookup(_customer(DayOfWeek.THU), config, routes=routes)

    assert _ids(candidates) == ["R-MON", "R-WED", "R-FRI", "R-THU-WIDE"]


def test_the_cap_is_on_distance_only_so_a_full_preferred_day_route_is_still_added():
    """Capacity is NOT part of the cap. An in-range Thursday route that is too
    full is added, fails `route_capacity`, and reaches the specialist as a
    rejected candidate -- a diagnostic a human can act on (split the order, move
    a stop), unlike distance."""
    routes = _near_top_three() + [
        _route("R-THU-FULL", DayOfWeek.THU, 0.05, avg_load_cases=980),  # 980+150 > 1000
    ]
    config = Config(use_preferred_day_candidate=True)
    customer = _customer(DayOfWeek.THU)

    candidates = _lookup(customer, config, routes=routes)
    assert "R-THU-FULL" in _ids(candidates)

    from smart_assignment.pipeline import evaluate_candidates

    evaluations = evaluate_candidates(customer, candidates, config)
    full = next(e for e in evaluations if e.route.route_id == "R-THU-FULL")
    assert full.feasible is False
    assert full.total_score == 0.0  # rejected candidates never carry merit
    assert [c.name for c in full.failed_constraints] == ["route_capacity"]


@pytest.mark.parametrize("top_n", [1, 2, 5])
def test_the_rule_composes_with_any_top_n(top_n: int):
    """Whatever the cut size, the invariant holds: at most one extra candidate,
    and a preferred-day route is present whenever one exists."""
    config = Config(top_n_candidate_routes=top_n, use_preferred_day_candidate=True)
    candidates = _lookup(_customer(DayOfWeek.THU), config)

    assert len(candidates) <= top_n + 1
    assert any(r.day == DayOfWeek.THU for r in candidates)
    assert len(_ids(candidates)) == len(set(_ids(candidates)))  # no duplicates


def test_config_reads_the_flag_from_the_environment(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_USE_PREFERRED_DAY_CANDIDATE", "false")
    assert Config.from_env().use_preferred_day_candidate is False

    monkeypatch.setenv("SMART_ASSIGNMENT_USE_PREFERRED_DAY_CANDIDATE", "true")
    assert Config.from_env().use_preferred_day_candidate is True

    monkeypatch.delenv("SMART_ASSIGNMENT_USE_PREFERRED_DAY_CANDIDATE")
    assert Config.from_env().use_preferred_day_candidate is True  # default ON
