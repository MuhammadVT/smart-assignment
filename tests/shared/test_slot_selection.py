"""
Unit tests for the location-aware delivery-slot selector
(shared/slot_selection.py). Deterministic, fast, no LLM or network.

Two steps are exercised separately and together:
  * identify_available_slots() — the ranked menu of a route's windows
    (location fit + contention), with no customer preference in the mix.
  * recommend_slot() — the final pick that THEN considers the preference,
    with a least-contended fallback.
"""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import DayOfWeek, GeoPoint, Route, RouteStop
from smart_assignment.shared.slot_selection import (
    SLOT_BASIS_BETWEEN_STOPS,
    SLOT_BASIS_LEAST_CONTENDED,
    SLOT_BASIS_NONE,
    SLOT_BASIS_PREFERENCE,
    _window_for_reference_time,
    identify_available_slots,
    nearest_neighbors,
    recommend_slot,
    stop_reference_time,
)

MORNING = (time(7, 0), time(10, 0))
AFTERNOON = (time(13, 0), time(15, 0))

# A downtown-Houston-ish anchor for the prospect; deltas below stay small so
# haversine ordering is unambiguous.
PROSPECT = GeoPoint(29.7500, -95.3600)


def _route(windows, stops) -> Route:
    return Route(
        route_id="RTE-T",
        name="Test Route",
        day=DayOfWeek.TUE,
        service_center=GeoPoint(29.7500, -95.3600),
        service_radius_miles=25.0,
        vehicle_capacity_cases=1000,
        avg_load_cases=100,
        available_windows=list(windows),
        committed_stops=list(stops),
    )


def _stop(lat, lng, window) -> RouteStop:
    return RouteStop("067-000000", GeoPoint(lat, lng), delivery_time_window=window)


# --- stop_reference_time: the phase A/B seam --------------------------------


def test_stop_reference_time_is_window_midpoint():
    assert stop_reference_time(_stop(29.75, -95.36, (time(8, 0), time(12, 0)))) == time(10, 0)
    assert stop_reference_time(_stop(29.75, -95.36, MORNING)) == time(8, 30)


def test_stop_reference_time_none_without_window():
    assert stop_reference_time(_stop(29.75, -95.36, None)) is None


# --- nearest_neighbors ------------------------------------------------------


def test_nearest_neighbors_sorted_and_capped():
    near = _stop(29.7501, -95.3601, MORNING)
    mid = _stop(29.7600, -95.3700, MORNING)
    far = _stop(29.9000, -95.5000, MORNING)
    got = nearest_neighbors(PROSPECT, [far, near, mid], k=2)
    assert [n.stop for n in got] == [near, mid]  # nearest first, capped at k
    assert got[0].distance_miles < got[1].distance_miles


def test_nearest_neighbors_respects_max_miles_and_nonpositive_k():
    near = _stop(29.7501, -95.3601, MORNING)
    far = _stop(29.9000, -95.5000, MORNING)
    # A tight cap drops the far stop entirely.
    capped = nearest_neighbors(PROSPECT, [near, far], k=5, max_miles=1.0)
    assert [n.stop for n in capped] == [near]
    # A non-positive k means "no neighbors vote".
    assert nearest_neighbors(PROSPECT, [near, far], k=0) == []


# --- _window_for_reference_time ---------------------------------------------


def test_window_for_reference_time_prefers_tightest_containing_window():
    wide = (time(7, 0), time(12, 0))
    tight = (time(8, 0), time(10, 0))
    # 09:00 sits inside both; the tighter window wins.
    assert _window_for_reference_time(time(9, 0), [wide, tight]) == tight


def test_window_for_reference_time_falls_back_to_nearest_by_gap():
    # 16:30 is inside neither; the afternoon window is far closer in time.
    assert _window_for_reference_time(time(16, 30), [MORNING, AFTERNOON]) == AFTERNOON


def test_window_for_reference_time_none_when_no_windows():
    assert _window_for_reference_time(time(9, 0), []) is None


# --- identify_available_slots: the menu -------------------------------------


def test_menu_lists_every_offered_window():
    route = _route([MORNING, AFTERNOON], [_stop(29.7501, -95.3601, MORNING)])
    windows = {o.window for o in identify_available_slots(PROSPECT, route, Config())}
    assert windows == {MORNING, AFTERNOON}


def test_location_fit_puts_the_neighbors_window_first_over_earlier_window():
    # The only nearby stop is served in the AFTERNOON; a far stop holds the
    # MORNING. Location fit must rank the afternoon window first even though the
    # morning window starts earlier.
    near_pm = _stop(29.7501, -95.3601, AFTERNOON)
    far_am = _stop(29.9000, -95.5000, MORNING)
    route = _route([MORNING, AFTERNOON], [near_pm, far_am])
    options = identify_available_slots(PROSPECT, route, Config())
    assert options[0].window == AFTERNOON
    assert options[0].basis == SLOT_BASIS_BETWEEN_STOPS
    assert options[0].fit_score > 0
    # The far morning stop still contributes a vote (default neighbor_count=3
    # sees both stops), but the very close afternoon stop dominates by
    # inverse-distance weight, so the afternoon window ranks first.
    morning = next(o for o in options if o.window == MORNING)
    assert morning.fit_score < options[0].fit_score
    assert options[0].fit_score > 0.9


def test_inverse_distance_weight_lets_one_close_stop_outvote_two_far_ones():
    close_pm = _stop(29.75005, -95.36005, AFTERNOON)  # essentially on top of the prospect
    far_am1 = _stop(29.8000, -95.4200, MORNING)
    far_am2 = _stop(29.8100, -95.4300, MORNING)
    route = _route([MORNING, AFTERNOON], [far_am1, far_am2, close_pm])
    options = identify_available_slots(PROSPECT, route, Config())  # neighbor_count=3
    assert options[0].window == AFTERNOON  # the single close stop wins on weight


def test_zero_fit_orders_by_least_contended_then_earliest_start():
    # Stops carry no windows -> no votes -> every fit is 0. Ordering then falls
    # to contention (both 0 here) and finally earliest start.
    route = _route([AFTERNOON, MORNING], [_stop(29.7501, -95.3601, None)])
    options = identify_available_slots(PROSPECT, route, Config())
    assert [o.window for o in options] == [MORNING, AFTERNOON]
    assert all(o.fit_score == 0.0 for o in options)
    assert all(o.basis == SLOT_BASIS_LEAST_CONTENDED for o in options)


def test_committed_overlap_counts_contention_and_orders_least_contended_first():
    # Force the fallback path (no neighbor votes) via a tight distance cap, so
    # ordering is driven purely by contention: MORNING has 3 stops, AFTERNOON 1.
    stops = [
        _stop(29.80, -95.42, MORNING),
        _stop(29.81, -95.43, MORNING),
        _stop(29.82, -95.44, MORNING),
        _stop(29.83, -95.45, AFTERNOON),
    ]
    route = _route([MORNING, AFTERNOON], stops)
    cfg = Config(slot_neighbor_max_miles=0.25)  # every stop is farther than this
    options = identify_available_slots(PROSPECT, route, cfg)
    by_window = {o.window: o for o in options}
    assert by_window[MORNING].committed_overlap == 3
    assert by_window[AFTERNOON].committed_overlap == 1
    assert options[0].window == AFTERNOON  # least contended first
    assert options[0].basis == SLOT_BASIS_LEAST_CONTENDED


def test_empty_route_windows_yield_empty_menu():
    route = _route([], [_stop(29.7501, -95.3601, MORNING)])
    assert identify_available_slots(PROSPECT, route, Config()) == []


# --- recommend_slot: the final pick, now considering preference -------------


def test_recommend_accommodates_preference_even_over_better_location_fit():
    # The nearby stop clusters the prospect into the AFTERNOON, but the customer
    # prefers the MORNING, which an offered window covers -> honor the preference.
    near_pm = _stop(29.7501, -95.3601, AFTERNOON)
    route = _route([MORNING, AFTERNOON], [near_pm])
    options = identify_available_slots(PROSPECT, route, Config())
    sel = recommend_slot(options, preferred_window=MORNING, config=Config())
    assert sel.window == MORNING
    assert sel.basis == SLOT_BASIS_PREFERENCE
    assert sel.overlap_minutes == 180


def test_recommend_picks_greatest_overlap_among_preference_matches():
    early = (time(7, 0), time(9, 0))
    late = (time(9, 0), time(12, 0))
    route = _route([early, late], [_stop(29.7501, -95.3601, None)])
    options = identify_available_slots(PROSPECT, route, Config())
    # Preference 08:00-11:00 overlaps early by 60 min, late by 120 min -> late.
    sel = recommend_slot(options, preferred_window=(time(8, 0), time(11, 0)), config=Config())
    assert sel.window == late
    assert sel.overlap_minutes == 120
    assert sel.basis == SLOT_BASIS_PREFERENCE


def test_recommend_still_returns_a_slot_when_preference_cannot_be_met():
    # Route only offers an afternoon window; the morning preference can't be
    # honored, but the route is NOT dropped -- it still gets a slot, overlap 0.
    near_pm = _stop(29.7501, -95.3601, AFTERNOON)
    route = _route([AFTERNOON], [near_pm])
    options = identify_available_slots(PROSPECT, route, Config())
    sel = recommend_slot(options, preferred_window=MORNING, config=Config())
    assert sel.window == AFTERNOON
    assert sel.overlap_minutes == 0
    assert sel.basis == SLOT_BASIS_BETWEEN_STOPS  # location fit still explains the pick


def test_recommend_no_preference_takes_top_of_menu():
    near_pm = _stop(29.7501, -95.3601, AFTERNOON)
    far_am = _stop(29.9000, -95.5000, MORNING)
    route = _route([MORNING, AFTERNOON], [near_pm, far_am])
    options = identify_available_slots(PROSPECT, route, Config())
    sel = recommend_slot(options, preferred_window=None, config=Config())
    assert sel.window == AFTERNOON  # best location fit
    assert sel.overlap_minutes == 0
    assert sel.basis == SLOT_BASIS_BETWEEN_STOPS


def test_recommend_falls_back_to_least_contended_when_no_fit_no_preference():
    stops = [
        _stop(29.80, -95.42, MORNING),
        _stop(29.81, -95.43, MORNING),
        _stop(29.82, -95.44, AFTERNOON),
    ]
    route = _route([MORNING, AFTERNOON], stops)
    cfg = Config(slot_neighbor_max_miles=0.25)
    options = identify_available_slots(PROSPECT, route, cfg)
    sel = recommend_slot(options, preferred_window=None, config=cfg)
    assert sel.window == AFTERNOON  # the emptier window
    assert sel.basis == SLOT_BASIS_LEAST_CONTENDED


def test_recommend_on_empty_menu_returns_no_window():
    sel = recommend_slot([], preferred_window=MORNING, config=Config())
    assert sel.window is None
    assert sel.basis == SLOT_BASIS_NONE
    assert sel.overlap_minutes == 0


def test_no_committed_stops_recommends_earliest_window():
    route = _route([AFTERNOON, MORNING], [])
    options = identify_available_slots(PROSPECT, route, Config())
    sel = recommend_slot(options, preferred_window=None, config=Config())
    assert sel.window == MORNING  # earliest, since nothing to cluster against


@pytest.mark.parametrize("neighbor_count", [1, 3, 5])
def test_neighbor_count_does_not_crash_on_small_routes(neighbor_count):
    # Fewer stops than the neighbor count must not raise.
    route = _route([MORNING, AFTERNOON], [_stop(29.7501, -95.3601, AFTERNOON)])
    cfg = Config(slot_neighbor_count=neighbor_count)
    options = identify_available_slots(PROSPECT, route, cfg)
    assert options[0].window == AFTERNOON
