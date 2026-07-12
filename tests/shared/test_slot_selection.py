"""
Unit tests for the location- and time-aware delivery-slot selector
(shared/slot_selection.py). Deterministic, fast, no LLM or network.

Three steps are exercised:
  * identify_available_slots() — cluster the nearest committed stops by time and
    emit one candidate per cluster, centered on the proximity-weighted midpoint.
  * select_candidate_slots() — the top-N menu, always keeping any candidate that
    overlaps a stated preference.
  * recommend_slot() — the single pick, blending preference with fit/contention.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import DayOfWeek, GeoPoint, Route, RouteStop
from smart_assignment.shared.slot_selection import (
    SLOT_BASIS_BETWEEN_STOPS,
    SLOT_BASIS_LEAST_CONTENDED,
    SLOT_BASIS_NONE,
    SLOT_BASIS_PREFERENCE,
    centered_window,
    identify_available_slots,
    nearest_neighbors,
    recommend_slot,
    select_candidate_slots,
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


def _mins(t: time) -> int:
    return t.hour * 60 + t.minute


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
    assert [n.stop for n in got] == [near, mid]
    assert got[0].distance_miles < got[1].distance_miles


def test_nearest_neighbors_respects_max_miles_and_nonpositive_k():
    near = _stop(29.7501, -95.3601, MORNING)
    far = _stop(29.9000, -95.5000, MORNING)
    capped = nearest_neighbors(PROSPECT, [near, far], k=5, max_miles=1.0)
    assert [n.stop for n in capped] == [near]
    assert nearest_neighbors(PROSPECT, [near, far], k=0) == []


# --- centered_window --------------------------------------------------------


def test_centered_window_centres_on_the_anchor():
    assert centered_window(9 * 60, 180) == (time(7, 30), time(10, 30))
    assert centered_window(12 * 60, 120) == (time(11, 0), time(13, 0))


def test_centered_window_clamps_at_start_of_day():
    # Anchor 01:00 with a 3h window would start before midnight -> shifted.
    assert centered_window(60, 180) == (time(0, 0), time(3, 0))


def test_centered_window_clamps_at_end_of_day():
    start, end = centered_window(23 * 60 + 30, 180)
    assert end == time(23, 59)
    assert start == time(20, 59)  # kept its full length


# --- identify_available_slots: clustered, centered candidates ---------------


def test_between_two_adjacent_stops_centres_the_slot():
    # Two equidistant stops (symmetric around the prospect) with windows close
    # in time -> ONE cluster, centred on the midpoint between their refs.
    early = _stop(29.7505, -95.3600, (time(8, 0), time(9, 30)))   # ref 08:45
    late = _stop(29.7495, -95.3600, (time(9, 30), time(11, 0)))   # ref 10:15
    options = identify_available_slots(PROSPECT, _route([], [early, late]), Config())
    assert len(options) == 1
    o = options[0]
    assert o.basis == SLOT_BASIS_BETWEEN_STOPS
    assert o.anchor_time == time(9, 30)         # midpoint of 08:45 and 10:15
    assert o.window == (time(8, 0), time(11, 0))  # 3h centred on 09:30
    assert o.fit_score == 1.0                   # single cluster


def test_anchor_is_pulled_toward_the_closer_stop():
    # Same times, but the earlier stop is much closer -> the weighted midpoint
    # pulls earlier than the naive midpoint (09:30).
    near_early = _stop(29.75005, -95.36000, (time(7, 30), time(8, 30)))  # ref 08:00, ~0.003 mi
    far_late = _stop(29.7700, -95.3600, (time(10, 30), time(11, 30)))    # ref 11:00, ~1.4 mi
    options = identify_available_slots(PROSPECT, _route([], [near_early, far_late]), Config())
    assert len(options) == 1
    assert _mins(options[0].anchor_time) < _mins(time(9, 30))  # pulled toward 08:00


def test_neighbors_split_into_morning_and_afternoon_candidates():
    stops = [
        _stop(29.7505, -95.3600, (time(8, 0), time(9, 0))),    # morning
        _stop(29.7503, -95.3600, (time(8, 30), time(9, 30))),  # morning
        _stop(29.7498, -95.3600, (time(13, 30), time(14, 30))),  # afternoon
    ]
    options = identify_available_slots(PROSPECT, _route([], stops), Config())
    assert len(options) == 2
    anchors = sorted(_mins(o.anchor_time) for o in options)
    assert anchors[0] < 12 * 60 < anchors[1]  # one morning, one afternoon
    assert all(o.basis == SLOT_BASIS_BETWEEN_STOPS for o in options)


def test_fallback_to_route_windows_when_no_stop_carries_a_time():
    # No committed stop has a window -> fall back to the route's own windows,
    # centred on their midpoints, basis least_contended, fit 0.
    route = _route([MORNING, AFTERNOON], [_stop(29.7501, -95.3601, None)])
    options = identify_available_slots(PROSPECT, route, Config())
    assert {o.basis for o in options} == {SLOT_BASIS_LEAST_CONTENDED}
    assert all(o.fit_score == 0.0 for o in options)
    # centred on each window's midpoint (08:30 and 14:00), 3h long
    windows = {o.window for o in options}
    assert (time(7, 0), time(10, 0)) in windows     # 08:30 ± 90
    assert (time(12, 30), time(15, 30)) in windows  # 14:00 ± 90


def test_empty_route_and_no_timed_stops_yields_empty_menu():
    assert identify_available_slots(PROSPECT, _route([], []), Config()) == []


def test_contention_counts_overlapping_committed_stops():
    # A morning cluster's candidate window overlaps the two morning stops.
    stops = [
        _stop(29.7505, -95.3600, (time(8, 0), time(9, 0))),
        _stop(29.7503, -95.3600, (time(8, 30), time(9, 30))),
    ]
    options = identify_available_slots(PROSPECT, _route([], stops), Config())
    assert options[0].committed_overlap == 2


# --- select_candidate_slots: top-N + preference guarantee -------------------


def _four_cluster_route():
    # Four temporally-separated single-stop clusters, increasingly far away, so
    # the latest (17:30) is also the lowest-fit / lowest-ranked candidate.
    return _route([], [
        _stop(29.7505, -95.3600, (time(7, 0), time(8, 0))),      # near,   ref 07:30
        _stop(29.7520, -95.3600, (time(10, 30), time(11, 30))),  # mid,    ref 11:00
        _stop(29.7560, -95.3600, (time(14, 0), time(15, 0))),    # far,    ref 14:30
        _stop(29.7700, -95.3600, (time(17, 30), time(18, 30))),  # farthest, ref 18:00
    ])


def test_top_n_caps_the_menu_without_a_preference():
    cfg = Config(slot_neighbor_count=4)  # consider all four stops
    options = identify_available_slots(PROSPECT, _four_cluster_route(), cfg)
    assert len(options) == 4
    menu = select_candidate_slots(options, preferred_window=None, config=cfg)
    assert len(menu) == 3  # default slot_candidate_count
    # the farthest/lowest-fit (18:00) candidate is dropped
    assert all(o.anchor_time != time(18, 0) for o in menu)


def test_preference_overlapping_candidate_is_always_kept():
    cfg = Config(slot_neighbor_count=4)
    options = identify_available_slots(PROSPECT, _four_cluster_route(), cfg)
    # A late preference overlaps only the 4th (dropped-by-quality) candidate.
    pref = (time(17, 0), time(19, 0))
    menu = select_candidate_slots(options, preferred_window=pref, config=cfg)
    assert any(o.anchor_time == time(18, 0) for o in menu)  # kept despite the cap


def test_candidate_count_is_configurable():
    cfg = Config(slot_neighbor_count=4, slot_candidate_count=2)
    options = identify_available_slots(PROSPECT, _four_cluster_route(), cfg)
    menu = select_candidate_slots(options, None, cfg)
    assert len(menu) == 2


# --- recommend_slot: the final pick -----------------------------------------


def test_recommend_no_preference_takes_the_best_quality_candidate():
    stops = [
        _stop(29.75005, -95.36000, AFTERNOON),  # very close -> afternoon dominates
        _stop(29.9000, -95.5000, MORNING),      # far
    ]
    options = identify_available_slots(PROSPECT, _route([], stops), Config())
    menu = select_candidate_slots(options, None, Config())
    sel = recommend_slot(menu, preferred_window=None, config=Config())
    assert _mins(time(12, 0)) < _mins(sel.window[0]) or sel.window[0] >= time(12, 0)
    assert sel.overlap_minutes == 0
    assert sel.basis == SLOT_BASIS_BETWEEN_STOPS


def test_recommend_blends_toward_a_preference_overlapping_candidate():
    stops = [
        _stop(29.7505, -95.3600, (time(8, 0), time(9, 0))),      # morning cluster
        _stop(29.7503, -95.3600, (time(8, 30), time(9, 30))),
        _stop(29.7498, -95.3600, (time(13, 30), time(14, 30))),  # afternoon
    ]
    options = identify_available_slots(PROSPECT, _route([], stops), Config())
    menu = select_candidate_slots(options, (time(13, 0), time(15, 30)), Config())
    sel = recommend_slot(menu, preferred_window=(time(13, 0), time(15, 30)), config=Config())
    assert sel.basis == SLOT_BASIS_PREFERENCE
    assert sel.overlap_minutes > 0
    assert sel.window[0] >= time(12, 0)  # the afternoon candidate


def test_recommend_still_returns_a_slot_when_preference_cannot_be_met():
    # Afternoon-only neighbourhood; a morning preference can't be honored, but
    # the route is NOT dropped -- it still gets its afternoon slot, overlap 0.
    stops = [_stop(29.7501, -95.3601, AFTERNOON)]
    options = identify_available_slots(PROSPECT, _route([], stops), Config())
    menu = select_candidate_slots(options, MORNING, Config())
    sel = recommend_slot(menu, preferred_window=MORNING, config=Config())
    assert sel.window is not None
    assert sel.overlap_minutes == 0
    assert sel.basis == SLOT_BASIS_BETWEEN_STOPS


def test_recommend_on_empty_menu_returns_no_window():
    sel = recommend_slot([], preferred_window=MORNING, config=Config())
    assert sel.window is None
    assert sel.basis == SLOT_BASIS_NONE
    assert sel.overlap_minutes == 0


def test_slot_window_minutes_is_configurable():
    stops = [_stop(29.7501, -95.3601, AFTERNOON)]  # ref 14:00
    options = identify_available_slots(PROSPECT, _route([], stops), Config(slot_window_minutes=240))
    assert options[0].window == (time(12, 0), time(16, 0))  # 4h centred on 14:00
