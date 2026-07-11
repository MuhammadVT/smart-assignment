"""
Location-aware delivery-slot selection.

Two deterministic steps, deliberately kept separate:

  1. `identify_available_slots()` — given a prospect's location and a route,
     enumerate EVERY window the route offers, each annotated with how well it
     fits the prospect relative to the route's nearest committed stops ("a slot
     between adjacent stops") and how contended it already is. This is pure
     route + geography; it never looks at the customer's stated preference.

  2. `recommend_slot()` — THEN consider the customer's preference: accommodate
     it when an offered window can, otherwise fall back to the best slot from
     the menu — the between-stops pick when location fit exists, or the
     least-contended window when it doesn't. A route is never eliminated for a
     window miss (the window is a soft factor), so this always returns a slot
     whenever the route offers one.

Phased fidelity: today a committed stop carries only its TW1 *permitted*
window, not a real planned-arrival time. `stop_reference_time` is the single
seam that turns a stop into a "when is the truck near here" clock value; phase
A uses the window midpoint, and a later phase can return a real arrival ETA
there without any caller changing.

Like the rest of the constraint/scoring layer (see constraints.py), this is
deterministic Python, never LLM reasoning, so every slot pick stays auditable
and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import haversine_miles
from smart_assignment.shared.models import GeoPoint, Route, RouteStop, SlotOption, Window
from smart_assignment.shared.timeutils import (
    duration_minutes,
    overlap_minutes,
    window_midpoint,
)

# Why a given slot won -- surfaced on the evaluation trace and the tool JSON.
SLOT_BASIS_BETWEEN_STOPS = "between_adjacent_stops"
SLOT_BASIS_LEAST_CONTENDED = "least_contended"
SLOT_BASIS_PREFERENCE = "preference_accommodated"
SLOT_BASIS_NONE = "no_windows"

# Guards a divide-by-zero when a committed stop sits exactly on the prospect.
_DISTANCE_EPS = 1e-6


@dataclass(frozen=True)
class Neighbor:
    """A committed stop and its distance from the prospect."""

    stop: RouteStop
    distance_miles: float


@dataclass(frozen=True)
class SlotSelection:
    """The single recommended window, why it won, and its overlap (minutes)
    with the customer's preferred window (0 when there is no preference)."""

    window: Optional[Window]
    basis: str
    overlap_minutes: int


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def stop_reference_time(stop: RouteStop) -> Optional[time]:
    """
    Coarse proxy for "when the truck is at this stop" -- THE phase A/B seam.

    Phase A (today): the midpoint of the stop's TW1 permitted window. A stop
    with no window can't vote, so this returns None.

    Phase B (later): return a real planned-arrival ETA once `RouteStop` carries
    one, falling back to the midpoint. No caller changes.
    """
    if stop.delivery_time_window is None:
        return None
    return window_midpoint(stop.delivery_time_window)


def nearest_neighbors(
    location: GeoPoint,
    stops: list[RouteStop],
    k: int,
    max_miles: Optional[float] = None,
) -> list[Neighbor]:
    """
    The `k` committed stops closest to `location`, nearest first. Stops farther
    than `max_miles` (when set) are dropped before the cut. A non-positive `k`
    yields an empty list.
    """
    if k <= 0:
        return []
    neighbors = [Neighbor(s, haversine_miles(location, s.location)) for s in stops]
    if max_miles is not None:
        neighbors = [n for n in neighbors if n.distance_miles <= max_miles]
    neighbors.sort(key=lambda n: n.distance_miles)
    return neighbors[:k]


def _window_for_reference_time(ref: time, available: list[Window]) -> Optional[Window]:
    """
    Map one stop's reference time onto the offered window it belongs to: the
    tightest window that contains it, else the nearest window by time gap
    (earliest-start breaks ties). None only when there are no windows.
    """
    if not available:
        return None
    ref_m = _minutes(ref)
    containing = [w for w in available if _minutes(w[0]) <= ref_m <= _minutes(w[1])]
    if containing:
        return min(containing, key=duration_minutes)

    def _gap(w: Window) -> int:
        return min(abs(ref_m - _minutes(w[0])), abs(ref_m - _minutes(w[1])))

    return min(available, key=lambda w: (_gap(w), _minutes(w[0])))


def _committed_overlap_count(window: Window, stops: list[RouteStop]) -> int:
    """
    How many committed stops' windows overlap `window` -- the contention of
    that slot. Fewer overlaps = emptier slot = the least-contended fallback.
    """
    return sum(
        1
        for s in stops
        if s.delivery_time_window is not None
        and overlap_minutes(window, s.delivery_time_window) > 0
    )


def identify_available_slots(
    customer_location: GeoPoint,
    route: Route,
    config: Config,
) -> list[SlotOption]:
    """
    STEP 1 -- enumerate every window the route offers, annotated with location
    fit and contention, ranked best-first. No preference input.

    Fit is an inverse-distance-weighted vote: each of the route's nearest
    committed stops backs the one offered window its reference time maps to,
    weighted by how close it is to the prospect, then normalized to 0-1 across
    all the votes. A window no nearby stop maps to gets fit 0 but is still
    included in the menu.

    Ordering: highest location fit first ("between adjacent stops"), then the
    least-contended window (fewest committed stops in it), then earliest start.
    """
    # Distinct windows only; guards against any accidental duplicates upstream.
    available = list(dict.fromkeys(route.available_windows))
    if not available:
        return []

    neighbors = nearest_neighbors(
        customer_location,
        route.committed_stops,
        config.slot_neighbor_count,
        config.slot_neighbor_max_miles,
    )

    raw_votes: dict[Window, float] = {}
    for n in neighbors:
        ref = stop_reference_time(n.stop)
        if ref is None:
            continue
        w = _window_for_reference_time(ref, available)
        if w is None:
            continue
        raw_votes[w] = raw_votes.get(w, 0.0) + 1.0 / (n.distance_miles + _DISTANCE_EPS)
    total_weight = sum(raw_votes.values())

    options: list[SlotOption] = []
    for w in available:
        fit = (raw_votes.get(w, 0.0) / total_weight) if total_weight > 0 else 0.0
        options.append(
            SlotOption(
                window=w,
                fit_score=fit,
                committed_overlap=_committed_overlap_count(w, route.committed_stops),
                basis=SLOT_BASIS_BETWEEN_STOPS if fit > 0 else SLOT_BASIS_LEAST_CONTENDED,
            )
        )

    options.sort(key=lambda o: (-o.fit_score, o.committed_overlap, _minutes(o.window[0])))
    return options


def recommend_slot(
    options: list[SlotOption],
    preferred_window: Optional[Window],
    config: Config,
) -> SlotSelection:
    """
    STEP 2 -- pick the final window, NOW considering the customer's preference.

    - No options (the route offers no windows): ``(None, no_windows, 0)``.
    - Preference accommodated: among options whose window overlaps the
      preferred window, take the greatest overlap (tie-break: better location
      fit, then earlier start). Basis ``preference_accommodated``.
    - Otherwise (no preference, or none of the options overlaps it): take the
      top of the already-ranked menu -- the between-stops pick when location
      fit exists, the least-contended window when it doesn't. The route still
      gets a slot; `overlap_minutes` is the (possibly 0) overlap with the
      preference.

    `overlap_minutes` is always the route's *best achievable* overlap with the
    preferred window, so the `window_match` scorer keeps its original meaning.
    """
    if not options:
        return SlotSelection(None, SLOT_BASIS_NONE, 0)

    if preferred_window is not None:
        overlapping = [(o, overlap_minutes(preferred_window, o.window)) for o in options]
        overlapping = [(o, ov) for (o, ov) in overlapping if ov > 0]
        if overlapping:
            best, best_overlap = max(
                overlapping,
                key=lambda pair: (pair[1], pair[0].fit_score, -_minutes(pair[0].window[0])),
            )
            return SlotSelection(best.window, SLOT_BASIS_PREFERENCE, best_overlap)

    chosen = options[0]
    overlap = overlap_minutes(preferred_window, chosen.window) if preferred_window else 0
    return SlotSelection(chosen.window, chosen.basis, overlap)
