"""
Location- and time-aware delivery-slot selection.

The prospect should be delivered *when the truck is already in their
neighborhood*. We approximate that from the route's nearest committed stops:

  1. `identify_available_slots()` — take the prospect's geographically nearest
     committed stops ("adjacent stops"), group them into TEMPORAL clusters (the
     same neighbors can split into a morning group and an afternoon group), and
     emit one candidate slot per cluster. Each candidate window is a fixed
     length (`slot_window_minutes`) CENTERED on the cluster's
     inverse-distance-weighted midpoint time — a slot that literally sits
     *between* the adjacent stops, pulled toward the closer ones. No customer
     preference in the mix here.

  2. `select_candidate_slots()` — keep the top-N candidates per route by
     quality (fit + low contention), but ALWAYS keep any candidate that overlaps
     the customer's stated preference, even if it falls outside the top-N. This
     is the menu handed to the recommender (and, later, to an LLM).

  3. `recommend_slot()` — pick one from the menu with a soft blend of preference
     overlap, location fit, and low contention (not a hard "preference wins"
     gate). A route is never eliminated for a slot miss.

Phased fidelity: today a committed stop carries only its TW1 *permitted* window,
not a real planned-arrival time. `stop_reference_time` is the single seam that
turns a stop into a "when is the truck near here" clock value; phase A uses the
window midpoint, and a later phase can return a real planned-arrival ETA there —
and, with a stop *sequence*, the interpolation becomes true bracketing between
the two sequential stops the prospect is inserted between — without any caller
changing.

Deterministic Python throughout, so every slot pick stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import haversine_miles
from smart_assignment.shared.models import GeoPoint, Route, RouteStop, SlotOption, Window
from smart_assignment.shared.timeutils import overlap_minutes, window_midpoint

# Why a given slot exists / won -- surfaced on the trace and the tool JSON.
SLOT_BASIS_BETWEEN_STOPS = "between_adjacent_stops"
SLOT_BASIS_LEAST_CONTENDED = "least_contended"
SLOT_BASIS_PREFERENCE = "preference_accommodated"
SLOT_BASIS_NONE = "no_windows"

# Guards a divide-by-zero when a committed stop sits exactly on the prospect.
_DISTANCE_EPS = 1e-6
_DAY_MAX = 24 * 60 - 1  # 23:59 in minutes


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


def _time_from_minutes(m: float) -> time:
    """Clamp minutes-since-midnight into a valid clock time (max 23:59)."""
    m = int(round(m))
    m = max(0, min(m, _DAY_MAX))
    return time(m // 60, m % 60)


def centered_window(anchor_minutes: float, minutes: int) -> Window:
    """A window of ``minutes`` length CENTERED on ``anchor_minutes``, clamped so
    it never runs before 00:00 or past end of day (shifted, keeping its length)."""
    start = anchor_minutes - minutes / 2
    end = anchor_minutes + minutes / 2
    if start < 0:
        start, end = 0, minutes
    if end > _DAY_MAX:
        end, start = _DAY_MAX, _DAY_MAX - minutes
    return (_time_from_minutes(start), _time_from_minutes(end))


def stop_reference_time(stop: RouteStop) -> Optional[time]:
    """
    Coarse proxy for "when the truck is at this stop" -- THE phase A/B seam.

    Phase A (today): the midpoint of the stop's TW1 permitted window. A stop
    with no window can't anchor a slot, so this returns None.

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


def _committed_overlap_count(window: Window, stops: list[RouteStop]) -> int:
    """How many committed stops' windows overlap `window` -- its contention."""
    return sum(
        1
        for s in stops
        if s.delivery_time_window is not None
        and overlap_minutes(window, s.delivery_time_window) > 0
    )


def _cluster_by_time(
    timed: list[tuple[Neighbor, int]], gap_minutes: int
) -> list[list[tuple[Neighbor, int]]]:
    """Group (neighbor, reference-minutes) pairs into temporal clusters: sorted
    by time, a gap larger than `gap_minutes` starts a new cluster."""
    clusters: list[list[tuple[Neighbor, int]]] = []
    current: list[tuple[Neighbor, int]] = []
    for item in sorted(timed, key=lambda p: p[1]):
        if current and item[1] - current[-1][1] > gap_minutes:
            clusters.append(current)
            current = []
        current.append(item)
    if current:
        clusters.append(current)
    return clusters


def _weight(distance_miles: float) -> float:
    return 1.0 / (distance_miles + _DISTANCE_EPS)


def _contention_score(committed_overlap: int) -> float:
    """Emptier is better: a smooth 0-1 signal from the raw contention count."""
    return 1.0 / (1.0 + committed_overlap)


def _quality(option: SlotOption, config: Config) -> float:
    """Preference-free candidate quality: blend of location fit and low
    contention, normalized over the two active weights."""
    wf, wc = config.slot_weight_fit, config.slot_weight_contention
    total = (wf + wc) or 1.0
    return (wf * option.fit_score + wc * _contention_score(option.committed_overlap)) / total


def identify_available_slots(
    customer_location: GeoPoint,
    route: Route,
    config: Config,
) -> list[SlotOption]:
    """
    STEP 1 -- the route's candidate slots for this prospect, ranked best-first,
    with no customer preference in the mix.

    Primary path: cluster the nearest committed stops by time and emit one
    candidate per cluster, centered on the cluster's inverse-distance-weighted
    midpoint (the "slot between adjacent stops"), with a `fit_score` equal to
    that cluster's share of the total proximity weight.

    Fallback (no nearby stop carries a time -- e.g. an empty or window-less
    route): offer each of the route's own windows, centered on its midpoint,
    with fit 0 and basis `least_contended`.
    """
    neighbors = nearest_neighbors(
        customer_location,
        route.committed_stops,
        config.slot_neighbor_count,
        config.slot_neighbor_max_miles,
    )
    timed = [
        (n, _minutes(ref))
        for n in neighbors
        if (ref := stop_reference_time(n.stop)) is not None
    ]

    options: list[SlotOption] = []
    if timed:
        total_weight = sum(_weight(n.distance_miles) for n, _ in timed)
        for cluster in _cluster_by_time(timed, config.slot_cluster_gap_minutes):
            w_sum = sum(_weight(n.distance_miles) for n, _ in cluster)
            anchor = sum(_weight(n.distance_miles) * m for n, m in cluster) / w_sum
            window = centered_window(anchor, config.slot_window_minutes)
            options.append(
                SlotOption(
                    window=window,
                    fit_score=w_sum / total_weight,
                    committed_overlap=_committed_overlap_count(window, route.committed_stops),
                    basis=SLOT_BASIS_BETWEEN_STOPS,
                    anchor_time=_time_from_minutes(anchor),
                )
            )
    else:
        for w in dict.fromkeys(route.available_windows):
            anchor = _minutes(window_midpoint(w))
            window = centered_window(anchor, config.slot_window_minutes)
            options.append(
                SlotOption(
                    window=window,
                    fit_score=0.0,
                    committed_overlap=_committed_overlap_count(window, route.committed_stops),
                    basis=SLOT_BASIS_LEAST_CONTENDED,
                    anchor_time=_time_from_minutes(anchor),
                )
            )

    options.sort(key=lambda o: (-_quality(o, config), _minutes(o.window[0])))
    return options


def select_candidate_slots(
    options: list[SlotOption],
    preferred_window: Optional[Window],
    config: Config,
) -> list[SlotOption]:
    """
    STEP 2 -- the top-N candidate slots kept per route (the menu).

    Keep the best `slot_candidate_count` by quality, but ALWAYS include any
    candidate that overlaps the customer's stated preference, even if it ranks
    below the cut -- so a preferred-time option is never dropped from the menu.
    Assumes `options` is already quality-ranked (as identify_available_slots
    returns it).
    """
    if not options:
        return []
    n = max(1, config.slot_candidate_count)
    kept = list(options[:n])
    if preferred_window is not None:
        for o in options[n:]:
            if overlap_minutes(preferred_window, o.window) > 0:
                kept.append(o)
    return kept


def _blended_score(option: SlotOption, preferred_window: Optional[Window], config: Config) -> float:
    """Soft blend used to pick the single recommended slot: preference overlap
    (when stated) + location fit + low contention, normalized over active terms."""
    wf, wc = config.slot_weight_fit, config.slot_weight_contention
    fit_term = wf * option.fit_score + wc * _contention_score(option.committed_overlap)
    if preferred_window is None:
        return fit_term / ((wf + wc) or 1.0)
    wp = config.slot_weight_preference
    window_len = max(1, config.slot_window_minutes)
    pref_frac = overlap_minutes(preferred_window, option.window) / window_len
    return (wp * pref_frac + fit_term) / ((wp + wf + wc) or 1.0)


def blended_slot_score(
    option: SlotOption, preferred_window: Optional[Window], config: Config
) -> float:
    """Public view of the deterministic blend score for one candidate slot -- the
    exact number `recommend_slot` maximizes. Surfaced as *reference* evidence to
    the grounded slot picker (the LLM may agree with or diverge from it), while
    remaining the deterministic fallback pick."""
    return _blended_score(option, preferred_window, config)


def recommend_slot(
    options: list[SlotOption],
    preferred_window: Optional[Window],
    config: Config,
) -> SlotSelection:
    """
    STEP 3 -- pick one slot from the menu, blending preference with fit and
    contention (a soft weighting, not a hard preference gate). Windows are
    already fixed-length and centered, so the pick is returned as-is. A route is
    never eliminated for a slot miss; `overlap_minutes` is the chosen window's
    overlap with the preference (0 when there is none), which feeds `window_match`.
    """
    if not options:
        return SlotSelection(None, SLOT_BASIS_NONE, 0)

    chosen = max(options, key=lambda o: (_blended_score(o, preferred_window, config), o.fit_score))
    overlap = overlap_minutes(preferred_window, chosen.window) if preferred_window else 0
    basis = SLOT_BASIS_PREFERENCE if (preferred_window and overlap > 0) else chosen.basis
    return SlotSelection(chosen.window, basis, overlap)
