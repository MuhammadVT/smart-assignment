"""
Builders for the route-slot tests. Two flavors:

  - `route()` / `customer()` build real domain objects for the *scoring* tests
    (openness, score_route_slot), which run through build_context.
  - `scored_eval()` builds a CandidateEvaluation with its `scored_slots`
    pre-populated, so the decide/evidence/verifier tests control the totals
    directly without depending on the scoring math.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import (
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_SLOT_AVAILABILITY,
    FACTOR_WINDOW_MATCH,
)
from smart_assignment.shared.models import (
    CandidateEvaluation,
    ConstraintOutcome,
    CustomerProfile,
    DayOfWeek,
    FactorScore,
    GeoPoint,
    PreferredSlot,
    Route,
    RouteStop,
    ScoredSlot,
    SlotOption,
)
from smart_assignment.shared.slot_selection import SLOT_BASIS_BETWEEN_STOPS

_CENTER = GeoPoint(29.75, -95.36)


def stop(lat, lon, window, tier=None) -> RouteStop:
    return RouteStop(
        customer_number="067-000000",
        location=GeoPoint(lat, lon),
        delivery_time_window=window,
        customer_tier=tier,
    )


def route(route_id="RTE-4100", name="Central", committed=None, day=DayOfWeek.TUE) -> Route:
    return Route(
        route_id=route_id,
        name=name,
        day=day,
        service_center=_CENTER,
        service_radius_miles=25.0,
        vehicle_capacity_cases=1000,
        avg_load_cases=100,
        committed_stops=committed or [],
    )


def customer(pref=None, cases=90, loc=_CENTER) -> CustomerProfile:
    slot_ = PreferredSlot(DayOfWeek.TUE, pref) if pref else None
    c = CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=cases,
        preferred_slot=slot_,
    )
    c.location = loc
    return c


def slot_option(window, overlap=0, anchor=None, basis=SLOT_BASIS_BETWEEN_STOPS) -> SlotOption:
    return SlotOption(
        window=window,
        fit_score=1.0,
        committed_overlap=overlap,
        basis=basis,
        anchor_time=anchor,
    )


def _factors(total_pieces) -> list[FactorScore]:
    """Build a plausible factor breakdown from (name, value, weight) triples."""
    return [FactorScore(name=n, weight=w, value=v, detail=f"{n}={v}") for (n, v, w) in total_pieces]


def scored_slot(window, avail, total, overlap=0, with_window=True) -> ScoredSlot:
    pieces = [
        (FACTOR_GEO_CLUSTERING, 0.8, 0.35),
        (FACTOR_CAPACITY_BUFFER, 1.0, 0.25),
    ]
    if with_window:
        pieces.append((FACTOR_WINDOW_MATCH, 0.7, 0.20))
    pieces.append((FACTOR_SLOT_AVAILABILITY, avail, 0.20))
    return ScoredSlot(
        slot=slot_option(window, overlap=overlap),
        factor_scores=_factors(pieces),
        total_score=total,
    )


def scored_eval(route_id, name, scored_slots, feasible=True) -> CandidateEvaluation:
    r = route(route_id=route_id, name=name)
    outcomes = [ConstraintOutcome(name="route_capacity", passed=feasible, detail="ok")]
    best = max(scored_slots, key=lambda s: s.total_score) if scored_slots else None
    return CandidateEvaluation(
        route=r,
        distance_miles=0.5,
        chosen_window=best.slot.window if best else None,
        remaining_capacity_after=400,
        utilization_after=0.6,
        constraint_outcomes=outcomes,
        factor_scores=best.factor_scores if best else [],
        total_score=best.total_score if best else 0.0,
        window_basis=best.slot.basis if best else "",
        available_slots=[s.slot for s in scored_slots],
        scored_slots=list(scored_slots),
    )


# Two convenience windows.
MORNING = (time(8, 30), time(11, 30))
AFTERNOON = (time(12, 30), time(15, 30))
