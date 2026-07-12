"""Builders for the slotpick tests -- construct evaluations with a controlled
candidate menu directly, no pipeline/geocoder needed."""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    DayOfWeek,
    GeoPoint,
    PreferredSlot,
    Route,
    SlotOption,
)
from smart_assignment.shared.slot_selection import SLOT_BASIS_BETWEEN_STOPS


def slot(window, fit, overlap, anchor, basis=SLOT_BASIS_BETWEEN_STOPS) -> SlotOption:
    return SlotOption(
        window=window,
        fit_score=fit,
        committed_overlap=overlap,
        basis=basis,
        anchor_time=anchor,
    )


def _route() -> Route:
    return Route(
        route_id="RTE-4100",
        name="Central Houston",
        day=DayOfWeek.TUE,
        service_center=GeoPoint(29.75, -95.36),
        vehicle_capacity_cases=1000,
        avg_load_cases=100,
    )


def evaluation(slots, chosen_index=0) -> CandidateEvaluation:
    chosen = slots[chosen_index]
    return CandidateEvaluation(
        route=_route(),
        distance_miles=0.1,
        chosen_window=chosen.window,
        remaining_capacity_after=300,
        utilization_after=0.7,
        window_basis=chosen.basis,
        available_slots=list(slots),
        total_score=0.9,
    )


def customer(pref=None) -> CustomerProfile:
    slot_ = PreferredSlot(DayOfWeek.TUE, pref) if pref else None
    return CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_slot=slot_,
    )


# A two-candidate menu: a morning slot and an afternoon slot.
MORNING = slot((time(7, 20), time(10, 20)), fit=0.7, overlap=2, anchor=time(8, 50))
AFTERNOON = slot((time(13, 0), time(16, 0)), fit=0.3, overlap=1, anchor=time(14, 30))
