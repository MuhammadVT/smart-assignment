"""
Shared deterministic tools — reusable across any workflow, regardless of
its orchestration pattern (graph, sequential pipeline, multi-agent tree).

These are plain Python functions — no LLM involved. Constraint checking
(capacity, hours, temperature compatibility) is treated as pure code, not
agent reasoning, on purpose: a model should never be in a position to
"reason" a stop onto an already-full truck. Any workflow that needs this
logic should call these functions directly rather than reimplementing it.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.config import MAX_UTILIZATION_AFTER_ASSIGNMENT
from smart_assignment.shared.models import (
    CustomerProfile,
    FeasibleSlotOption,
    RouteSlot,
)

# ---------------------------------------------------------------------------
# Geocode + cluster a customer into a service zone
# ---------------------------------------------------------------------------


def geocode_and_cluster_customer(customer: CustomerProfile) -> dict:
    """
    [STUB] Resolve the customer's service zone for route matching.

    [ASSUMPTION] Real implementation should call Sysco's existing
    geocoding/territory service (likely already used for route planning)
    to assign a zone ID. Here we derive a crude zone from lat/lng buckets
    purely so the rest of the pipeline has something to match against.
    Replace `_naive_zone_bucket` with the real territory lookup.
    """
    zone_id = _naive_zone_bucket(customer.latitude, customer.longitude)
    return {
        "customer": customer,
        "zone_id": zone_id,
    }


def _naive_zone_bucket(lat: float, lng: float) -> str:
    """[STUB] Placeholder zone assignment — NOT production logic."""
    lat_bucket = int(lat * 10)
    lng_bucket = int(lng * 10)
    return f"ZONE_{lat_bucket}_{lng_bucket}"


# ---------------------------------------------------------------------------
# Hard constraint filtering (pure function, deterministic)
# ---------------------------------------------------------------------------


def filter_feasible_slots(
    customer: CustomerProfile,
    candidate_routes: list[RouteSlot],
) -> list[FeasibleSlotOption]:
    """
    Apply hard operational constraints. Anything that fails here is
    categorically infeasible — not a matter of LLM judgment.

    Hard constraints applied, in order:
      1. Vehicle temperature zone compatibility
      2. Remaining vehicle capacity >= customer's order volume
      3. Projected utilization stays within MAX_UTILIZATION_AFTER_ASSIGNMENT
         (see shared/config.py — [ASSUMPTION], confirm with ops)
      4. At least one open arrival window within driver shift hours
         (i.e. doesn't require unpaid/illegal overtime)
    """
    feasible: list[FeasibleSlotOption] = []

    for route in candidate_routes:
        if (
            route.vehicle_temp_zone != "mixed"
            and route.vehicle_temp_zone != customer.product_temp_zone
        ):
            continue  # hard fail: truck can't carry this product type

        committed_volume = sum(s.case_volume for s in route.committed_stops)
        remaining_capacity = route.vehicle_capacity_cases - committed_volume

        if remaining_capacity < customer.weekly_order_volume_cases:
            continue  # hard fail: not enough room on the truck

        projected_utilization = (
            committed_volume + customer.weekly_order_volume_cases
        ) / route.vehicle_capacity_cases
        if projected_utilization > MAX_UTILIZATION_AFTER_ASSIGNMENT:
            continue  # hard fail: would exceed safe capacity buffer

        if not route.available_arrival_windows:
            continue  # hard fail: no open windows at all

        for window in route.available_arrival_windows:
            if not _window_within_shift(window, route.driver_shift_start, route.driver_shift_end):
                continue  # hard fail: would require driver overtime

            feasible.append(
                FeasibleSlotOption(
                    route_slot=route,
                    proposed_arrival_window=window,
                    remaining_capacity_after_assignment=remaining_capacity
                    - customer.weekly_order_volume_cases,
                    geographic_fit_score=_geo_fit_score(route),
                    capacity_utilization_after=projected_utilization,
                    matches_customer_preference=_matches_preference(customer, route, window),
                )
            )

    return feasible


def _window_within_shift(window: tuple[time, time], shift_start: time, shift_end: time) -> bool:
    return window[0] >= shift_start and window[1] <= shift_end


def _geo_fit_score(route: RouteSlot) -> float:
    """
    [ASSUMPTION] Proxy for geographic clustering quality. Real version
    should compute actual stop-to-stop distance/drive-time delta versus
    inserting this stop into the existing route sequence (i.e. marginal
    cost of insertion). Here: more committed stops on the route = assumed
    tighter clustering, capped at 1.0.
    """
    return min(1.0, 0.3 + 0.15 * len(route.committed_stops))


def _matches_preference(
    customer: CustomerProfile, route: RouteSlot, window: tuple[time, time]
) -> bool:
    day_ok = (customer.requested_days is None) or (route.day in customer.requested_days)
    if customer.requested_time_window is None:
        time_ok = True
    else:
        pref_start, pref_end = customer.requested_time_window
        time_ok = window[0] >= pref_start and window[1] <= pref_end
    return day_ok and time_ok
