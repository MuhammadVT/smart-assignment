"""
Hard-constraint checking (spec step 3).

Design intent — *modularity*:
  - Every hard constraint is a small pure function with a uniform signature,
    registered in `HARD_CONSTRAINTS`. Adding, removing, or reordering a
    constraint is a one-line change and requires touching nothing else.
  - Constraint logic is deterministic Python, never LLM reasoning: a model
    should not be in a position to "reason" a customer onto a full truck or
    into an unserviceable area. These are objectively checkable facts.

`EvalContext` pre-computes the shared geo/capacity/window facts once per
(customer, route) so individual constraints and the scorer don't recompute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import haversine_miles
from smart_assignment.shared.models import (
    ConstraintOutcome,
    CustomerProfile,
    Route,
    Window,
)
from smart_assignment.shared.timeutils import best_overlapping_window


@dataclass
class EvalContext:
    """Facts about one (customer, route) pair, computed once and reused."""

    distance_miles: float  # customer -> route service center
    avg_stop_distance_miles: float  # customer -> route's committed stops (clustering)
    committed_volume: int
    remaining_capacity_after: int
    utilization_after: float
    best_window: Optional[Window]
    window_overlap_minutes: int


def build_context(customer: CustomerProfile, route: Route) -> EvalContext:
    assert customer.location is not None, "customer must be geocoded before evaluation"

    distance = haversine_miles(customer.location, route.service_center)

    if route.committed_stops:
        stop_dists = [haversine_miles(customer.location, s.location) for s in route.committed_stops]
        avg_stop_distance = sum(stop_dists) / len(stop_dists)
    else:
        avg_stop_distance = distance  # nothing to cluster against yet

    committed = route.committed_volume_cases
    remaining_after = route.vehicle_capacity_cases - committed - customer.order_quantity_cases
    utilization_after = (committed + customer.order_quantity_cases) / route.vehicle_capacity_cases

    best_window, overlap = best_overlapping_window(
        customer.preferred_window, route.available_windows
    )

    return EvalContext(
        distance_miles=distance,
        avg_stop_distance_miles=avg_stop_distance,
        committed_volume=committed,
        remaining_capacity_after=remaining_after,
        utilization_after=utilization_after,
        best_window=best_window,
        window_overlap_minutes=overlap,
    )


# --- Individual hard constraints -------------------------------------------

ConstraintFn = Callable[[CustomerProfile, Route, EvalContext, Config], ConstraintOutcome]


def geographic_serviceability(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> ConstraintOutcome:
    limit = min(route.service_radius_miles, config.max_service_distance_miles)
    passed = ctx.distance_miles <= limit
    return ConstraintOutcome(
        name="geographic_serviceability",
        passed=passed,
        detail=(f"{ctx.distance_miles:.1f} mi from route center " f"(limit {limit:.1f} mi)"),
    )


def route_capacity(
    customer: CustomerProfile, route: Route, ctx: EvalContext, config: Config
) -> ConstraintOutcome:
    passed = (
        ctx.remaining_capacity_after >= 0
        and ctx.utilization_after <= config.max_utilization_after_assignment
    )
    return ConstraintOutcome(
        name="route_capacity",
        passed=passed,
        detail=(
            f"utilization {ctx.utilization_after:.0%} post-add "
            f"(limit {config.max_utilization_after_assignment:.0%}), "
            f"{ctx.remaining_capacity_after} cases headroom"
        ),
    )


# NOTE: the customer's preferred delivery window is intentionally NOT a hard
# constraint. It is a *soft* preference handled by the `window_match` scoring
# factor (see shared/scoring.py) — it influences ranking but never eliminates a
# route. `EvalContext` still pre-computes the window overlap for that scorer.


# Registry — the ordered set of hard constraints applied to every candidate.
HARD_CONSTRAINTS: list[ConstraintFn] = [
    geographic_serviceability,
    route_capacity,
]


def evaluate_constraints(
    customer: CustomerProfile,
    route: Route,
    ctx: EvalContext,
    config: Config,
    constraints: Optional[list[ConstraintFn]] = None,
) -> list[ConstraintOutcome]:
    checks = constraints if constraints is not None else HARD_CONSTRAINTS
    return [check(customer, route, ctx, config) for check in checks]
