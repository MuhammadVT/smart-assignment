"""
Domain models for the delivery slot recommendation workflow.

[ASSUMPTION BLOCK — READ FIRST]
None of Sysco's actual data schemas, capacity rules, or route systems were
provided. Every field below is a reasonable guess at what a route/capacity
system would expose, modeled loosely on common DSD (direct store delivery)
/ foodservice distribution patterns. Treat this file as the contract to
correct once real schemas (e.g., from a TMS like Roadnet, Descartes, or an
internal system) are available. Swapping these models and the two tool
functions in `tools.py` that populate them should be the only changes
needed to point this workflow at real systems — the graph and the
reasoning agent should not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Optional


class DayOfWeek(str, Enum):
    MON = "MON"
    TUE = "TUE"
    WED = "WED"
    THU = "THU"
    FRI = "FRI"
    SAT = "SAT"


@dataclass
class CustomerProfile:
    """
    [ASSUMPTION] Minimal new-customer intake fields. In reality this likely
    comes from a CRM / onboarding form. `weekly_order_volume_cases` and
    `product_temp_zone` materially affect which routes/vehicles can serve
    the account, so they're treated as required.
    """

    customer_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    weekly_order_volume_cases: int  # estimated case volume per delivery
    product_temp_zone: str  # [ASSUMPTION] "dry" | "refrigerated" | "frozen" | "mixed"
    requested_days: Optional[list[DayOfWeek]] = None  # customer preference, soft constraint
    requested_time_window: Optional[tuple[time, time]] = (
        None  # customer preference, soft constraint
    )
    delivery_priority: str = "standard"  # [ASSUMPTION] "standard" | "high" (e.g. contractual SLA)


@dataclass
class CommittedStop:
    """An existing customer stop already locked into a route's schedule."""

    customer_id: str
    arrival_window: tuple[time, time]
    case_volume: int


@dataclass
class RouteSlot:
    """
    [ASSUMPTION] Represents one (route, day) instance — i.e. a specific
    truck running a specific day of the week — with its capacity and
    already-committed stops. A "route" in this model is recurring
    (same route number runs every Tue, for example) but capacity and
    committed stops are evaluated per day-instance since they vary daily.
    """

    route_id: str
    day: DayOfWeek
    vehicle_id: str
    vehicle_capacity_cases: int
    vehicle_temp_zone: str  # must match customer's product_temp_zone (or be "mixed")
    driver_id: str
    driver_shift_start: time
    driver_shift_end: time
    service_zone_ids: list[str]  # geographic zones this route is authorized/optimized to serve
    committed_stops: list[CommittedStop] = field(default_factory=list)
    available_arrival_windows: list[tuple[time, time]] = field(default_factory=list)
    # ^ [ASSUMPTION] pre-computed open windows within the shift not yet
    #   claimed by committed stops, e.g. from a routing engine's slack report.


@dataclass
class FeasibleSlotOption:
    """A RouteSlot that has passed all hard constraints, with computed fit metrics."""

    route_slot: RouteSlot
    proposed_arrival_window: tuple[time, time]
    remaining_capacity_after_assignment: int
    geographic_fit_score: (
        float  # 0-1, [ASSUMPTION] proximity/clustering with existing committed stops on this route
    )
    capacity_utilization_after: float  # 0-1
    matches_customer_preference: bool


@dataclass
class SlotRecommendation:
    """Final explainable output of the workflow."""

    customer_id: str
    recommended_route_id: Optional[str]
    recommended_day: Optional[str]
    recommended_window: Optional[str]
    confidence: float  # 0-1, model-reported confidence in this recommendation
    reasoning: str
    rejected_alternatives: list[str]  # human-readable reasons other slots were ruled out
    requires_human_review: bool
    review_reason: Optional[str] = None
