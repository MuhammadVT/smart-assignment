"""
Domain models for the Smart Assignment slot-recommendation workflow.

These are intentionally small, framework-agnostic dataclasses so that any
orchestration layer (the plain-Python pipeline in
`workflows/slot_recommendation/pipeline.py`, the ADK graph, a future
sequential/multi-agent workflow) shares the exact same data contracts.

[MOCK / ASSUMPTION]
None of Sysco's real schemas were provided. Field choices below model a
foodservice DSD (direct-store-delivery) domain: routes are trucks that run
a given weekday delivering *cases* to accounts (restaurants, cafeterias,
etc.). Swap `integrations/route_capacity_client.py` and
`integrations/geocoding_client.py` for real systems and — as long as they
populate these dataclasses — nothing downstream needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import Optional

# A delivery window is a simple (start, end) pair of clock times.
Window = tuple[time, time]


class DayOfWeek(str, Enum):
    MON = "MON"
    TUE = "TUE"
    WED = "WED"
    THU = "THU"
    FRI = "FRI"
    SAT = "SAT"


@dataclass(frozen=True)
class GeoPoint:
    """A geocoded latitude/longitude coordinate."""

    latitude: float
    longitude: float


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferredSlot:
    """
    A customer's preferred delivery slot — always a **day of week** plus a
    time-of-day window. A slot is meaningless without a day, so both are
    required; a customer with no preference simply has ``preferred_slot=None``.
    This is a *soft* preference (it feeds scoring, never a hard constraint).
    """

    day: DayOfWeek
    window: Window  # (start, end) time-of-day


@dataclass
class CustomerProfile:
    """
    New-customer intake (address, order quantity, optional preferred slot).

    `customer_number` is the Sysco identifier in ``NNN-NNNNNN`` form (site/OpCo
    + per-site number); see `shared/customer.py`. `location` is populated by the
    geocoding step; it is None until then. `name` is descriptive only — the
    workflow identifies customers by `customer_number`, never by name.
    """

    customer_number: str
    name: str
    address: str
    order_quantity_cases: int
    preferred_slot: Optional[PreferredSlot] = None  # soft preference: day + time window
    location: Optional[GeoPoint] = None  # filled in by geo-lookup


# ---------------------------------------------------------------------------
# Route / capacity data
# ---------------------------------------------------------------------------


@dataclass
class RouteStop:
    """An existing account already committed to a route's schedule."""

    customer_number: str  # Sysco customer number (NNN-NNNNNN)
    location: GeoPoint
    case_volume: int


@dataclass
class Route:
    """
    One (route, weekday) instance — a specific truck running a specific day.

    `service_center` + `service_radius_miles` describe the route's
    serviceable area; `committed_stops` are the accounts already on it (used
    for both remaining-capacity and geographic-clustering math).
    """

    route_id: str
    name: str
    day: DayOfWeek
    service_center: GeoPoint
    service_radius_miles: float
    vehicle_capacity_cases: int
    available_windows: list[Window] = field(default_factory=list)
    committed_stops: list[RouteStop] = field(default_factory=list)

    @property
    def committed_volume_cases(self) -> int:
        return sum(stop.case_volume for stop in self.committed_stops)


# ---------------------------------------------------------------------------
# Evaluation results (constraint + scoring trace)
# ---------------------------------------------------------------------------


@dataclass
class ConstraintOutcome:
    """Result of one hard-constraint check against one route."""

    name: str
    passed: bool
    detail: str


@dataclass
class FactorScore:
    """One weighted scoring factor's contribution for one route."""

    name: str
    weight: float
    value: float  # normalized 0.0 - 1.0
    detail: str

    @property
    def weighted(self) -> float:
        return self.weight * self.value


@dataclass
class CandidateEvaluation:
    """
    Full evaluation trace for a single candidate route: the geo/capacity
    facts, every hard-constraint outcome, and (if feasible) the scoring
    breakdown. This is what makes the recommendation auditable.
    """

    route: Route
    distance_miles: float
    chosen_window: Optional[Window]
    remaining_capacity_after: int
    utilization_after: float
    constraint_outcomes: list[ConstraintOutcome] = field(default_factory=list)
    factor_scores: list[FactorScore] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def feasible(self) -> bool:
        return bool(self.constraint_outcomes) and all(c.passed for c in self.constraint_outcomes)

    @property
    def failed_constraints(self) -> list[ConstraintOutcome]:
        return [c for c in self.constraint_outcomes if not c.passed]


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    ESCALATED_NO_FEASIBLE_SLOT = "ESCALATED_NO_FEASIBLE_SLOT"
    ESCALATED_LOW_SCORE = "ESCALATED_LOW_SCORE"


@dataclass
class SlotRecommendation:
    """
    The explainable output of the workflow for one customer.

    `total_score` is the winning route's own weighted score from Step 4 (see
    `shared/scoring.score_candidate`) — not a separately-computed "confidence."
    A route's own merit shouldn't be discounted just because another candidate
    happened to score nearly as well, so the escalation gate compares this
    number directly against `Config.total_score_threshold`.
    """

    customer_number: str
    customer_name: str
    decision: Decision
    total_score: float
    reasoning: str
    recommended_route_id: Optional[str] = None
    recommended_route_name: Optional[str] = None
    recommended_day: Optional[str] = None
    recommended_window: Optional[str] = None
    factor_breakdown: list[FactorScore] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)
    review_reason: Optional[str] = None

    @property
    def requires_human_review(self) -> bool:
        return self.decision != Decision.RECOMMENDED


@dataclass
class RecommendationResult:
    """
    Everything the workflow produced for one customer — the final
    recommendation plus the full trace of what was considered. Carried
    around so the CLI/UI can render an auditable decision, not just an answer.
    """

    customer: CustomerProfile
    candidates_considered: list[CandidateEvaluation]  # top-N, with constraint outcomes
    ranked_feasible: list[CandidateEvaluation]  # feasible options, best first
    recommendation: SlotRecommendation
