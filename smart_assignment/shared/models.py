"""
Domain models for the Smart Assignment slot-recommendation workflow.

These are intentionally small, framework-agnostic dataclasses so that any
orchestration layer (the plain-Python pipeline in `pipeline.py`, the
conversational agent's tool wrappers in `tools/slot_recommendation.py`, a
future sub-agent split) shares the exact same data contracts.

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

    New customers are **prospects** — Salesforce/CRM has their address, but
    they don't have a Sysco customer number yet, so `address` (not
    `customer_number`) is the primary identifier and drives geocoding.
    `customer_number` is an optional placeholder in the Sysco ``NNN-NNNNNN``
    form (site/OpCo + per-site number; see `shared/customer.py`) for the case
    where this workflow is run for an account that already has one.
    `location` is populated by the geocoding step; it is None until then.
    `name` is descriptive only.
    """

    name: str
    address: str
    order_quantity_cases: int
    customer_number: Optional[str] = None  # optional Sysco number, if already on file
    preferred_slot: Optional[PreferredSlot] = None  # soft preference: day + time window
    location: Optional[GeoPoint] = None  # filled in by geo-lookup

    @property
    def lookup_key(self) -> str:
        """Stable identifier for this customer: the Sysco number if on file, else address."""
        return self.customer_number or self.address


# ---------------------------------------------------------------------------
# Route / capacity data
# ---------------------------------------------------------------------------


@dataclass
class RouteStop:
    """An existing account already committed to a route's schedule."""

    customer_number: str  # Sysco customer number (NNN-NNNNNN)
    location: GeoPoint
    delivery_time_window: Optional[Window] = None  # TW1 open/close times from historical data
    customer_tier: Optional[str] = None  # Sysco cust tier ("4"/"5"/"Perks"/"Other"), if known


@dataclass
class Route:
    """
    One (route, weekday) instance — a specific truck running a specific day.

    `service_center` + `service_radius_miles` describe the route's
    serviceable area; `committed_stops` are the accounts already on it (used
    for geographic-clustering math). Load and capacity fields feed capacity math.
    """

    route_id: str
    name: str
    day: DayOfWeek
    service_center: GeoPoint
    service_radius_miles: Optional[float] = None
    vehicle_capacity_weight: float = 0.0
    vehicle_capacity_cases: float = 0.0
    vehicle_capacity_cubes: float = 0.0
    avg_load_weight: float = 0.0
    avg_load_cases: float = 0.0
    avg_load_cubes: float = 0.0
    available_windows: list[Window] = field(default_factory=list)
    committed_stops: list[RouteStop] = field(default_factory=list)

    @property
    def committed_volume_cases(self) -> int:
        return self.avg_load_cases


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


@dataclass(frozen=True)
class SlotOption:
    """
    One window a route offers, annotated for slot selection: how well it fits
    the prospect's location relative to the route's committed stops, and how
    contended it already is. Produced by `identify_available_slots`
    (shared/slot_selection.py); the recommendation step picks among these.
    """

    window: Window
    fit_score: float  # 0.0-1.0 inverse-distance-weighted support from nearby committed stops
    committed_overlap: int  # how many committed stops' windows overlap this one (contention)
    basis: str  # why this option exists: "between_adjacent_stops" | "least_contended"


@dataclass
class CandidateEvaluation:
    """
    Full evaluation trace for a single candidate route: the geo/capacity
    facts, every hard-constraint outcome, and (if feasible) the scoring
    breakdown. This is what makes the recommendation auditable.

    `chosen_window` is the single recommended slot; `available_slots` is the
    full menu that was considered (with fit + contention), and `window_basis`
    records why `chosen_window` won.
    """

    route: Route
    distance_miles: float
    chosen_window: Optional[Window]
    remaining_capacity_after: int
    utilization_after: float
    constraint_outcomes: list[ConstraintOutcome] = field(default_factory=list)
    factor_scores: list[FactorScore] = field(default_factory=list)
    total_score: float = 0.0
    window_basis: str = ""
    available_slots: list[SlotOption] = field(default_factory=list)

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

    `customer_number` is optional -- most new customers are prospects with no
    Sysco number yet, so `customer_address` is always populated as the
    fallback identifier for display.
    """

    customer_name: str
    decision: Decision
    total_score: float
    reasoning: str
    customer_number: Optional[str] = None
    customer_address: Optional[str] = None
    recommended_route_id: Optional[str] = None
    recommended_route_name: Optional[str] = None
    recommended_day: Optional[str] = None
    recommended_window: Optional[str] = None
    recommended_window_basis: Optional[str] = None  # why this slot was chosen (audit trail)
    factor_breakdown: list[FactorScore] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)
    review_reason: Optional[str] = None
    # Populated only by the grounded-judgment path (see the `judgment` package)
    # when an escalation-side case was resampled: each entry is one independent
    # sample's reasoned take, surfaced to the specialist so they see where the
    # model agreed or was split. Empty for the default weighted-sum path.
    alternative_takes: list[str] = field(default_factory=list)
    # Set by GroundedJudge when grounded judgment was requested but the LLM path
    # failed (no backend/credentials, unparseable/ungrounded reply) and it fell
    # back to the deterministic weighted result. Lets the UI tell the user the
    # reasoning shown is the deterministic fallback, not grounded output.
    grounded_fallback: bool = False
    grounded_fallback_reason: Optional[str] = None

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
