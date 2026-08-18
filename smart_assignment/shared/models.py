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


# What every intake path stores when a prospect has no business name: a
# DESCRIPTION, not a name. Named here because more than one caller writes it and
# at least one reader must recognise it -- eval/case_source.py replays curated
# production decisions, and treating this placeholder as a real name made the
# replay demand the agent extract "New prospect" as the customer's name, which it
# rightly refuses to do.
PROSPECT_PLACEHOLDER_NAME = "New prospect"


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

    window: Window  # the recommended (fixed-length, centered) window
    fit_score: float  # 0.0-1.0 proximity-weight share of this candidate's stop cluster
    committed_overlap: int  # how many committed stops' windows overlap this one (contention)
    basis: str  # why this option exists: "between_adjacent_stops" | "least_contended"
    anchor_time: Optional[time] = None  # the interpolated time the window is centered on


@dataclass
class ScoredSlot:
    """One candidate slot scored as its own (route, slot) option, for the
    route-slot scoring path (see shared/scoring.score_route_slot). `total_score`
    is this slot's weighted total; `factor_scores` its per-slot breakdown
    (geo/capacity shared from the route, window_match/availability slot-specific).
    Populated for every feasible route that produced at least one candidate slot.
    """

    slot: SlotOption
    factor_scores: list[FactorScore]
    total_score: float

    @property
    def window(self) -> Window:
        return self.slot.window

    @property
    def basis(self) -> str:
        return self.slot.basis


@dataclass
class CandidateEvaluation:
    """
    Full evaluation trace for a single candidate route: the geo/capacity
    facts, every hard-constraint outcome, and (if feasible) the scoring
    breakdown. This is what makes the recommendation auditable.

    `chosen_window` is the single recommended slot; `available_slots` is the
    full menu that was considered (with fit + contention), and `window_basis`
    records why `chosen_window` won.

    `scored_slots` is populated only on the route-slot path: each candidate slot
    scored as its own option. On that path `total_score`, `chosen_window` and
    `factor_scores` mirror the route's BEST scored slot, so route-level ranking
    reflects the best obtainable route-slot.
    """

    route: Route
    distance_miles: float
    chosen_window: Optional[Window]
    # TODO how remaining_capacity_after & utilization_after are different
    remaining_capacity_after: int
    utilization_after: float
    constraint_outcomes: list[ConstraintOutcome] = field(default_factory=list)
    factor_scores: list[FactorScore] = field(default_factory=list)
    total_score: float = 0.0
    window_basis: str = ""
    available_slots: list[SlotOption] = field(default_factory=list)
    scored_slots: list[ScoredSlot] = field(default_factory=list)

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
    `shared/scoring.score_route_slot`) — not a separately-computed "confidence."
    A route's own merit shouldn't be discounted just because another candidate
    happened to score nearly as well, so the escalation gate compares this
    number directly against `Config.route_slot_score_threshold`.

    `customer_number` is optional -- most new customers are prospects with no
    Sysco number yet, so `customer_address` is always populated as the
    fallback identifier for display.
    """

    customer_name: str
    decision: Decision
    total_score: float
    reasoning: str
    customer_number: Optional[str] = None
    # TODO should address be required?
    customer_address: Optional[str] = None
    recommended_route_id: Optional[str] = None
    recommended_route_name: Optional[str] = None
    recommended_day: Optional[str] = None
    recommended_window: Optional[str] = None
    recommended_window_basis: Optional[str] = None  # why this slot was chosen (audit trail)
    # Set only when a verified grounded route-slot choice produced the pick --
    # its grounded rationale. None on the deterministic path.
    recommended_window_rationale: Optional[str] = None
    # Structured, grounded explanation of a RECOMMENDED route-slot pick, populated
    # only when the grounded route-slot decision (see the `routeslot` package)
    # succeeded. Each field maps to its own UI section so the ops manager sees the
    # rationale AND the trade-off, not a one-liner. All None/empty on the
    # deterministic path, so flag-off output is unchanged. `reasoning` is still set
    # (composed from these) so existing consumers keep working.
    decision_summary: Optional[str] = None  # one action line
    primary_reasons: list[str] = field(default_factory=list)  # the decisive factors
    key_tradeoff: Optional[str] = None  # what the winner gives up, and why it's acceptable
    runner_up: Optional[str] = None  # the next-best option and why it lost
    default_comparison: Optional[str] = None  # agreed-with / diverged-from the weighted default
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

    # --- lossless round-trip through JSON-able session state -----------------
    #
    # Step 5 can be non-deterministic (the grounded route-slot decision samples
    # and may resample), so a surface that needs the SAME decision twice must
    # carry it rather than recompute it -- see webapp/llm_chat, which would
    # otherwise show the agent's narration over a second, independently-sampled
    # result. Kept next to the fields so a newly added field is hard to forget.

    def to_state_dict(self) -> dict:
        """A JSON-safe snapshot carrying EVERY field (see `from_state_dict`)."""
        return {
            "customer_name": self.customer_name,
            "decision": self.decision.value,
            "total_score": self.total_score,
            "reasoning": self.reasoning,
            "customer_number": self.customer_number,
            "customer_address": self.customer_address,
            "recommended_route_id": self.recommended_route_id,
            "recommended_route_name": self.recommended_route_name,
            "recommended_day": self.recommended_day,
            "recommended_window": self.recommended_window,
            "recommended_window_basis": self.recommended_window_basis,
            "recommended_window_rationale": self.recommended_window_rationale,
            "decision_summary": self.decision_summary,
            "primary_reasons": list(self.primary_reasons),
            "key_tradeoff": self.key_tradeoff,
            "runner_up": self.runner_up,
            "default_comparison": self.default_comparison,
            "factor_breakdown": [
                {"name": f.name, "weight": f.weight, "value": f.value, "detail": f.detail}
                for f in self.factor_breakdown
            ],
            "rejected_alternatives": list(self.rejected_alternatives),
            "review_reason": self.review_reason,
            "alternative_takes": list(self.alternative_takes),
            "grounded_fallback": self.grounded_fallback,
            "grounded_fallback_reason": self.grounded_fallback_reason,
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> "SlotRecommendation":
        """Rebuild from `to_state_dict`. Raises on an unknown decision value, so a
        corrupt snapshot fails loudly at the call site (which then recomputes)
        rather than silently producing a wrong card."""
        return cls(
            customer_name=data["customer_name"],
            decision=Decision(data["decision"]),
            total_score=data["total_score"],
            reasoning=data["reasoning"],
            customer_number=data.get("customer_number"),
            customer_address=data.get("customer_address"),
            recommended_route_id=data.get("recommended_route_id"),
            recommended_route_name=data.get("recommended_route_name"),
            recommended_day=data.get("recommended_day"),
            recommended_window=data.get("recommended_window"),
            recommended_window_basis=data.get("recommended_window_basis"),
            recommended_window_rationale=data.get("recommended_window_rationale"),
            decision_summary=data.get("decision_summary"),
            primary_reasons=list(data.get("primary_reasons") or []),
            key_tradeoff=data.get("key_tradeoff"),
            runner_up=data.get("runner_up"),
            default_comparison=data.get("default_comparison"),
            factor_breakdown=[
                FactorScore(
                    name=f["name"], weight=f["weight"], value=f["value"], detail=f["detail"]
                )
                for f in (data.get("factor_breakdown") or [])
            ],
            rejected_alternatives=list(data.get("rejected_alternatives") or []),
            review_reason=data.get("review_reason"),
            alternative_takes=list(data.get("alternative_takes") or []),
            grounded_fallback=bool(data.get("grounded_fallback", False)),
            grounded_fallback_reason=data.get("grounded_fallback_reason"),
        )


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
