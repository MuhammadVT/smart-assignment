"""
Plain-Python orchestration of the Smart Assignment workflow — the single
source of truth for the 5-step process. The conversational agent's tools
(`tools/slot_recommendation.py`) and the offline demo (`scripts/run_local.py`)
both drive these same functions, so there is no logic drift between
"runnable now" and "deployable on ADK".

    1. intake            — validate the new customer's profile
    2. geo_lookup        — geocode + pick Top-N nearest candidate routes
    3. evaluate          — hard-constraint check each candidate (constraints.py)
    4. rank              — score every (route, slot) pair & sort  (scoring.py)
    5. decide            — recommend the best route-slot, or escalate to a human
                           (routeslot/decide.py)

Every collaborator (routes source, geocoder, config) is injectable, so pointing
this at real systems is a matter of passing different arguments — not editing
this file.
"""

from __future__ import annotations

from typing import Optional

from smart_assignment.integrations.geocoding_client import resolve_geocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.constraints import build_context, evaluate_constraints
from smart_assignment.shared.customer import validate_customer_number
from smart_assignment.shared.geo import Geocoder, haversine_miles
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    RecommendationResult,
    Route,
    ScoredSlot,
)
from smart_assignment.shared.scoring import score_route_slot
from smart_assignment.shared.slot_selection import SLOT_BASIS_NONE

# --- Step 1: intake ---------------------------------------------------------


def intake(customer: CustomerProfile) -> CustomerProfile:
    # New customers are prospects -- address (sourced from Salesforce/CRM) is
    # the required, primary identifier and drives geocoding.
    if not customer.address or not customer.address.strip():
        raise ValueError("customer address is required (it's the primary identifier)")
    # customer_number is an optional placeholder for accounts that already
    # have a Sysco number; only enforce the format when one was given.
    if customer.customer_number:
        customer.customer_number = validate_customer_number(customer.customer_number)
    if customer.order_quantity_cases <= 0:
        raise ValueError(
            f"{customer.lookup_key}: order_quantity_cases must be positive, "
            f"got {customer.order_quantity_cases}"
        )
    return customer


# --- Step 2: geo-lookup (geocode + Top-N nearest routes) --------------------


def geo_lookup(
    customer: CustomerProfile,
    routes: list[Route],
    geocoder: Geocoder,
    config: Config,
) -> list[Route]:
    customer.location = geocoder.geocode(customer.address)
    ranked_by_proximity = sorted(
        routes,
        key=lambda r: haversine_miles(customer.location, r.service_center),
    )
    return ranked_by_proximity[: config.top_n_candidate_routes]


# --- Step 3 + 4: evaluate constraints, then score the feasible ones ---------


def evaluate_candidates(
    customer: CustomerProfile, candidates: list[Route], config: Config
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    for route in candidates:
        ctx = build_context(customer, route, config)
        outcomes = evaluate_constraints(customer, route, ctx, config)
        evaluation = CandidateEvaluation(
            route=route,
            distance_miles=ctx.distance_miles,
            chosen_window=None,  # set below from the best scored slot, if any
            remaining_capacity_after=ctx.remaining_capacity_after,
            utilization_after=ctx.utilization_after,
            constraint_outcomes=outcomes,
            window_basis=SLOT_BASIS_NONE,  # replaced below when a slot is scored
            available_slots=ctx.available_slots,
        )
        _apply_route_slot_scores(customer, route, ctx, evaluation, config)
        evaluations.append(evaluation)
    return evaluations


def _apply_route_slot_scores(
    customer: CustomerProfile,
    route: Route,
    ctx,
    evaluation: CandidateEvaluation,
    config: Config,
) -> None:
    """Score each candidate slot as its own (route, slot) option and fold the
    route's BEST scored slot back onto the evaluation.

    Runs for EVERY candidate so that "the window this route offers" has one
    definition everywhere -- including on an infeasible route, where it is the
    diagnostic a specialist reads ("this route would have suited your Tuesday
    morning, but it's out of area").

    MERIT (`total_score`, `factor_scores`) is promoted only for a FEASIBLE
    candidate. A rejected route must never carry a score: hard constraints are
    absolute, and a merit number beside a rejection invites "it scored well, why
    wasn't it used?". Infeasible candidates therefore keep the 0.0 / empty
    defaults, and every decision layer filters on `feasible` before reasoning
    (see routeslot.decide._all_route_slots and routeslot.evidence).

    A route that produced no candidate slot keeps `chosen_window=None` and a 0.0
    score: it offers no assignable (route, slot) option, so it can only ever be
    reported, never recommended (see routeslot.decide._escalate_no_slot)."""
    scored = [
        ScoredSlot(slot=slot, factor_scores=fb, total_score=tot)
        for slot in evaluation.available_slots
        for fb, tot in [score_route_slot(customer, route, ctx, slot, config)]
    ]
    if not scored:
        return
    evaluation.scored_slots = scored
    best = max(scored, key=lambda s: s.total_score)
    evaluation.chosen_window = best.slot.window
    evaluation.window_basis = best.slot.basis
    if evaluation.feasible:
        evaluation.total_score = best.total_score
        evaluation.factor_scores = best.factor_scores


def rank_feasible(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    feasible = [e for e in evaluations if e.feasible]
    return sorted(feasible, key=lambda e: e.total_score, reverse=True)


# --- End-to-end -------------------------------------------------------------


def run_slot_recommendation(
    customer: CustomerProfile,
    routes: Optional[list[Route]] = None,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
) -> RecommendationResult:
    """Run the full workflow for one customer and return the complete trace.

    Step 5 is a single decision over the deterministically enumerated
    (route, slot) options (see `routeslot.decide_route_slot`). Hard constraints
    run first and remain the only thing that can eliminate a candidate, so the
    decision only ever ranks and gates the *feasible* survivors.

    Whether an LLM reasons over those options is internal to that layer
    (`use_grounded_route_slot_pick` / `use_grounded_route_slot_escalation`); it
    falls back to the deterministic threshold decision on any failure, so this
    still runs fully offline with no backend or credentials.
    """
    config = config or DEFAULT_CONFIG
    geocoder = geocoder or resolve_geocoder()

    customer = intake(customer)
    all_routes = routes if routes is not None else fetch_candidate_routes()
    candidates = geo_lookup(customer, all_routes, geocoder, config)
    evaluations = evaluate_candidates(customer, candidates, config)

    # Imported lazily so importing the pipeline never pulls in the LLM plumbing.
    from smart_assignment.routeslot import decide_route_slot

    recommendation = decide_route_slot(customer, evaluations, config)

    return RecommendationResult(
        customer=customer,
        candidates_considered=evaluations,
        ranked_feasible=rank_feasible(evaluations),
        recommendation=recommendation,
    )
