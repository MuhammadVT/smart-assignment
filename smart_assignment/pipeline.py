"""
Plain-Python orchestration of the Smart Assignment workflow — the single
source of truth for the 5-step process. The conversational agent's tools
(`tools/slot_recommendation.py`) and the offline demo (`scripts/run_local.py`)
both drive these same functions, so there is no logic drift between
"runnable now" and "deployable on ADK".

    1. intake            — validate the new customer's profile
    2. geo_lookup        — geocode + pick Top-N nearest candidate routes (plus
                           the nearest preferred-day route, if the Top-N misses
                           the day the customer asked for)
    3. evaluate          — hard-constraint check each candidate (constraints.py)
    4. rank              — score every (route, slot) pair & sort  (scoring.py)
    5. decide            — recommend the best route-slot, or escalate to a human
                           (routeslot/decide.py)

Every collaborator (routes source, geocoder, config) is injectable, so pointing
this at real systems is a matter of passing different arguments — not editing
this file.
"""

from __future__ import annotations

import logging
from typing import Optional

from smart_assignment.integrations.geocoding_client import resolve_geocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.constraints import (
    build_context,
    evaluate_constraints,
    service_distance_limit,
)
from smart_assignment.shared.customer import validate_customer_number
from smart_assignment.shared.geo import Geocoder, haversine_miles
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    RecommendationResult,
    Route,
    ScoredSlot,
    SlotRecommendation,
)
from smart_assignment.shared.scoring import score_route_slot
from smart_assignment.shared.slot_selection import SLOT_BASIS_NONE

logger = logging.getLogger(__name__)

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
    """Geocode the customer and return the candidate routes to evaluate: the
    Top-N nearest, plus -- when `use_preferred_day_candidate` is on -- the
    nearest in-range route running on the customer's preferred day, if none of
    the Top-N already does (see `_preferred_day_candidate`)."""
    customer.location = geocoder.geocode(customer.address)
    ranked_by_proximity = sorted(
        routes,
        key=lambda r: haversine_miles(customer.location, r.service_center),
    )
    candidates = ranked_by_proximity[: config.top_n_candidate_routes]

    if config.use_preferred_day_candidate:
        extra = _preferred_day_candidate(customer, ranked_by_proximity, candidates, config)
        if extra is not None:
            logger.info(
                "Top-%d nearest routes run no %s; adding the nearest %s route %s (%.1f mi) "
                "as an extra candidate so the stated preference is scoreable.",
                config.top_n_candidate_routes,
                customer.preferred_slot.day.value,
                customer.preferred_slot.day.value,
                extra.route_id,
                haversine_miles(customer.location, extra.service_center),
            )
            candidates.append(extra)

    return candidates


def _preferred_day_candidate(
    customer: CustomerProfile,
    ranked_by_proximity: list[Route],
    candidates: list[Route],
    config: Config,
) -> Optional[Route]:
    """The nearest SERVICEABLE route running on the customer's preferred DAY, or
    ``None`` when there is nothing to add.

    Returns ``None`` when no preference was stated, when a candidate already runs
    on that day, or when no route in the whole set both runs on it and is within
    range. Because ``ranked_by_proximity`` is distance-ordered, a preferred-day
    route already inside the Top-N *is* the nearest one -- so "add the nearest
    preferred-day route" and "add one only when the day is missing" are the same
    rule, and this can never duplicate a candidate.

    **Capped at the service-area limit.** A route beyond
    ``constraints.service_distance_limit`` would provably fail
    ``geographic_serviceability``, so adding it would only put a
    guaranteed-rejected route in front of a specialist -- noise, not a
    diagnostic. The limit is per-route (a route's own radius, capped by the
    global ceiling) and comes from the constraint's own helper, so this filter
    can never disagree with the constraint that follows it. The scan continues
    past an out-of-range route rather than giving up, since a farther route may
    declare a wider radius and still be serviceable.

    The cap is deliberately about DISTANCE only. A preferred-day route that is in
    range but too full is still added, fails ``route_capacity``, and shows up as
    a rejected candidate -- that one a human genuinely can act on (split the
    order, move a stop), and unlike distance it depends on the order size rather
    than on geography alone."""
    preferred = customer.preferred_slot
    if preferred is None:
        return None
    if any(route.day == preferred.day for route in candidates):
        return None
    return next(
        (
            route
            for route in ranked_by_proximity
            if route.day == preferred.day
            and haversine_miles(customer.location, route.service_center)
            <= service_distance_limit(route, config)
        ),
        None,
    )


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
    recommendation: Optional[SlotRecommendation] = None,
) -> RecommendationResult:
    """Run the full workflow for one customer and return the complete trace.

    Step 5 is a single decision over the deterministically enumerated
    (route, slot) options (see `routeslot.decide_route_slot`). Hard constraints
    run first and remain the only thing that can eliminate a candidate, so the
    decision only ever ranks and gates the *feasible* survivors.

    #  TODO:help me understand
    Whether an LLM reasons over those options is internal to that layer
    (`use_grounded_route_slot_pick` / `use_grounded_route_slot_escalation`); it
    falls back to the deterministic threshold decision on any failure, so this
    still runs fully offline with no backend or credentials.

    `recommendation` REUSES a decision already made for this same customer,
    skipping step 5 entirely. Steps 1-4 are deterministic, so recomputing them is
    always reproducible -- but step 5 is not when grounded reasoning is on (it
    samples, and may resample for consensus). A surface that needs the *same*
    decision it already showed the user must pass it here rather than re-deciding
    and getting a second, independently-sampled answer (see
    `webapp/llm_chat._visualization_from_state`).
    """
    config = config or DEFAULT_CONFIG
    geocoder = geocoder or resolve_geocoder()

    customer = intake(customer)
    all_routes = routes if routes is not None else fetch_candidate_routes()
    candidates = geo_lookup(customer, all_routes, geocoder, config)
    evaluations = evaluate_candidates(customer, candidates, config)

    if recommendation is None:
        # Imported lazily so importing the pipeline never pulls in the LLM plumbing.
        from smart_assignment.routeslot import decide_route_slot

        recommendation = decide_route_slot(customer, evaluations, config)

    return RecommendationResult(
        customer=customer,
        candidates_considered=evaluations,
        ranked_feasible=rank_feasible(evaluations),
        recommendation=recommendation,
    )
