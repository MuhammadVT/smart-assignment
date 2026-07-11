"""
Plain-Python orchestration of the Smart Assignment workflow — the single
source of truth for the 5-step process. The conversational agent's tools
(`tools/slot_recommendation.py`) and the offline demo (`scripts/run_local.py`)
both drive these same functions, so there is no logic drift between
"runnable now" and "deployable on ADK".

    1. intake            — validate the new customer's profile
    2. geo_lookup        — geocode + pick Top-N nearest candidate routes
    3. evaluate          — hard-constraint check each candidate (constraints.py)
    4. rank              — weighted multi-factor score & sort  (scoring.py)
    5. decide            — recommend the top slot, or escalate to a human

Every collaborator (routes source, geocoder, reasoner, config) is injectable,
so pointing this at real systems is a matter of passing different arguments —
not editing this file.
"""

from __future__ import annotations

from typing import Optional

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.constraints import (
    CONSTRAINT_LABEL,
    build_context,
    evaluate_constraints,
)
from smart_assignment.shared.customer import validate_customer_number
from smart_assignment.shared.geo import Geocoder, haversine_miles
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    Decision,
    RecommendationResult,
    Route,
    SlotRecommendation,
)
from smart_assignment.shared.scoring import score_candidate
from smart_assignment.shared.timeutils import fmt_window
from smart_assignment.reasoning import LLMReasoner, Reasoner, compute_total_score

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
            chosen_window=ctx.best_window,
            remaining_capacity_after=ctx.remaining_capacity_after,
            utilization_after=ctx.utilization_after,
            constraint_outcomes=outcomes,
            window_basis=ctx.window_basis,
            available_slots=ctx.available_slots,
        )
        if evaluation.feasible:
            breakdown, total = score_candidate(customer, route, ctx, config)
            evaluation.factor_scores = breakdown
            evaluation.total_score = total
        evaluations.append(evaluation)
    return evaluations


def rank_feasible(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    feasible = [e for e in evaluations if e.feasible]
    return sorted(feasible, key=lambda e: e.total_score, reverse=True)


# --- Step 5: decide (recommend or escalate) --------------------------------


def decide(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    reasoner: Reasoner,
    config: Config,
) -> SlotRecommendation:
    ranked = rank_feasible(evaluations)
    infeasible = [e for e in evaluations if not e.feasible]
    total_score = compute_total_score(ranked)
    reasoning = reasoner.explain(customer, ranked, infeasible, total_score, config)

    rejected: list[str] = []
    for cand in ranked[1:]:
        rejected.append(
            f"{cand.route.route_id} ({cand.route.day.value}): feasible but scored "
            f"{cand.total_score:.2f}"
        )
    for cand in infeasible:
        failed = ", ".join(CONSTRAINT_LABEL.get(c.name, c.name) for c in cand.failed_constraints)
        rejected.append(f"{cand.route.route_id} ({cand.route.day.value}): infeasible — {failed}")

    if not ranked:
        return SlotRecommendation(
            customer_number=customer.customer_number,
            customer_address=customer.address,
            customer_name=customer.name,
            decision=Decision.ESCALATED_NO_FEASIBLE_SLOT,
            total_score=total_score,
            reasoning=reasoning,
            rejected_alternatives=rejected,
            review_reason="No candidate route satisfied all hard constraints.",
        )

    winner = ranked[0]
    escalate = total_score < config.total_score_threshold
    decision = Decision.ESCALATED_LOW_SCORE if escalate else Decision.RECOMMENDED
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=decision,
        total_score=total_score,
        reasoning=reasoning,
        recommended_route_id=winner.route.route_id,
        recommended_route_name=winner.route.name,
        recommended_day=winner.route.day.value,
        recommended_window=fmt_window(winner.chosen_window),
        recommended_window_basis=winner.window_basis or None,
        factor_breakdown=winner.factor_scores,
        rejected_alternatives=rejected,
        review_reason=(
            f"Total score {total_score:.0%} below {config.total_score_threshold:.0%} threshold."
            if escalate
            else None
        ),
    )


# --- End-to-end -------------------------------------------------------------


def run_slot_recommendation(
    customer: CustomerProfile,
    routes: Optional[list[Route]] = None,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
    reasoner: Optional[Reasoner] = None,
) -> RecommendationResult:
    """Run the full workflow for one customer and return the complete trace."""
    config = config or DEFAULT_CONFIG
    geocoder = geocoder or MockGeocoder()
    # LLM-backed reasoning by default; it transparently falls back to the
    # deterministic trace when GOOGLE_API_KEY / Vertex credentials are absent,
    # so this still runs fully offline.
    reasoner = reasoner or LLMReasoner(config)

    customer = intake(customer)
    all_routes = routes if routes is not None else fetch_candidate_routes()
    candidates = geo_lookup(customer, all_routes, geocoder, config)
    evaluations = evaluate_candidates(customer, candidates, config)
    recommendation = decide(customer, evaluations, reasoner, config)

    return RecommendationResult(
        customer=customer,
        candidates_considered=evaluations,
        ranked_feasible=rank_feasible(evaluations),
        recommendation=recommendation,
    )
