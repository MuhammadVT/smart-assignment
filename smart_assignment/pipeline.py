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

from typing import TYPE_CHECKING, Optional

from smart_assignment.integrations.geocoding_client import resolve_geocoder
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
    ScoredSlot,
    SlotRecommendation,
)
from smart_assignment.shared.scoring import score_candidate, score_route_slot
from smart_assignment.shared.timeutils import fmt_window
from smart_assignment.reasoning import LLMReasoner, Reasoner, compute_total_score

if TYPE_CHECKING:
    from smart_assignment.judgment import Judge

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
            if config.use_route_slot_scoring:
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
    route's BEST scored slot back onto the evaluation, so route-level ranking and
    the existing serialization reflect the best obtainable route-slot."""
    scored = [
        ScoredSlot(slot=slot, factor_scores=fb, total_score=tot)
        for slot in evaluation.available_slots
        for fb, tot in [score_route_slot(customer, route, ctx, slot, config)]
    ]
    if not scored:
        return
    evaluation.scored_slots = scored
    best = max(scored, key=lambda s: s.total_score)
    evaluation.total_score = best.total_score
    evaluation.factor_scores = best.factor_scores
    evaluation.chosen_window = best.slot.window
    evaluation.window_basis = best.slot.basis


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
    judge: Optional["Judge"] = None,
) -> RecommendationResult:
    """Run the full workflow for one customer and return the complete trace.

    `judge` selects the step-5 decision strategy. Precedence:
      1. an explicitly-passed `judge` always wins;
      2. otherwise, if `config.use_grounded_judgment` is on, an LLM makes the
         recommend/escalate call over the evidence packet (`judgment` package);
      3. otherwise, the weighted-sum `decide(...)` gated on
         `total_score_threshold` (the default).
    Hard constraints run first in every case, so the choice only affects how the
    *feasible* survivors are ranked and gated. Note the grounded path needs an
    LLM backend + credentials; without them it transparently falls back to the
    weighted deterministic result, so this still runs fully offline.
    """
    config = config or DEFAULT_CONFIG
    geocoder = geocoder or resolve_geocoder()
    # LLM-backed reasoning by default; it transparently falls back to the
    # deterministic trace when GOOGLE_API_KEY / Vertex credentials are absent,
    # so this still runs fully offline.
    reasoner = reasoner or LLMReasoner(config)

    # No explicit strategy injected -> honor the config flag. This is what makes
    # SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT take effect on the offline demo, the
    # page generator, and the web app -- not just the conversational tool.
    if judge is None and config.use_grounded_judgment:
        from smart_assignment.judgment import default_judge

        judge = default_judge(config, reasoner=reasoner)

    customer = intake(customer)
    all_routes = routes if routes is not None else fetch_candidate_routes()
    candidates = geo_lookup(customer, all_routes, geocoder, config)
    evaluations = evaluate_candidates(customer, candidates, config)

    if config.use_route_slot_scoring and judge is None:
        # The decision unit is the (route, slot) pair: one grounded decision over
        # route-slot options that also absorbs the slot pick (see the `routeslot`
        # package). Its own grounded/deterministic + fallback logic is internal,
        # so slotpick's separate pass is skipped here.
        from smart_assignment.routeslot import decide_route_slot

        recommendation = decide_route_slot(customer, evaluations, config)
    else:
        if judge is not None:
            recommendation = judge.decide(customer, evaluations, config)
        else:
            recommendation = decide(customer, evaluations, reasoner, config)

        # Optionally let an LLM pick the winning route's final slot from its
        # candidate menu (constrained + grounded); a no-op unless
        # use_grounded_slot_selection is on, and it never changes the route/score.
        if config.use_grounded_slot_selection:
            from smart_assignment.slotpick import refine_slot

            refine_slot(recommendation, evaluations, customer, config)

    return RecommendationResult(
        customer=customer,
        candidates_considered=evaluations,
        ranked_feasible=rank_feasible(evaluations),
        recommendation=recommendation,
    )
