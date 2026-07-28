"""
The escalation-triage agent's single data tool: assemble, from session state,
the grounded facts a specialist brief needs.

Read-only by design -- it re-derives the candidate evaluation (feasible +
infeasible, with the raw per-route facts from `triage/evidence.py`) and returns
it alongside the escalation reason and any split model opinions. It
never writes state and never changes the decision, the route, or a score; the
triage agent only *explains* what deterministic code already decided.

State keys and helpers are reused from ``tools/slot_recommendation.py`` (the
same module the web app already imports internals from), so triage stays in
lock-step with how the conversational tools store the profile and the last
recommendation.
"""

from __future__ import annotations

from google.adk.tools import ToolContext

from smart_assignment.pipeline import evaluate_candidates
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.geo import GeocodingError
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    SlotRecommendation,
)
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_PROFILE_KEY,
    _find_candidates,
    _geocoding_error_result,
    _profile_from_state_dict,
)
from smart_assignment.triage.evidence import build_evidence_packet
from smart_assignment.triage.verifier import collect_grounding, verify_brief

# Grounding facts stashed by get_escalation_context so the self-check tool and
# the after-model backstop can verify a brief without re-deriving the trace.
_STATE_TRIAGE_GROUNDING_KEY = "sa_triage_grounding"


def build_escalation_context(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    decision_facts: dict,
    config: Config,
) -> dict:
    """The grounded escalation context -- the exact dict ``get_escalation_context``
    returns, but assembled from plain objects with **no** ``ToolContext``.

    This is the seam that lets a non-ADK caller (the batch runner) reuse triage
    without a session or an agent loop: the conversational tool reads the two
    state keys and calls this; batch derives ``customer``/``evaluations`` from a
    ``RecommendationResult`` and calls it directly (see
    ``escalation_context_from_recommendation``). Pure and side-effect-free -- it
    reads no state and writes none, so both callers get an identical dict.

    ``decision_facts`` carries the last decision's scalar facts the brief cites:
    ``decision``, ``review_reason``, ``proposed_route_id``, ``total_score``,
    ``alternative_takes``.
    """
    packet = build_evidence_packet(customer, evaluations, config)
    return {
        "ok": True,
        "customer": {
            "name": customer.name,
            "address": customer.address,
            "order_quantity_cases": customer.order_quantity_cases,
            "preferred_slot": packet.customer.get("preferred_slot"),
        },
        "decision": decision_facts.get("decision"),
        "review_reason": decision_facts.get("review_reason"),
        "proposed_route_id": decision_facts.get("proposed_route_id"),
        "total_score": decision_facts.get("total_score"),
        "feasible_candidates": packet.feasible_candidates,
        "infeasible_candidates": packet.infeasible_candidates,
        "alternative_takes": decision_facts.get("alternative_takes") or [],
    }


def escalation_context_from_recommendation(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    recommendation: SlotRecommendation,
    config: Config,
) -> dict:
    """Build the escalation context straight from a pipeline ``RecommendationResult``'s
    parts -- the ergonomic entry point for the batch runner, which already holds the
    ``customer``, the ``evaluations`` (``candidates_considered``), and the
    ``recommendation``. A thin adapter over :func:`build_escalation_context` that maps
    the ``SlotRecommendation`` fields onto the ``decision_facts`` the builder wants."""
    return build_escalation_context(
        customer,
        evaluations,
        {
            "decision": recommendation.decision.value,
            "review_reason": recommendation.review_reason,
            "proposed_route_id": recommendation.recommended_route_id,
            "total_score": recommendation.total_score,
            "alternative_takes": recommendation.alternative_takes,
        },
        config,
    )


def get_escalation_context(tool_context: ToolContext) -> dict:
    """Return the grounded facts for triaging the current escalation.

    Reads the in-progress customer profile and the last recommendation from
    session state, re-derives the full candidate evaluation, and returns:
    the customer/order, why it escalated, the proposed route (if any), every
    feasible and infeasible route with its raw facts, and any split automated
    opinions (``alternative_takes``).

    Returns:
      On success: ``{"ok": true, ...}`` with the fields above.
      On failure: ``{"ok": false, "error": "..."}`` when there is nothing to
      triage -- no profile yet, no recommendation yet, or the last
      recommendation was auto-approved (so no human review is needed).
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile:
        return {"ok": False, "error": "No customer profile on file yet -- run intake first."}

    last = tool_context.state.get(_STATE_LAST_RECOMMENDATION_KEY)
    if not last:
        return {
            "ok": False,
            "error": "No recommendation to triage yet -- call recommend_or_escalate first.",
        }
    if not last.get("requires_human_review"):
        return {
            "ok": False,
            "error": "The last recommendation was auto-approved; there is nothing to triage.",
        }

    customer = _profile_from_state_dict(profile)
    try:
        candidates = _find_candidates(customer)
    except GeocodingError as exc:
        return _geocoding_error_result(exc)

    evaluations = evaluate_candidates(customer, candidates, DEFAULT_CONFIG)
    context = build_escalation_context(
        customer,
        evaluations,
        {
            "decision": last.get("decision"),
            "review_reason": last.get("review_reason"),
            "proposed_route_id": last.get("recommended_route_id"),
            "total_score": last.get("total_score"),
            "alternative_takes": last.get("alternative_takes", []),
        },
        DEFAULT_CONFIG,
    )
    # Stash the groundable facts so check_brief_grounding (and the after-model
    # backstop) can verify the brief without re-deriving the whole trace.
    tool_context.state[_STATE_TRIAGE_GROUNDING_KEY] = collect_grounding(context)
    return context


def check_brief_grounding(tool_context: ToolContext, brief: str) -> dict:
    """Verify that every number and route-id in a drafted triage ``brief`` is
    grounded in the escalation context.

    Call this after drafting the brief and before finalizing it. If it returns
    "ok": false, revise the brief to remove or correct the flagged figures --
    do not invent replacements -- then call this again.

    Returns:
      {"ok": true, "message": "..."} when everything is grounded, or
      {"ok": false, "ungrounded_numbers": [...], "ungrounded_routes": [...],
       "ungrounded_days": [...], "ungrounded_times": [...],
       "message": "<what to fix>"}. Returns {"ok": false, "error": ...} if
      get_escalation_context hasn't run yet.
    """
    grounding = tool_context.state.get(_STATE_TRIAGE_GROUNDING_KEY)
    if not grounding:
        return {"ok": False, "error": "Call get_escalation_context first."}
    result = verify_brief(brief or "", grounding)
    if result.ok:
        return {"ok": True, "message": "All figures and routes in the brief are grounded."}
    return {
        "ok": False,
        "ungrounded_numbers": result.ungrounded_numbers,
        "ungrounded_routes": result.ungrounded_routes,
        "ungrounded_days": result.ungrounded_days,
        "ungrounded_times": result.ungrounded_times,
        "message": result.caveat(),
    }
