"""
The escalation-triage agent's single data tool: assemble, from session state,
the grounded facts a specialist brief needs.

Read-only by design -- it re-derives the candidate evaluation (feasible +
infeasible, with the same raw per-route facts the grounded-judgment layer uses)
and returns it alongside the escalation reason and any split model opinions. It
never writes state and never changes the decision, the route, or a score; the
triage agent only *explains* what deterministic code already decided.

State keys and helpers are reused from ``tools/slot_recommendation.py`` (the
same module the web app already imports internals from), so triage stays in
lock-step with how the conversational tools store the profile and the last
recommendation.
"""

from __future__ import annotations

from google.adk.tools import ToolContext

from smart_assignment.judgment.evidence import build_evidence_packet
from smart_assignment.pipeline import evaluate_candidates
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.geo import GeocodingError
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_PROFILE_KEY,
    _find_candidates,
    _geocoding_error_result,
    _profile_from_state_dict,
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
    packet = build_evidence_packet(customer, evaluations, DEFAULT_CONFIG)

    return {
        "ok": True,
        "customer": {
            "name": customer.name,
            "address": customer.address,
            "order_quantity_cases": customer.order_quantity_cases,
            "preferred_slot": packet.customer.get("preferred_slot"),
        },
        "decision": last.get("decision"),
        "review_reason": last.get("review_reason"),
        "proposed_route_id": last.get("recommended_route_id"),
        "total_score": last.get("total_score"),
        "feasible_candidates": packet.feasible_candidates,
        "infeasible_candidates": packet.infeasible_candidates,
        "alternative_takes": last.get("alternative_takes", []),
    }
