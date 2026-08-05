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

# How many times check_brief_grounding will hand back revision feedback within one
# triage invocation before telling the agent to finalize (see that tool, and
# _STATE_TRIAGE_CHECK_COUNT_KEY below). Two = the first check plus one revision.
#
# Each round costs a FULL regeneration of the brief -- the agent passes the whole
# brief as the tool's argument, then writes it again as its final answer -- and a
# brief generation is the only call in this system measured to reach the sage
# request timeout. So an unbounded revision loop multiplies the chance the whole
# turn dies, while adding no guarantee: agent.py's _finalize_brief backstop always
# re-verifies the final brief and appends a caveat naming anything still
# ungrounded, whether or not the agent ever self-checked.
MAX_GROUNDING_CHECKS = 2
_STATE_TRIAGE_CHECK_COUNT_KEY = "sa_triage_check_count"


def decision_thresholds(config: Config) -> dict:
    """The decision bars an escalation brief legitimately needs to name.

    An escalation is *defined* by a threshold it failed to clear, and the brief is
    explicitly asked to name "the specific gate that tripped ... with the exact
    numbers". Those bars used to be absent from the context, so the deterministic
    verifier flagged them as ungrounded even when the context's own
    ``review_reason`` had handed the agent the number ("No route-slot cleared the
    55% auto-assign bar"). The agent could only satisfy the check by dropping the
    figure, which cost it two or three full rewrites to discover.

    Publishing them here fixes that at the source: they are real facts of the
    decision, so they belong in the evidence packet, and ``collect_grounding``
    picks them up like any other fact. Fractions (0.55), matching how every other
    ratio in the context is stored -- the verifier's percent-vs-fraction rule lets
    a brief write "55%".
    """
    return {
        "auto_assign_score_bar": config.route_slot_score_threshold,
        "utilization_ceiling": config.max_utilization_after_assignment,
        "safe_utilization": round(
            config.max_utilization_after_assignment - config.capacity_buffer_safety_margin, 4
        ),
    }


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
        "thresholds": decision_thresholds(config),
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
    # Every triage invocation starts here, so this is where the per-invocation
    # revision budget resets -- a second escalation in the same session gets its
    # own full budget rather than inheriting the first one's spent count.
    tool_context.state[_STATE_TRIAGE_CHECK_COUNT_KEY] = 0
    return context


def check_brief_grounding(tool_context: ToolContext, brief: str) -> dict:
    """Verify that every number and route-id in a drafted triage ``brief`` is
    grounded in the escalation context.

    Call this after drafting the brief and before finalizing it. If it returns
    "ok": false, revise the brief to remove or correct the flagged figures --
    do not invent replacements -- then call this again.

    Returns:
      {"ok": true, "message": "..."} when everything is grounded;
      {"ok": false, "ungrounded_numbers": [...], "ungrounded_routes": [...],
       "ungrounded_days": [...], "ungrounded_times": [...],
       "message": "<what to fix>"} when something is not grounded; that payload
      also carries "stop": true once the revision budget is spent, meaning: do
      NOT check again -- drop whatever is still flagged and finalize the brief.
      Returns {"ok": false, "error": ...} if get_escalation_context hasn't run yet.
    """
    grounding = tool_context.state.get(_STATE_TRIAGE_GROUNDING_KEY)
    if not grounding:
        return {"ok": False, "error": "Call get_escalation_context first."}

    checks_used = int(tool_context.state.get(_STATE_TRIAGE_CHECK_COUNT_KEY) or 0) + 1
    tool_context.state[_STATE_TRIAGE_CHECK_COUNT_KEY] = checks_used

    result = verify_brief(brief or "", grounding)
    if result.ok:
        return {"ok": True, "message": "All figures and routes in the brief are grounded."}

    payload = {
        "ok": False,
        "ungrounded_numbers": result.ungrounded_numbers,
        "ungrounded_routes": result.ungrounded_routes,
        "ungrounded_days": result.ungrounded_days,
        "ungrounded_times": result.ungrounded_times,
        "message": result.caveat(),
    }
    if checks_used >= MAX_GROUNDING_CHECKS:
        # Budget spent. ``ok`` stays honest -- the brief really isn't fully
        # grounded -- and ``stop`` says what to do about it. Nothing is hidden by
        # stopping: agent.py's _finalize_brief re-verifies the final brief
        # deterministically and appends a caveat naming anything still ungrounded,
        # so the guarantee holds without another costly rewrite round.
        payload["stop"] = True
        payload["message"] = (
            f"{result.caveat()} Revision budget reached ({MAX_GROUNDING_CHECKS} checks) "
            "-- do NOT call this tool again. Drop or correct the flagged items and "
            "output your brief now; anything left will be flagged automatically."
        )
    return payload
