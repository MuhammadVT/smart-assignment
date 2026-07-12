"""
Conversational tool wrappers around the slot_recommendation pipeline.

Each function here is a thin, self-contained wrapper around a single
pipeline.py step -- the LLM (see smart_assignment/agent.py) calls these as
tools instead of computing anything itself. This keeps the whole thing
deterministic and auditable: the agent orchestrates *when* to call a step
and narrates the result, but every distance, constraint check, score, and
decision still comes straight from the same plain Python already covered by
tests/test_pipeline.py.

State: the in-progress customer profile lives in `tool_context.state` as a
plain JSON-serializable dict (see `_profile_to_state_dict` /
`_profile_from_state_dict`), not as a CustomerProfile object -- ADK session
state must be JSON-safe, and a plain dict is also the easiest thing to
inspect while debugging a conversation. Downstream products (candidate
routes, constraint outcomes, scores) are recomputed fresh from that profile
on every call rather than cached here, so a revision (a changed address,
cases, or preferred slot) always flows through correctly with no
invalidation logic to get wrong. The one exception is geocoding itself: a
real geocode is a network call, not free the way MockGeocoder's was, so
`CensusGeocoder` (see integrations/census_geocoder.py) caches successful
lookups process-wide by address -- geocoding the same address on 3
different tool calls in one turn costs one real request, not three.

Flexibility note: each tool below is independent, keyed only through
session state -- none of them call each other directly. That means any one
of them can later be lifted into its own sub-agent (wrapped in an
`AgentTool`-backed `LlmAgent`) and swapped into the parent agent's
`tools=[...]` list without changing this file or the other tools.
"""

from __future__ import annotations

from typing import Optional

from google.adk.tools import ToolContext

from smart_assignment.integrations.census_geocoder import CensusGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.constraints import CONSTRAINT_LABEL, build_context
from smart_assignment.shared.geo import AddressNotFoundError, GeocodingError
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    DayOfWeek,
    PreferredSlot,
    Route,
)
from smart_assignment.shared.timeutils import fmt_time, fmt_window, parse_time
from smart_assignment.judgment import default_judge
from smart_assignment.pipeline import evaluate_candidates, geo_lookup, intake
from smart_assignment.reasoning import LLMReasoner

# Namespaced so this doesn't collide with other state a larger app might keep.
_STATE_PROFILE_KEY = "sa_profile"
_STATE_LAST_RECOMMENDATION_KEY = "sa_last_recommendation"

# Real geocoder for the conversational path (pipeline.run_slot_recommendation's
# own default stays MockGeocoder -- see geocoding_client.py's docstring -- so
# the offline demo, GitHub Pages generator, and test suite are unaffected).
_GEOCODER = CensusGeocoder()


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _profile_to_state_dict(customer: CustomerProfile) -> dict:
    slot = customer.preferred_slot
    return {
        "name": customer.name,
        "address": customer.address,
        "order_quantity_cases": customer.order_quantity_cases,
        "customer_number": customer.customer_number,
        "preferred_day": slot.day.value if slot else None,
        "preferred_window_start": fmt_time(slot.window[0]) if slot else None,
        "preferred_window_end": fmt_time(slot.window[1]) if slot else None,
    }


def _profile_from_state_dict(profile: dict) -> CustomerProfile:
    slot = None
    day = profile.get("preferred_day")
    if day:
        slot = PreferredSlot(
            DayOfWeek(day),
            (
                parse_time(profile["preferred_window_start"]),
                parse_time(profile["preferred_window_end"]),
            ),
        )
    return CustomerProfile(
        name=profile.get("name") or "New prospect",
        address=profile.get("address", ""),
        order_quantity_cases=profile.get("order_quantity_cases", 0),
        customer_number=profile.get("customer_number"),
        preferred_slot=slot,
    )


def _serialize_evaluation(e: CandidateEvaluation) -> dict:
    out = {
        "route_id": e.route.route_id,
        "name": e.route.name,
        "day": e.route.day.value,
        "distance_miles": round(e.distance_miles, 1),
        "feasible": e.feasible,
        "utilization_after": round(e.utilization_after, 4),
        "constraints": [
            {
                "name": CONSTRAINT_LABEL.get(c.name, c.name),
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in e.constraint_outcomes
        ],
    }
    out["chosen_window"] = fmt_window(e.chosen_window)
    out["window_basis"] = e.window_basis
    out["available_slots"] = [
        {
            "window": fmt_window(s.window),
            "anchor_time": fmt_time(s.anchor_time) if s.anchor_time else None,
            "fit_score": round(s.fit_score, 4),
            "committed_overlap": s.committed_overlap,
            "basis": s.basis,
        }
        for s in e.available_slots
    ]
    if e.feasible:
        out["factor_scores"] = [
            {"name": f.name, "weight": f.weight, "value": round(f.value, 4), "detail": f.detail}
            for f in e.factor_scores
        ]
        out["total_score"] = round(e.total_score, 4)
    return out


def _find_candidates(customer: CustomerProfile) -> list[Route]:
    """Geocode + Top-N lookup (step 2), shared by every tool below that needs
    it. Raises `GeocodingError` (see shared/geo.py) on failure; callers
    convert that to the `{"ok": False, "error": ...}` tool-result shape via
    `_geocoding_error_result` rather than letting it crash the tool call."""
    return geo_lookup(customer, fetch_candidate_routes(), _GEOCODER, DEFAULT_CONFIG)


def _geocoding_error_result(exc: GeocodingError) -> dict:
    if isinstance(exc, AddressNotFoundError):
        return _error(
            f"I couldn't find a location for '{exc.address}' -- ask the customer to "
            f"double-check it, or provide a more complete address (street, city, state, ZIP)."
        )
    return _error(
        "The geocoding service is temporarily unavailable -- ask the customer to try "
        "again in a moment."
    )


# --- Step 1: intake (conversational, mergeable) -----------------------------


def intake_customer(
    tool_context: ToolContext,
    address: Optional[str] = None,
    order_quantity_cases: Optional[int] = None,
    preferred_day: Optional[str] = None,
    preferred_window_start: Optional[str] = None,
    preferred_window_end: Optional[str] = None,
    customer_number: Optional[str] = None,
    name: Optional[str] = None,
    clear_preferred_slot: bool = False,
) -> dict:
    """
    Record or update the prospect's intake details for this conversation.

    Call this first, and again any time the customer gives you a new or
    corrected value (e.g. "actually make it Tuesday instead", or "the order
    is 200 cases not 150"). Only pass the fields that changed -- anything
    already on file from an earlier call in this conversation is kept
    automatically, so you never need to repeat the full profile.

    Args:
      address: The prospect's street address. Required before any other
        step can run -- this is the primary identifier, since most new
        customers are prospects with no Sysco customer number yet.
      order_quantity_cases: The size of the order, in cases. Must be a
        positive number.
      preferred_day: Preferred delivery day of week, one of MON/TUE/WED/
        THU/FRI/SAT, if the customer stated one. Must be given together
        with preferred_window_start and preferred_window_end.
      preferred_window_start: Preferred delivery window start, 24-hour
        "HH:MM" (e.g. "07:00"). Must be given together with preferred_day
        and preferred_window_end.
      preferred_window_end: Preferred delivery window end, 24-hour "HH:MM".
      customer_number: An existing Sysco customer number ("NNN-NNNNNN"),
        only if the account already has one -- most prospects do not, and
        omitting it is the default, expected case.
      name: The business/contact name, if known. Not required to proceed.
      clear_preferred_slot: Set true if the customer says they no longer
        have a day/time preference, to remove one recorded earlier.

    Returns:
      On success: {"ok": true, "profile": {...the full current profile...}}.
      On failure: {"ok": false, "error": "<what to ask the customer to fix>"}.
      Always relay a failure to the customer and ask for a correction --
      never guess or invent a value yourself.
    """
    profile = dict(tool_context.state.get(_STATE_PROFILE_KEY) or {})

    if address is not None:
        profile["address"] = address
    if order_quantity_cases is not None:
        profile["order_quantity_cases"] = order_quantity_cases
    if customer_number is not None:
        profile["customer_number"] = customer_number
    if name is not None:
        profile["name"] = name

    if clear_preferred_slot:
        profile["preferred_day"] = None
        profile["preferred_window_start"] = None
        profile["preferred_window_end"] = None
    else:
        if preferred_day is not None:
            profile["preferred_day"] = preferred_day.strip().upper()
        if preferred_window_start is not None:
            profile["preferred_window_start"] = preferred_window_start
        if preferred_window_end is not None:
            profile["preferred_window_end"] = preferred_window_end

    slot_fields = (
        profile.get("preferred_day"),
        profile.get("preferred_window_start"),
        profile.get("preferred_window_end"),
    )
    if any(slot_fields) and not all(slot_fields):
        return _error(
            "A preferred delivery slot needs a day AND both a start and end "
            "time -- ask the customer for whichever part is missing, or drop "
            "the preference entirely."
        )

    if not profile.get("address"):
        return _error("I still need the customer's address before I can do anything else.")
    if not profile.get("order_quantity_cases"):
        return _error("I still need the order quantity, in cases, before I can do anything else.")

    try:
        customer = _profile_from_state_dict(profile)
    except ValueError as exc:
        return _error(f"That preferred slot doesn't parse: {exc}")

    try:
        intake(customer)
    except ValueError as exc:
        return _error(str(exc))

    profile = _profile_to_state_dict(customer)
    tool_context.state[_STATE_PROFILE_KEY] = profile
    return {"ok": True, "profile": profile}


# --- Step 2: geo-lookup ------------------------------------------------------


def find_candidate_routes(tool_context: ToolContext) -> dict:
    """
    Geocode the prospect's address and find the nearest candidate delivery
    routes (step 2 of the workflow).

    Call this only after intake_customer has returned {"ok": true}.

    Returns:
      {"ok": true,
       "geocoded_location": {"latitude": .., "longitude": ..},
       "candidate_routes": [{"route_id", "name", "day", "distance_miles"}, ...]}
      or {"ok": false, "error": "..."} if intake hasn't been completed yet, or
      if the address couldn't be geocoded.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile:
        return _error("Call intake_customer first -- there's no address on file yet.")
    customer = _profile_from_state_dict(profile)
    try:
        candidates = _find_candidates(customer)
    except GeocodingError as exc:
        return _geocoding_error_result(exc)
    return {
        "ok": True,
        "geocoded_location": {
            "latitude": customer.location.latitude,
            "longitude": customer.location.longitude,
        },
        "candidate_routes": [
            {
                "route_id": r.route_id,
                "name": r.name,
                "day": r.day.value,
                "distance_miles": round(build_context(customer, r).distance_miles, 1),
            }
            for r in candidates
        ],
    }


# --- Steps 3 + 4: hard constraints, then weighted scoring -------------------


def evaluate_and_score_routes(tool_context: ToolContext) -> dict:
    """
    Apply the hard constraints (service area, truck capacity) to the
    candidate routes, and weight-score every route that passes (steps 3
    and 4 of the workflow -- pipeline.py already combines them in one pass).

    Call this only after intake_customer has returned {"ok": true}; it
    geocodes and finds candidates internally, so you don't need to call
    find_candidate_routes first unless you also want to narrate that step.

    Returns:
      {"ok": true, "routes": [
        {"route_id", "name", "day", "distance_miles", "feasible",
         "utilization_after", "constraints": [{"name", "passed", "detail"}],
         "chosen_window", "window_basis" (why that slot was chosen),
         "available_slots": [{"window", "fit_score", "committed_overlap",
         "basis"}], "factor_scores" (only if feasible): [{"name", "weight",
         "value", "detail"}], "total_score" (only if feasible)},
        ...]}
      or {"ok": false, "error": "..."}.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile:
        return _error("Call intake_customer first -- there's no address on file yet.")
    customer = _profile_from_state_dict(profile)
    try:
        candidates = _find_candidates(customer)
    except GeocodingError as exc:
        return _geocoding_error_result(exc)
    evaluations = evaluate_candidates(customer, candidates, DEFAULT_CONFIG)
    return {"ok": True, "routes": [_serialize_evaluation(e) for e in evaluations]}


# --- Step 5: recommend or escalate ------------------------------------------


def recommend_or_escalate(tool_context: ToolContext) -> dict:
    """
    Rank the feasible routes and produce the final recommendation or
    escalation, with a full reasoning trace (step 5, the last step).

    Call this only after intake_customer has returned {"ok": true}; it
    re-derives candidates and scores internally. If the result's
    "requires_human_review" is true, you MUST call request_input to loop in
    a specialist before treating this prospect as done -- never present a
    low-score or no-feasible-slot result as final on your own.

    Returns:
      {"ok": true, "decision", "requires_human_review", "total_score",
       "recommended_route_id", "recommended_route_name", "recommended_day",
       "recommended_window", "recommended_window_basis" (why that slot was
       chosen), "reasoning", "rejected_alternatives", "review_reason"}
      or {"ok": false, "error": "..."}.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile:
        return _error("Call intake_customer first -- there's no address on file yet.")
    customer = _profile_from_state_dict(profile)
    try:
        candidates = _find_candidates(customer)
    except GeocodingError as exc:
        return _geocoding_error_result(exc)
    evaluations = evaluate_candidates(customer, candidates, DEFAULT_CONFIG)
    # Step-5 strategy comes from config: with SMART_ASSIGNMENT_USE_GROUNDED_JUDGMENT
    # off (the default) this is the existing weighted-sum pick narrated by the
    # LLM-backed reasoner; with it on, an LLM makes the recommend/escalate call
    # over the evidence packet (see the `judgment` package). Either way the LLM
    # path transparently falls back to the deterministic trace/pick on any
    # error, so this stays safe when the model/credentials are unavailable.
    judge = default_judge(DEFAULT_CONFIG, reasoner=LLMReasoner(DEFAULT_CONFIG))
    rec = judge.decide(customer, evaluations, DEFAULT_CONFIG)

    result = {
        "ok": True,
        "decision": rec.decision.value,
        "requires_human_review": rec.requires_human_review,
        "total_score": rec.total_score,
        "recommended_route_id": rec.recommended_route_id,
        "recommended_route_name": rec.recommended_route_name,
        "recommended_day": rec.recommended_day,
        "recommended_window": rec.recommended_window,
        "recommended_window_basis": rec.recommended_window_basis,
        "reasoning": rec.reasoning,
        "rejected_alternatives": rec.rejected_alternatives,
        "review_reason": rec.review_reason,
        # Split model opinions from grounded-judgment resampling (empty on the
        # weighted path). Surfaced so the escalation-triage sub-agent can show
        # the specialist where the automated judgment was divided.
        "alternative_takes": rec.alternative_takes,
    }
    tool_context.state[_STATE_LAST_RECOMMENDATION_KEY] = result
    return result
