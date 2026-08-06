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

import logging
import re
from typing import Optional

from google.adk.tools import ToolContext

from smart_assignment.integrations.geocoding_client import resolve_geocoder
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
    SlotRecommendation,
)
from smart_assignment.shared.timeutils import fmt_time, fmt_window, parse_time
from smart_assignment.address_resolve import resolve_from_geocoder
from smart_assignment.pipeline import evaluate_candidates, geo_lookup, intake

logger = logging.getLogger(__name__)

# Namespaced so this doesn't collide with other state a larger app might keep.
_STATE_PROFILE_KEY = "sa_profile"
_STATE_LAST_RECOMMENDATION_KEY = "sa_last_recommendation"
# The full decision, kept SEPARATE from the agent-facing summary above so the
# agent's context stays lean while a surface that must re-render the very same
# decision can rebuild it exactly (see `cached_decision_for`). Bound to the
# profile it was computed from, because step 5 is non-deterministic under
# grounded reasoning and must never be reused for a different prospect.
_STATE_LAST_DECISION_KEY = "sa_last_decision"

# Every state key that belongs to ONE prospect and must never survive into the
# next. Cleared together whenever intake detects a new prospect (see the guard in
# `intake_customer`): a stale decision would let `cached_decision_for` re-render --
# and the triage tool ground a specialist brief on -- the PREVIOUS customer's
# outcome. The triage keys are declared in triage/context.py; they are spelled as
# literals here because triage imports this module, so importing them back would
# be a cycle. tests/tools/test_slot_recommendation.py asserts the spellings match.
_PROSPECT_SCOPED_STATE_KEYS = (
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_LAST_DECISION_KEY,
    "sa_triage_grounding",  # triage.context._STATE_TRIAGE_GROUNDING_KEY
    "sa_triage_check_count",  # triage.context._STATE_TRIAGE_CHECK_COUNT_KEY
)

# Geocoder for the conversational path, chosen by SMART_ASSIGNMENT_GEOCODER
# (census | mock; default census -- see geocoding_client.resolve_geocoder).
# Every surface resolves through the same factory and both load the same .env,
# so adk web and the web app use the same geocoder.
_GEOCODER = resolve_geocoder()


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _normalize_address(address: str) -> str:
    """Canonical form for deciding whether two intake addresses are THE SAME
    place: casefolded, punctuation dropped, whitespace collapsed. Deliberately
    aggressive -- a spurious mismatch here would reset a profile over a formatting
    difference ("St." vs "St"), while two genuinely different addresses cannot be
    made equal by dropping punctuation."""
    return " ".join(re.sub(r"[^\w]+", " ", address).casefold().split())


def _has_recorded_decision(state) -> bool:
    """Whether the profile currently on file has already produced a decision
    (recommendation or escalation) this conversation."""
    return bool(
        state.get(_STATE_LAST_RECOMMENDATION_KEY) or state.get(_STATE_LAST_DECISION_KEY)
    )


def _reset_prospect_state(state) -> None:
    """Clear every prospect-scoped key (decision snapshot, triage grounding) so
    nothing from the previous customer can be re-rendered or cited for the next
    one. Keys are set to None rather than deleted: ADK session state propagates
    assignments through its state delta, and every reader uses ``.get(...)``, so
    None reads exactly like absent."""
    for key in _PROSPECT_SCOPED_STATE_KEYS:
        state[key] = None


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

    Exception: a DIFFERENT address after a completed recommendation or
    escalation starts a fresh prospect automatically (nothing is kept).

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
      clear_preferred_slot: Set true ONLY to REMOVE a preferred slot recorded
        EARLIER in this conversation, when the customer changes their mind
        ("actually, any day works"). If a customer simply has no preference,
        do not pass this -- just omit the preferred_* fields.

    Returns:
      On success: {"ok": true, "profile": {...the full current profile...}}.
      On failure: {"ok": false, "error": "<what to ask the customer to fix>"}.
      Always relay a failure to the customer and ask for a correction --
      never guess or invent a value yourself.
    """
    profile = dict(tool_context.state.get(_STATE_PROFILE_KEY) or {})

    # A NEW PROSPECT, not a revision: the profile on file already produced a
    # decision, and this call brings a different address. Merging here is how one
    # customer's unstated fields (order size, preferred slot) silently became the
    # next customer's -- observed live as a prospect decided against a delivery
    # preference its customer never expressed. Start fresh from the passed fields
    # and drop every prospect-scoped leftover (stale decision snapshot, triage
    # grounding). Deterministic on purpose: after a decision, a different address
    # IS a different customer, and no model judgment can override that. An address
    # CORRECTION is unaffected -- it happens before a decision exists (see the
    # resolve_address flow), where this guard never fires.
    stored_address = profile.get("address")
    if (
        address is not None
        and stored_address
        and _normalize_address(address) != _normalize_address(stored_address)
        and _has_recorded_decision(tool_context.state)
    ):
        logger.info(
            "Intake received a new address after a concluded decision; starting a "
            "fresh prospect. Previous address: %r -> new address: %r. The previous "
            "profile and its decision/triage state were discarded.",
            stored_address,
            address,
        )
        profile = {}
        _reset_prospect_state(tool_context.state)

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

    # Persist the accumulated fields NOW, before the validation checks below.
    # Intake is conversational and often arrives one field at a time (address this
    # turn, order quantity the next); if we only saved on full success, a partial
    # call would be thrown away on its required-field error and the customer would
    # be asked to repeat what they already gave. On success this raw dict is
    # replaced by the normalized profile below.
    tool_context.state[_STATE_PROFILE_KEY] = profile

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


def start_new_prospect(tool_context: ToolContext) -> dict:
    """
    Discard the customer currently on file and start fresh for a DIFFERENT one.

    Call this FIRST -- before intake_customer -- whenever the user moves on to
    another customer in the same conversation ("new customer", "next prospect",
    "another one"), so nothing from the previous customer carries over. It
    works even when the new customer's message has no address yet. Never call
    it for a correction or revision of the CURRENT customer -- that would throw
    away details the user already gave and force them to repeat everything.

    Returns:
      {"ok": true, "message": "..."} -- then proceed with intake_customer for
      the new customer's details as usual.
    """
    # Model-declared boundary between customers. Trustworthy in exactly one
    # direction: a spurious call costs a re-ask, never contamination -- it only
    # ever DISCARDS state. The deterministic guard in intake_customer still fires
    # on its own whenever this call was forgotten but the address changed after a
    # decision, so correctness never rests on the model remembering it. A separate
    # no-arg tool rather than an intake_customer parameter, deliberately: the
    # golden eval pins intake's argument dict exactly, and the IN_ORDER trajectory
    # matcher tolerates extra tool CALLS -- so even a spuriously-called boundary
    # can never flake the eval, while an extra argument did (measured live).
    profile = dict(tool_context.state.get(_STATE_PROFILE_KEY) or {})
    if profile:
        logger.info(
            "start_new_prospect: discarding the profile on file (address %r) and "
            "its decision/triage state.",
            profile.get("address"),
        )
    tool_context.state[_STATE_PROFILE_KEY] = None
    _reset_prospect_state(tool_context.state)
    return {
        "ok": True,
        "message": (
            "Started a fresh prospect; nothing from the previous customer is on "
            "file. Collect the new customer's address and order size."
        ),
    }


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


# --- On-demand lookup: where is the prospect? (NOT a pipeline step) ---------
#
# Deliberately a SIBLING of find_candidate_routes, not a split of it. Step 2 owns
# "geocode AND rank the nearest routes" and stays exactly as it was; this answers
# the side question "where is this address?" on its own, without fetching and
# ranking the route set or filling the model's context with candidates the user
# never asked about.
#
# Two properties keep it safe to call at any point in a conversation:
#   * It reads only the ADDRESS, so it works before the order quantity has been
#     given -- geocoding needs nothing else to be on file.
#   * It writes NO state. The geocoder caches successful lookups process-wide
#     (see the module docstring), so a repeat costs no extra request, and staying
#     stateless preserves the "recompute fresh from the profile" invariant: a
#     corrected address can never be answered with a stale point.


def geocode_prospect_address(tool_context: ToolContext) -> dict:
    """
    Return the map coordinates (latitude/longitude) of the prospect's address
    currently on file -- e.g. "where is this customer located?", "what are the
    coordinates?", "did that address resolve?".

    This is a lookup, not a workflow step: it does NOT check routes, capacity, or
    availability, and it never replaces recommend_or_escalate for a route/slot
    decision. Call it when the user asks about the LOCATION itself.

    Call this after intake_customer has recorded an address.

    Returns:
      {"ok": true, "address": "...",
       "geocoded_location": {"latitude": .., "longitude": ..}}
      or {"ok": false, "error": "..."} if there's no address on file yet, or if
      the address couldn't be geocoded. The result carries the coordinates and
      the address only -- never state a city, county, or neighborhood that a tool
      didn't return.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    address = (profile or {}).get("address")
    if not address:
        return _error("Call intake_customer first -- there's no address on file yet.")
    try:
        location = _GEOCODER.geocode(address)
    except GeocodingError as exc:
        # Same failure shape and wording as every other tool here, so an address
        # miss routes to resolve_address identically no matter who hit it first.
        return _geocoding_error_result(exc)
    return {
        "ok": True,
        "address": address,
        "geocoded_location": {
            "latitude": location.latitude,
            "longitude": location.longitude,
        },
    }


# --- Step 2b: address resolution (grounded typo/ambiguity correction) -------


def resolve_address(tool_context: ToolContext) -> dict:
    """
    Suggest a corrected delivery address when the on-file address couldn't be
    geocoded (find_candidate_routes / evaluate / recommend returned an error
    saying the address wasn't found).

    This asks the geocoder for its real candidate matches and picks the closest
    one to what the customer typed -- it NEVER invents an address. The result is
    a SUGGESTION only: present it to the customer and get their confirmation
    before doing anything with it.

    Returns:
      On a suggestion: {"ok": true, "needs_confirmation": true,
        "original_address", "suggested_address", "alternatives": [...],
        "rationale", "message"} -- show the message, then wait for the customer
        to confirm or correct. Only after they confirm, call intake_customer with
        the confirmed address and continue the workflow.
      When nothing close was found: {"ok": false, "no_suggestions": true,
        "message": "..."} -- relay it and ask the customer to double-check.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile or not profile.get("address"):
        return _error("There's no address on file yet -- call intake_customer first.")
    address = profile["address"]

    # Feature off (or, defensively, any failure): today's behavior.
    if not DEFAULT_CONFIG.use_address_resolution:
        return _double_check_result(address)

    try:
        resolved = resolve_from_geocoder(address, _GEOCODER, DEFAULT_CONFIG)
    except Exception:  # noqa: BLE001 - never worse than the deterministic fallback
        return _double_check_result(address)

    if resolved is None:
        return _double_check_result(address)

    return {
        "ok": True,
        "needs_confirmation": True,
        "original_address": address,
        "suggested_address": resolved.chosen.formatted,
        "alternatives": [c.formatted for c in resolved.alternatives],
        "rationale": resolved.rationale,
        "message": (
            f"I couldn't find '{address}' exactly. Did you mean "
            f"'{resolved.chosen.formatted}'? Please confirm before I proceed -- "
            f"or pick one of the alternatives, or give a corrected address."
        ),
    }


def _double_check_result(address: str) -> dict:
    """The no-suggestion fallback -- today's 'ask the customer to double-check it'
    behavior, returned when address resolution is off, finds nothing, or fails."""
    return {
        "ok": False,
        "no_suggestions": True,
        "error": (
            f"I couldn't find a close match for '{address}' -- ask the customer to "
            f"double-check it, or provide a more complete address (street, city, "
            f"state, ZIP)."
        ),
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


def _decide_and_store(tool_context: ToolContext, profile: dict) -> dict:
    """Run the geo -> evaluate -> decide core for a profile already on file and
    store the result, returning the agent-facing decision dict.

    This is the shared body of `recommend_or_escalate` (called as its own final
    step) and `assign_prospect` (the consolidated one-shot tool). Both MUST return
    the identical shape and write the SAME state keys, so the decision -- and the
    per-turn decision snapshot every rendering surface reuses -- lives here exactly
    once. Steps 2-4 are re-derived from the profile as before: cheap and stateless
    (see the module docstring), and geocoding is cached, so re-deriving here costs
    no extra network call."""
    customer = _profile_from_state_dict(profile)
    try:
        candidates = _find_candidates(customer)
    except GeocodingError as exc:
        return _geocoding_error_result(exc)
    evaluations = evaluate_candidates(customer, candidates, DEFAULT_CONFIG)
    # Step 5: one decision over the (route, slot) options (see `routeslot`).
    # Whether an LLM reasons over them is internal to that layer, and it falls
    # back to the deterministic threshold decision on any error -- so this stays
    # safe when the model/credentials are unavailable.
    from smart_assignment.routeslot import decide_route_slot

    rec = decide_route_slot(customer, evaluations, DEFAULT_CONFIG)

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
        "recommended_window_rationale": rec.recommended_window_rationale,
        "reasoning": rec.reasoning,
        # Structured grounded explanation of a route-slot pick (see the `routeslot`
        # package). None/empty unless the grounded route-slot decision succeeded.
        "decision_summary": rec.decision_summary,
        "primary_reasons": rec.primary_reasons,
        "key_tradeoff": rec.key_tradeoff,
        "runner_up": rec.runner_up,
        "default_comparison": rec.default_comparison,
        "rejected_alternatives": rec.rejected_alternatives,
        "review_reason": rec.review_reason,
        # Split model opinions from grounded-judgment resampling (empty on the
        # weighted path). Surfaced so the escalation-triage sub-agent can show
        # the specialist where the automated judgment was divided.
        "alternative_takes": rec.alternative_takes,
    }
    tool_context.state[_STATE_LAST_RECOMMENDATION_KEY] = result
    # Snapshot the decision against the exact profile that produced it, so a
    # surface rendering the same turn reuses THIS decision instead of sampling a
    # second, possibly different one (see webapp/llm_chat).
    tool_context.state[_STATE_LAST_DECISION_KEY] = {
        "profile": dict(profile),
        "recommendation": rec.to_state_dict(),
    }
    return result


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
       chosen), "reasoning", "rejected_alternatives", "review_reason", and -- on a
       route-slot RECOMMENDED pick -- the structured explanation the agent should
       present: "decision_summary", "primary_reasons", "key_tradeoff",
       "runner_up", "default_comparison"}
      or {"ok": false, "error": "..."}.
    """
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    if not profile:
        return _error("Call intake_customer first -- there's no address on file yet.")
    return _decide_and_store(tool_context, profile)


# --- Consolidated one-shot: intake + decide in a single tool call -----------


def assign_prospect(
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
    Assign a prospect end-to-end in ONE call: record/merge intake, then geocode,
    score, and produce the final recommendation or escalation.

    This is the consolidated equivalent of calling intake_customer ->
    find_candidate_routes -> evaluate_and_score_routes -> recommend_or_escalate in
    sequence, collapsed into a single tool call. It exists for non-interactive
    surfaces (a prospect pulled from Salesforce with a complete profile) where the
    step-by-step narration is not needed and fewer model round-trips matter. It
    changes NO decision logic -- it reuses intake_customer and the exact same
    geo/evaluate/decide core (`_decide_and_store`) the step-by-step tools use, and
    writes the identical state, including the per-turn decision snapshot.

    Intake fields are optional: pass them to record the prospect in this same call,
    or omit them when the profile is already on file (e.g. seeded into session
    state before the turn). Anything already on file is kept and merged, exactly
    like intake_customer.

    Returns:
      The same shape as recommend_or_escalate on success; or, when intake is
      incomplete/invalid, intake_customer's {"ok": false, "error": "..."}; or a
      geocoding {"ok": false, "error": "..."} if the address can't be resolved.
    """
    intake_result = intake_customer(
        tool_context,
        address=address,
        order_quantity_cases=order_quantity_cases,
        preferred_day=preferred_day,
        preferred_window_start=preferred_window_start,
        preferred_window_end=preferred_window_end,
        customer_number=customer_number,
        name=name,
        clear_preferred_slot=clear_preferred_slot,
    )
    if not intake_result.get("ok"):
        # Intake couldn't complete (missing/invalid field) -- relay it unchanged,
        # exactly as if intake_customer had been called on its own.
        return intake_result
    # intake_customer just normalized and stored the profile; decide from it.
    profile = tool_context.state.get(_STATE_PROFILE_KEY)
    return _decide_and_store(tool_context, profile)


def cached_decision_for(state: dict, profile: dict) -> Optional[SlotRecommendation]:
    """The decision `recommend_or_escalate` already made for exactly this profile,
    or ``None`` if there isn't one.

    Returns ``None`` whenever anything is off -- no snapshot, a snapshot for a
    different profile (the prospect was revised mid-conversation), or a snapshot
    that can't be parsed. Every such case means "recompute", which is always
    correct: the caller then simply runs the decision once itself. That keeps the
    reuse safe for every config, including the fully deterministic one where the
    recomputed answer would have matched anyway."""
    snapshot = state.get(_STATE_LAST_DECISION_KEY)
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("profile") != profile:
        return None  # stale: this decision belongs to a different prospect
    try:
        return SlotRecommendation.from_state_dict(snapshot["recommendation"])
    except (KeyError, TypeError, ValueError) as exc:  # corrupt/older snapshot
        logger.warning("Ignoring unreadable cached decision (%s); recomputing.", exc)
        return None
