"""
One-shot, non-conversational entry point: *a prospect record in, a decision out.*

`pipeline.run_slot_recommendation` has always been the agent-free way to run this
workflow (it is what `scripts/run_local.py`, the page generator, the deterministic
web brain, and the eval scorer already drive). What it lacked was a **named,
structured front door**: callers had to build a `CustomerProfile` themselves, pick
a reasoner, and serialize a `RecommendationResult` by hand -- so every integration
re-invented that glue, and there was no way to ask for a *cheaper* run.

This module is that front door, and it is deliberately thin:

    assign(address="...", order_quantity_cases=90, profile="economy") -> dict
    assign_batch([{...}, {...}], profile="balanced")                  -> list[dict]

Why this is a facade and NOT a second implementation
----------------------------------------------------
Every number still comes from `pipeline.run_slot_recommendation`. Nothing here
scores, ranks, or decides -- it builds the profile, picks a `Config`, calls the
one pipeline, and serializes. So the "fast path" and the conversational agent can
never disagree about an answer; they differ only in *who orchestrates the steps*
and *how much LLM reasoning is switched on*.

Cost profiles (the actual latency/cost lever)
---------------------------------------------
Where the time and money go is not the call shape -- it is which LLM decision
layers are active. A `profile` selects that, per call:

| profile    | LLM calls / prospect | what runs                                    |
|------------|----------------------|----------------------------------------------|
| `economy`  | **0**                | pure deterministic pipeline (the default)    |
| `balanced` | ~1                   | one grounded route-slot decision, no resample|
| `full`     | 1-5                  | whatever the environment configures          |

`economy` is not a degraded mode: it is *exactly* the deterministic result every
LLM layer in this repo already guarantees as its fallback floor (see
`docs/architecture/README.md`). Invoking it directly just skips the calls that
would have fallen back to it anyway -- so it needs no separate verification story,
and it is the safe default here.

Because the profile is a **per-call argument** (and an explicit `config=` always
wins), one process can serve an interactive agent on the full configuration and a
batch API on `economy` at the same time. That is the reason this is a function
with an injectable `Config` rather than a deployment-level switch.

Import weight
-------------
This module deliberately imports **no Google ADK** -- not `agent`, not `tools`
(whose `ToolContext` import would pull ADK in). A service that only needs
decisions should not pay the agent stack's import cost. `tests/test_runtime.py`
asserts that property so it cannot silently regress.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Optional

from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.geo import AddressNotFoundError, Geocoder, GeocodingError
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    DayOfWeek,
    PreferredSlot,
    RecommendationResult,
    Route,
)
from smart_assignment.shared.timeutils import fmt_window, parse_time

# The cost profiles callers select by name (see `resolve_config`).
PROFILE_ECONOMY = "economy"
PROFILE_BALANCED = "balanced"
PROFILE_FULL = "full"
PROFILES = (PROFILE_ECONOMY, PROFILE_BALANCED, PROFILE_FULL)

DEFAULT_PROFILE = PROFILE_ECONOMY

# Machine-readable failure kinds on the `{"ok": False}` shape, so a caller can
# branch on the cause without string-matching the human-facing message.
ERROR_INVALID_INTAKE = "invalid_intake"
ERROR_ADDRESS_NOT_FOUND = "address_not_found"
ERROR_GEOCODER_UNAVAILABLE = "geocoder_unavailable"


# ---------------------------------------------------------------------------
# Cost profiles
# ---------------------------------------------------------------------------


def economy_config(base: Optional[Config] = None) -> Config:
    """Every LLM decision layer off: the deterministic pipeline, and nothing else.

    This reproduces the exact result each grounded layer falls back to on a
    backend/verification failure, so it is the repo's documented floor rather than
    a new behavior -- it simply never makes the call in the first place. Zero LLM
    calls, no credentials needed, and the lowest latency this workflow can have.
    """
    base = base or DEFAULT_CONFIG
    return replace(
        base,
        use_route_slot_scoring=False,
        use_grounded_route_slot_escalation=False,
        use_grounded_judgment=False,
        use_grounded_slot_selection=False,
        use_escalation_triage=False,
        use_address_resolution=False,
    )


def balanced_config(base: Optional[Config] = None) -> Config:
    """One grounded call, no resampling: the (route, slot) decision reasons over
    the enumerated menu, but an escalation-side case is *not* resampled k times.

    Keeps the part that changes decision quality (grounded reasoning + the
    structured explanation over route-slot pairs) and drops the parts that
    multiply cost without changing the shape of the answer -- the k-sample
    consensus, the triage brief, and LLM narration. The deterministic threshold
    decision remains the fallback, exactly as on the full path.
    """
    base = base or DEFAULT_CONFIG
    return replace(
        base,
        use_route_slot_scoring=True,
        use_grounded_route_slot_escalation=True,
        judgment_sample_count=1,
        use_grounded_judgment=False,
        use_grounded_slot_selection=False,
        use_escalation_triage=False,
    )


def resolve_config(profile: str = DEFAULT_PROFILE, base: Optional[Config] = None) -> Config:
    """The `Config` for a named cost profile. ``full`` returns the environment's
    own configuration unchanged, so it behaves exactly like every other surface."""
    key = (profile or DEFAULT_PROFILE).strip().lower()
    if key == PROFILE_ECONOMY:
        return economy_config(base)
    if key == PROFILE_BALANCED:
        return balanced_config(base)
    if key == PROFILE_FULL:
        return base or DEFAULT_CONFIG
    raise ValueError(f"unknown profile {profile!r}; expected one of {', '.join(PROFILES)}")


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------


def build_profile(
    address: str,
    order_quantity_cases: int,
    *,
    preferred_day: Optional[str] = None,
    preferred_window_start: Optional[str] = None,
    preferred_window_end: Optional[str] = None,
    name: Optional[str] = None,
    customer_number: Optional[str] = None,
) -> CustomerProfile:
    """A `CustomerProfile` from plain primitives (the shape an API/CLI caller has).

    Raises `ValueError` with a caller-facing message on anything malformed -- a
    partial preferred slot, an unknown day, an unparseable time. Validation of the
    *business* rules (address required, positive order quantity, customer-number
    format) stays in `pipeline.intake`, which runs downstream; this only converts.
    """
    slot_parts = (preferred_day, preferred_window_start, preferred_window_end)
    slot: Optional[PreferredSlot] = None
    if any(slot_parts):
        if not all(slot_parts):
            raise ValueError(
                "A preferred delivery slot needs a day AND both a start and end "
                "time -- supply all three, or none."
            )
        try:
            day = DayOfWeek(str(preferred_day).strip().upper())
        except ValueError:
            raise ValueError(
                f"Unknown preferred_day {preferred_day!r}; expected one of "
                f"{', '.join(d.value for d in DayOfWeek)}."
            ) from None
        slot = PreferredSlot(
            day,
            (parse_time(str(preferred_window_start)), parse_time(str(preferred_window_end))),
        )

    return CustomerProfile(
        name=name or "New prospect",
        address=address or "",
        order_quantity_cases=order_quantity_cases or 0,
        customer_number=customer_number,
        preferred_slot=slot,
    )


# ---------------------------------------------------------------------------
# Serialization (the API contract)
# ---------------------------------------------------------------------------
#
# Deliberately its own shape rather than the conversational tool's: the tool
# result is written for an LLM to narrate (it carries the full candidate slot
# menu and factor detail), while this is a stable contract for a programmatic
# caller. Keeping them separate means the API shape is not hostage to prompt
# tuning, and it is what lets this module stay free of the ADK import that
# `tools/slot_recommendation.py` carries.


def _serialize_customer(customer: CustomerProfile) -> dict:
    slot = customer.preferred_slot
    out: dict[str, Any] = {
        "name": customer.name,
        "address": customer.address,
        "order_quantity_cases": customer.order_quantity_cases,
        "customer_number": customer.customer_number,
        "preferred_day": slot.day.value if slot else None,
        "preferred_window": fmt_window(slot.window) if slot else None,
    }
    if customer.location is not None:
        out["latitude"] = customer.location.latitude
        out["longitude"] = customer.location.longitude
    return out


def _serialize_candidate(evaluation: CandidateEvaluation) -> dict:
    out: dict[str, Any] = {
        "route_id": evaluation.route.route_id,
        "name": evaluation.route.name,
        "day": evaluation.route.day.value,
        "distance_miles": round(evaluation.distance_miles, 1),
        "feasible": evaluation.feasible,
        "utilization_after": round(evaluation.utilization_after, 4),
        "chosen_window": fmt_window(evaluation.chosen_window),
        "window_basis": evaluation.window_basis,
        "constraints": [
            {
                "name": CONSTRAINT_LABEL.get(c.name, c.name),
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in evaluation.constraint_outcomes
        ],
    }
    if evaluation.feasible:
        out["total_score"] = round(evaluation.total_score, 4)
        out["factor_scores"] = [
            {"name": f.name, "weight": f.weight, "value": round(f.value, 4), "detail": f.detail}
            for f in evaluation.factor_scores
        ]
    return out


def serialize_result(result: RecommendationResult, profile: str) -> dict:
    """The full, JSON-safe decision payload -- the answer plus the audit trail
    (every candidate considered, its constraint outcomes, and its score)."""
    rec = result.recommendation
    return {
        "ok": True,
        "profile": profile,
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
        # Structured grounded explanation; empty on the deterministic path.
        "decision_summary": rec.decision_summary,
        "primary_reasons": list(rec.primary_reasons),
        "key_tradeoff": rec.key_tradeoff,
        "runner_up": rec.runner_up,
        "default_comparison": rec.default_comparison,
        "rejected_alternatives": list(rec.rejected_alternatives),
        "review_reason": rec.review_reason,
        "alternative_takes": list(rec.alternative_takes),
        # True when a grounded layer was asked for but fell back deterministically,
        # so a caller can tell "reasoned" from "fell back" without reading logs.
        "grounded_fallback": rec.grounded_fallback,
        "grounded_fallback_reason": rec.grounded_fallback_reason,
        "customer": _serialize_customer(result.customer),
        "candidates_considered": [_serialize_candidate(e) for e in result.candidates_considered],
    }


def _error(message: str, kind: str) -> dict:
    return {"ok": False, "error": message, "error_kind": kind}


def _geocoding_error(exc: GeocodingError) -> dict:
    if isinstance(exc, AddressNotFoundError):
        return _error(
            f"Could not find a location for '{exc.address}' -- check the address, "
            f"or supply a more complete one (street, city, state, ZIP).",
            ERROR_ADDRESS_NOT_FOUND,
        )
    return _error(
        "The geocoding service is temporarily unavailable -- retry in a moment.",
        ERROR_GEOCODER_UNAVAILABLE,
    )


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def assign(
    address: str,
    order_quantity_cases: int,
    *,
    preferred_day: Optional[str] = None,
    preferred_window_start: Optional[str] = None,
    preferred_window_end: Optional[str] = None,
    name: Optional[str] = None,
    customer_number: Optional[str] = None,
    profile: str = DEFAULT_PROFILE,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
    routes: Optional[list[Route]] = None,
) -> dict:
    """Run the full workflow for one prospect and return a JSON-safe decision.

    Args:
      address, order_quantity_cases: the required intake (address is the primary
        identifier for a prospect; see `pipeline.intake`).
      preferred_day / preferred_window_start / preferred_window_end: an optional
        soft preference -- supply all three or none. Times are 24-hour "HH:MM".
      name, customer_number: optional descriptive / existing-account fields.
      profile: cost profile -- "economy" (default, 0 LLM calls), "balanced"
        (~1), or "full" (whatever the environment configures).
      config: an explicit `Config` ALWAYS wins over `profile`, so a caller with
        its own tuned configuration is never overridden.
      geocoder, routes: injectable collaborators (tests, batch reuse, replay).

    Returns:
      On success the payload from `serialize_result`. On failure
      ``{"ok": False, "error": "<caller-facing message>", "error_kind": "..."}``
      -- an invalid intake, an unfindable address, or an unavailable geocoder.
      Never raises for those cases: this is an API boundary, so an expected
      failure is a value, not an exception.
    """
    effective = config if config is not None else resolve_config(profile)

    try:
        customer = build_profile(
            address,
            order_quantity_cases,
            preferred_day=preferred_day,
            preferred_window_start=preferred_window_start,
            preferred_window_end=preferred_window_end,
            name=name,
            customer_number=customer_number,
        )
    except ValueError as exc:
        return _error(str(exc), ERROR_INVALID_INTAKE)

    kwargs: dict[str, Any] = {"config": effective}
    if geocoder is not None:
        kwargs["geocoder"] = geocoder
    if routes is not None:
        kwargs["routes"] = routes
    # `full` keeps the pipeline's own default reasoner (LLM-backed, with its own
    # deterministic fallback); the cost profiles narrate deterministically.
    if config is None and profile.strip().lower() != PROFILE_FULL:
        kwargs["reasoner"] = DeterministicReasoner()

    try:
        result = run_slot_recommendation(customer, **kwargs)
    except GeocodingError as exc:
        return _geocoding_error(exc)
    except ValueError as exc:
        # `pipeline.intake` rejected the profile (empty address, non-positive
        # order quantity, malformed customer number).
        return _error(str(exc), ERROR_INVALID_INTAKE)

    return serialize_result(result, profile if config is None else PROFILE_FULL)


def assign_batch(
    prospects: Iterable[dict],
    *,
    profile: str = DEFAULT_PROFILE,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
    routes: Optional[list[Route]] = None,
) -> list[dict]:
    """Run `assign` over many prospects, fetching the route world **once**.

    Each item is a dict of `assign`'s keyword arguments (``address`` and
    ``order_quantity_cases`` required). One prospect's failure is returned as its
    own ``{"ok": False, ...}`` entry rather than aborting the batch, so a bad row
    never costs you the other results. Order is preserved.
    """
    shared_routes = routes if routes is not None else fetch_candidate_routes()
    results: list[dict] = []
    for raw in prospects:
        fields = dict(raw)
        address = fields.pop("address", "")
        cases = fields.pop("order_quantity_cases", 0)
        # A caller-supplied per-item profile/config would defeat the shared route
        # fetch's assumptions about a single world; the batch settings govern.
        fields.pop("profile", None)
        fields.pop("config", None)
        try:
            results.append(
                assign(
                    address,
                    cases,
                    profile=profile,
                    config=config,
                    geocoder=geocoder,
                    routes=shared_routes,
                    **fields,
                )
            )
        except TypeError as exc:
            results.append(_error(f"Unrecognized prospect field: {exc}", ERROR_INVALID_INTAKE))
    return results


__all__ = [
    "assign",
    "assign_batch",
    "build_profile",
    "serialize_result",
    "resolve_config",
    "economy_config",
    "balanced_config",
    "PROFILES",
    "PROFILE_ECONOMY",
    "PROFILE_BALANCED",
    "PROFILE_FULL",
    "DEFAULT_PROFILE",
]
