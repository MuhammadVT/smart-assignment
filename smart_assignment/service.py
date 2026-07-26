"""
The headless decision service — one prospect in, one final decision out.

This is the non-conversational way to run the workflow, for a production caller
(a Salesforce-driven sync, a batch job) that has complete prospect details already
and needs the decision rendered on a sales consultant's page. It is *not* a second
pipeline: it builds a `CustomerProfile`, calls `pipeline.run_slot_recommendation`,
and serializes the result. Steps 1-4 stay plain deterministic Python and step 5 is
the same `routeslot.decide_route_slot` the conversational agent drives, under the
same `Config` — so the recommend/escalate call, the grounded reasoning, the
verifier, the resampling, and the deterministic fallback all behave identically.

What is deliberately absent, because there is nobody to talk to:

* **No LLM orchestration.** The conversational agent spends a model round-trip per
  workflow step deciding which tool to call next; here the order is fixed, so
  plain Python calls them. The LLM is only ever a leaf — it reasons over the
  enumerated (route, slot) options and returns a verified choice.
* **No address confirmation.** A production record is trusted and complete, and
  there is no user to confirm a correction with, so a geocoding failure is
  reported rather than repaired (see `AssignmentOutcome.error_kind`).
* **No escalation brief.** An escalation returns its structured facts; composing
  the specialist brief is a separate, on-demand step so that nothing open-ended
  sits on the decision's critical path.

Event-loop ownership (why `_llm_host_loop` exists)
--------------------------------------------------
`shared.llm.generate_text` is synchronous over an async backend. Called from
ordinary synchronous code it takes `_run_coro_blocking`'s "no loop here" path,
which runs `asyncio.run` — creating a fresh event loop per call and **closing** it
afterwards. That is fine once, but the sage backend caches an aiohttp session
process-wide, bound to the first loop that touched it; the second call then fails
with `RuntimeError: Event loop is closed`. A batch is exactly that pattern N times.

So the service keeps **one** long-lived event loop for the process and records it
as the host loop, which makes every grounded call submit its coroutine back to
that single loop. It never overrides a host loop a caller has already established
(the web app records uvicorn's loop via `offload_to_worker_thread`), so running
inside an async server keeps working unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from smart_assignment.integrations.geocoding_client import resolve_geocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.geo import (
    AddressNotFoundError,
    Geocoder,
    GeocodingError,
)
from smart_assignment.shared.llm import _HOST_EVENT_LOOP
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    DayOfWeek,
    PreferredSlot,
    RecommendationResult,
    Route,
)
from smart_assignment.shared.timeutils import fmt_time, fmt_window, parse_time

logger = logging.getLogger(__name__)

__all__ = [
    "AssignmentOutcome",
    "assign",
    "assign_many",
    "from_salesforce_record",
]


# ---------------------------------------------------------------------------
# One event loop per process, for the grounded calls
# ---------------------------------------------------------------------------

_LOOP_LOCK = threading.Lock()
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _background_loop() -> asyncio.AbstractEventLoop:
    """The process-wide loop grounded calls run on, started on first use.

    A daemon thread runs it forever, so it outlives any individual call and the
    backend's cached session stays valid for the life of the process. Idempotent:
    repeated calls return the same loop.
    """
    global _BACKGROUND_LOOP
    with _LOOP_LOCK:
        if _BACKGROUND_LOOP is not None and not _BACKGROUND_LOOP.is_closed():
            return _BACKGROUND_LOOP
        loop = asyncio.new_event_loop()
        threading.Thread(
            target=loop.run_forever,
            name="smart-assignment-llm-loop",
            daemon=True,
        ).start()
        _BACKGROUND_LOOP = loop
        return loop


@contextlib.contextmanager
def _llm_host_loop():
    """Point grounded calls at the process-wide loop for the duration of the block.

    Defers entirely to a host loop the caller has already established — the web
    app records uvicorn's server loop, and the backend's session is bound *there*,
    so overriding it would break the very thing this exists to protect.
    """
    existing = _HOST_EVENT_LOOP.get()
    if existing is not None and existing.is_running():
        yield  # somebody upstream owns the loop; leave it alone
        return
    token = _HOST_EVENT_LOOP.set(_background_loop())
    try:
        yield
    finally:
        _HOST_EVENT_LOOP.reset(token)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

# error_kind values, so a caller can branch without parsing the message.
ERROR_INTAKE = "intake"  # the record itself is invalid (no address, zero cases)
ERROR_ADDRESS_NOT_FOUND = "address_not_found"  # geocoder ran, found nothing
ERROR_GEOCODER_UNAVAILABLE = "geocoder_unavailable"  # transport/service problem
ERROR_INTERNAL = "internal"  # anything unexpected; logged with a traceback


@dataclass
class AssignmentOutcome:
    """One prospect's result: the decision, the evidence behind it, or why neither.

    `to_dict()` is the JSON-safe wire form. `result` is the in-process trace, kept
    off the wire form but available so a caller can render it with
    `reporting.page.build_workflow_payload` without re-running anything.
    """

    ok: bool
    customer: dict
    decision: Optional[dict] = None
    candidates: list[dict] = field(default_factory=list)
    # The specialist brief, when one was asked for and could be composed. Absent
    # by default: composing it is deferred so nothing model-driven sits on a
    # decision's critical path (see `assign`'s `include_brief`).
    brief: Optional[str] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    # Not serialized: the full RecommendationResult, for in-process rendering.
    result: Optional[RecommendationResult] = field(
        default=None, repr=False, compare=False
    )

    @property
    def requires_human_review(self) -> bool:
        """True when a specialist must look at this — including a failed run,
        which is never something to auto-assign on."""
        if not self.ok or self.result is None:
            return True
        return self.result.recommendation.requires_human_review

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "customer": self.customer,
            "requires_human_review": self.requires_human_review,
        }
        if self.ok:
            payload["decision"] = self.decision
            payload["candidates"] = self.candidates
            if self.brief is not None:
                payload["brief"] = self.brief
        else:
            payload["error"] = self.error
            payload["error_kind"] = self.error_kind
        return payload


# ---------------------------------------------------------------------------
# Input adapter
# ---------------------------------------------------------------------------

def from_salesforce_record(record: Mapping[str, Any]) -> CustomerProfile:
    """Build a `CustomerProfile` from a flat prospect record.

    Kept separate from `assign` on purpose: mapping a CRM's field names is the one
    part of this that will change when the upstream system does, so it lives in a
    single named place rather than in the signature of the decision call.

    Recognized keys (unknown keys are ignored, so an upstream payload can carry
    extra columns): ``address`` and ``order_quantity_cases`` are required;
    ``name``, ``customer_number`` and the optional preferred slot
    (``preferred_day`` plus ``preferred_window_start``/``preferred_window_end``,
    all three or none) are optional.

    Raises `ValueError` with a message naming the offending field — the caller
    turns that into a per-record failure rather than a crashed batch.
    """
    address = (record.get("address") or "").strip()
    if not address:
        raise ValueError("address is required (it is the prospect's primary identifier)")

    raw_cases = record.get("order_quantity_cases")
    if raw_cases is None or (isinstance(raw_cases, str) and not raw_cases.strip()):
        raise ValueError("order_quantity_cases is required")
    try:
        cases = int(raw_cases)
    except (TypeError, ValueError):
        raise ValueError(
            f"order_quantity_cases must be a whole number, got {raw_cases!r}"
        ) from None

    slot = _slot_from_record(record)

    return CustomerProfile(
        name=(record.get("name") or "").strip() or "New prospect",
        address=address,
        order_quantity_cases=cases,
        customer_number=(record.get("customer_number") or None),
        preferred_slot=slot,
    )


def _slot_from_record(record: Mapping[str, Any]) -> Optional[PreferredSlot]:
    """The optional preferred slot: all three parts, or none of them.

    A day without a window (or vice versa) is rejected rather than half-honored —
    a partially stated preference would silently score as no preference at all,
    which is exactly the kind of quiet wrong answer this codebase avoids.
    """
    day = (record.get("preferred_day") or "").strip().upper()
    start = (record.get("preferred_window_start") or "").strip()
    end = (record.get("preferred_window_end") or "").strip()

    if not any((day, start, end)):
        return None
    if not all((day, start, end)):
        raise ValueError(
            "a preferred slot needs preferred_day, preferred_window_start AND "
            "preferred_window_end -- supply all three, or none"
        )

    try:
        day_value = DayOfWeek(day)
    except ValueError:
        valid = ", ".join(d.value for d in DayOfWeek)
        raise ValueError(f"preferred_day must be one of {valid}, got {day!r}") from None

    try:
        window = (parse_time(start), parse_time(end))
    except (ValueError, IndexError):
        raise ValueError(
            f"preferred window times must be 24-hour \"HH:MM\", got {start!r} and {end!r}"
        ) from None

    return PreferredSlot(day_value, window)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _customer_dict(customer: CustomerProfile) -> dict:
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
        out["location"] = {
            "latitude": customer.location.latitude,
            "longitude": customer.location.longitude,
        }
    return out


def _evaluation_dict(evaluation: CandidateEvaluation) -> dict:
    """One candidate route's evidence, in the same shape the conversational tool
    reports it — including the rule that a rejected route carries **no** merit
    score (see `pipeline._apply_route_slot_scores`): a number beside a rejection
    invites "it scored well, why wasn't it used?"."""
    out: dict[str, Any] = {
        "route_id": evaluation.route.route_id,
        "route_name": evaluation.route.name,
        "day": evaluation.route.day.value,
        "distance_miles": round(evaluation.distance_miles, 1),
        "feasible": evaluation.feasible,
        "utilization_after": round(evaluation.utilization_after, 4),
        "remaining_capacity_after": evaluation.remaining_capacity_after,
        "constraints": [
            {
                "name": CONSTRAINT_LABEL.get(c.name, c.name),
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in evaluation.constraint_outcomes
        ],
        "chosen_window": fmt_window(evaluation.chosen_window),
        "window_basis": evaluation.window_basis,
        "available_slots": [
            {
                "window": fmt_window(s.window),
                "anchor_time": fmt_time(s.anchor_time) if s.anchor_time else None,
                "fit_score": round(s.fit_score, 4),
                "committed_overlap": s.committed_overlap,
                "basis": s.basis,
            }
            for s in evaluation.available_slots
        ],
    }
    if evaluation.feasible:
        out["total_score"] = round(evaluation.total_score, 4)
        out["factor_scores"] = [
            {"name": f.name, "weight": f.weight, "value": round(f.value, 4), "detail": f.detail}
            for f in evaluation.factor_scores
        ]
    return out


def _failure(customer: CustomerProfile, kind: str, message: str) -> AssignmentOutcome:
    return AssignmentOutcome(
        ok=False,
        customer=_customer_dict(customer),
        error=message,
        error_kind=kind,
    )


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def assign(
    customer: CustomerProfile,
    *,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
    routes: Optional[list[Route]] = None,
    include_brief: bool = False,
) -> AssignmentOutcome:
    """Decide one prospect's route and slot, or escalate.

    Returns an `AssignmentOutcome` in every case — it does not raise for a bad
    record, an unresolvable address, or an unexpected internal error, because a
    caller processing a queue must be able to record the failure and keep going.
    An internal error is logged with its traceback and reported as
    `error_kind="internal"`, so a genuine bug is loud in the logs while still
    being a single bad row rather than a stopped run.

    `config` defaults to `DEFAULT_CONFIG`, so the same environment that tunes the
    conversational agent tunes this. Pass one explicitly to run a different
    configuration in the same process (a deterministic batch alongside a grounded
    interactive surface, say). `routes` lets a batch fetch the route world once
    and share it; `geocoder` is injectable for offline runs and tests.

    `include_brief` composes the specialist brief inline on an escalation. It is
    **off by default on purpose**: the brief is only read when a specialist opens
    the escalation, and composing it inline puts the one open-ended, model-driven
    step in the system on the decision's critical path — where a slow or looping
    agent would delay, and a failing one could fail, a decision nobody may even
    look at. Prefer calling `triage.headless.compose_brief` on demand instead, and
    reserve this for a batch that must emit complete records with no follow-up
    call. Either way the decision is unaffected: a brief that cannot be composed
    is simply absent, and the structured escalation facts are already present.
    """
    config = config or DEFAULT_CONFIG
    try:
        with _llm_host_loop():
            result = run_slot_recommendation(
                customer, routes=routes, config=config, geocoder=geocoder
            )
    except ValueError as exc:
        # Raised by pipeline.intake for a structurally invalid profile.
        return _failure(customer, ERROR_INTAKE, str(exc))
    except AddressNotFoundError as exc:
        # Not transient: no amount of retrying fixes an address that doesn't
        # resolve. There is no user here to confirm a correction with, so this is
        # reported for a human to fix upstream.
        return _failure(customer, ERROR_ADDRESS_NOT_FOUND, str(exc))
    except GeocodingError as exc:
        # Transport/service problem -- a caller may reasonably retry this record.
        return _failure(customer, ERROR_GEOCODER_UNAVAILABLE, str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad record must not stop a batch
        logger.exception("Assignment failed for %s", customer.address)
        return _failure(customer, ERROR_INTERNAL, f"{type(exc).__name__}: {exc}")

    brief = None
    if include_brief and config.use_escalation_triage:
        if result.recommendation.requires_human_review:
            # Returns None on any failure, so a brief never gates the decision.
            from smart_assignment.triage.headless import compose_brief

            with _llm_host_loop():
                brief = compose_brief(result.customer, result.recommendation, config)

    return AssignmentOutcome(
        ok=True,
        customer=_customer_dict(result.customer),
        decision=result.recommendation.to_state_dict(),
        candidates=[_evaluation_dict(e) for e in result.candidates_considered],
        brief=brief,
        result=result,
    )


def assign_many(
    customers: Iterable[CustomerProfile],
    *,
    config: Optional[Config] = None,
    geocoder: Optional[Geocoder] = None,
    routes: Optional[list[Route]] = None,
    include_brief: bool = False,
) -> list[AssignmentOutcome]:
    """Decide a batch, one outcome per prospect, in input order.

    Two things it does that repeated `assign` calls would not:

    * **Fetches the route world once** and shares it across the batch, instead of
      once per prospect.
    * **Holds a single event loop open** for the whole run, so the backend's
      cached session survives from the first prospect to the last.

    A prospect that fails yields a failed `AssignmentOutcome` and the run
    continues. A failure to load the routes themselves is *not* a per-prospect
    problem and propagates, because a batch scored against no route world is
    invalid rather than partially useful.
    """
    config = config or DEFAULT_CONFIG
    geocoder = geocoder or resolve_geocoder()
    if routes is None:
        routes = fetch_candidate_routes()

    with _llm_host_loop():
        return [
            assign(
                customer,
                config=config,
                geocoder=geocoder,
                routes=routes,
                include_brief=include_brief,
            )
            for customer in customers
        ]
