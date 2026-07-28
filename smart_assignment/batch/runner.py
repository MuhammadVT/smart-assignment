"""
The batch runner: process a source of prospects through the deterministic
pipeline once each, and emit one result per prospect.

This is the whole point of batch mode -- the same ``run_slot_recommendation`` the
chat and web app already use, but driven unattended over many prospects and with
the two human-in-the-loop steps replaced by deterministic policies:

  * **Address (trust Salesforce as-is).** The Salesforce address is authoritative.
    If it can't be geocoded (or intake rejects the record), the prospect becomes a
    ``needs_attention`` record for a human to fix the source data -- batch never
    resolves or guesses an address (no ``address_resolve`` here), so the "never
    fabricate an actionable value" guarantee holds trivially.
  * **Escalation (record, not a chat pause).** There is no ``request_input``: an
    escalation is written as a record carrying the triage brief (composed WITHOUT
    the ADK agent, see ``triage.compose_brief``), which the SC reviews in the
    Customer View.

Nothing here changes the decision: hard constraints and scoring are the pipeline's,
and a per-prospect failure degrades to ``needs_attention`` -- it never aborts the
batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    BatchRecord,
    ResultSink,
)
from smart_assignment.batch.source import Prospect, ProspectSource
from smart_assignment.integrations.geocoding_client import resolve_geocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.geo import AddressNotFoundError, Geocoder, GeocodingError
from smart_assignment.shared.models import Route
from smart_assignment.triage import compose_brief, escalation_context_from_recommendation

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    """Per-outcome counts for one batch run (for the CLI and for tests)."""

    total: int
    recommend: int
    escalate: int
    needs_attention: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one(
    prospect: Prospect,
    config: Config,
    geocoder: Geocoder,
    routes: list[Route],
    generated_at: str,
) -> BatchRecord:
    """Run the pipeline once for one prospect and map the outcome to a BatchRecord.

    Known failures (a bad/unresolvable address, an invalid intake) become a
    ``needs_attention`` record per the address policy above; a clean decision becomes
    ``recommend``; an escalation becomes ``escalate`` with a (grounded or
    deterministic-floor) brief attached."""
    try:
        result = run_slot_recommendation(
            prospect.profile, routes=routes, config=config, geocoder=geocoder
        )
    except AddressNotFoundError as exc:
        return _needs_attention(prospect, generated_at, f"address not found: {exc.address}")
    except GeocodingError:
        return _needs_attention(prospect, generated_at, "geocoding service unavailable")
    except ValueError as exc:
        return _needs_attention(prospect, generated_at, f"invalid intake: {exc}")

    # Imported lazily so importing the batch package stays light (the page renderer
    # pulls in the reporting stack) -- the import is cached after the first prospect.
    from smart_assignment.reporting.page import build_workflow_payload

    payload = build_workflow_payload(result, config)
    rec = result.recommendation
    if not rec.requires_human_review:
        return BatchRecord(prospect.prospect_id, generated_at, OUTCOME_RECOMMEND, payload=payload)

    context = escalation_context_from_recommendation(
        result.customer, result.candidates_considered, rec, config
    )
    brief = compose_brief(context, config)
    return BatchRecord(
        prospect.prospect_id,
        generated_at,
        OUTCOME_ESCALATE,
        payload=payload,
        review_reason=rec.review_reason,
        triage_brief=brief or None,
    )


def _needs_attention(prospect: Prospect, generated_at: str, error: str) -> BatchRecord:
    return BatchRecord(
        prospect.prospect_id, generated_at, OUTCOME_NEEDS_ATTENTION, error=error
    )


class BatchRunner:
    """Sequentially runs a ``ProspectSource`` through the pipeline into a
    ``ResultSink``.

    Collaborators are injected (source, sink, geocoder, routes, config, clock) so the
    runner is trivially testable offline and points at real systems by argument, not
    by edit -- the same seam the pipeline itself uses. The routes list and the
    geocoder are resolved ONCE per run and reused for every prospect."""

    def __init__(
        self,
        source: ProspectSource,
        sink: ResultSink,
        config: Optional[Config] = None,
        geocoder: Optional[Geocoder] = None,
        routes: Optional[list[Route]] = None,
        clock: Optional[Callable[[], str]] = None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._config = config or DEFAULT_CONFIG
        self._geocoder = geocoder
        self._routes = routes
        self._clock = clock or _utc_now_iso

    def run(self) -> BatchSummary:
        geocoder = self._geocoder or resolve_geocoder()
        routes = self._routes if self._routes is not None else fetch_candidate_routes()

        counts = {OUTCOME_RECOMMEND: 0, OUTCOME_ESCALATE: 0, OUTCOME_NEEDS_ATTENTION: 0}
        total = 0
        for prospect in self._source.prospects():
            total += 1
            try:
                record = run_one(prospect, self._config, geocoder, routes, self._clock())
            except Exception as exc:  # noqa: BLE001 - one prospect must never abort the batch
                logger.exception(
                    "Batch entry %s failed unexpectedly; recording needs_attention.",
                    prospect.prospect_id,
                )
                record = _needs_attention(prospect, self._clock(), f"unexpected error: {exc}")
            self._sink.emit(record)
            counts[record.outcome] += 1

        return BatchSummary(
            total=total,
            recommend=counts[OUTCOME_RECOMMEND],
            escalate=counts[OUTCOME_ESCALATE],
            needs_attention=counts[OUTCOME_NEEDS_ATTENTION],
        )
