"""
The deterministic per-prospect batch engine, and the shared batch summary type.

``run_one`` runs the deterministic pipeline once for a single prospect and maps the
outcome to a :class:`BatchRecord`. It is the low-latency, credential-free, no-LLM
path -- and the **deterministic floor** the agent batch runner
(:class:`~smart_assignment.batch.agent_runner.AgentBatchRunner`) falls back to
whenever the agent is unavailable (no credentials/backend) or a single agent turn
fails. That fallback is what upholds the repo's "never worse than the deterministic
baseline" guarantee for batch mode.

The two human-in-the-loop steps are replaced by deterministic policies:

  * **Address (trust Salesforce as-is).** The Salesforce address is authoritative.
    If it can't be geocoded (or intake rejects the record), the prospect becomes a
    ``needs_attention`` record for a human to fix the source data -- batch never
    resolves or guesses an address, so the "never fabricate an actionable value"
    guarantee holds trivially.
  * **Escalation (record, not a chat pause).** There is no ``request_input``: an
    escalation is written as a record carrying the triage brief composed WITHOUT
    the ADK agent (see ``triage.compose_brief``), which the SC reviews in the
    Customer View.

Nothing here changes the decision: hard constraints and scoring are the pipeline's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    BatchRecord,
)
from smart_assignment.batch.source import Prospect
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import Config
from smart_assignment.shared.geo import AddressNotFoundError, Geocoder, GeocodingError
from smart_assignment.shared.models import Route
from smart_assignment.triage import compose_brief, escalation_context_from_recommendation


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
    """Run the deterministic pipeline once for one prospect and map the outcome to a
    BatchRecord.

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
