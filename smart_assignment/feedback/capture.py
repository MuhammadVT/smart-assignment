"""
The one entry point that records a piece of feedback.

``record_feedback(config, record)`` ties the pieces together in the order the
guarantees demand:

1. **Gate on the flag.** With ``use_human_feedback`` off it is a no-op returning
   a ``disabled`` result -- nothing is validated, written, or imported further.
2. **Validate deterministically.** A malformed record is rejected up front
   (``FeedbackValidationError``) so nothing downstream persists garbage.
3. **Scrub if configured.** When ``feedback_scrub_pii`` is on, the freeform note
   and any free-text context values are redacted *before* they are persisted.
   Off (the trusted-network case), the real PII is retained on purpose.
4. **Persist first (source of truth).** Write the durable JSONL record. This is
   the audit record and the curation input; it must land even if observability
   is down.
5. **Emit best-effort.** Emit the vendor-neutral OTLP span. A failure here is a
   silent no-op -- the annotation is already safely logged.

The function never raises on a persistence/emit problem (those degrade to a
logged ``ok=False``/no-op); it raises only ``FeedbackValidationError`` for a
structurally invalid record, which a caller/endpoint turns into a 4xx.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Dict

from smart_assignment.feedback.schema import FeedbackRecord, validate_feedback

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)


def _scrubbed(record: FeedbackRecord) -> FeedbackRecord:
    """A copy of ``record`` with its note and context PII-scrubbed. Imported
    lazily so the flag-off / scrub-off paths pull in no regex module."""
    from smart_assignment.feedback.scrub import scrub_context, scrub_text

    return replace(
        record,
        note=scrub_text(record.note),
        context=scrub_context(record.context),
    )


def record_feedback(config: "Config", record: FeedbackRecord) -> Dict[str, bool]:
    """Validate, (optionally) scrub, persist, and emit one feedback record.

    Returns a small status dict: ``{"disabled"|"persisted"|"emitted": bool}``.
    Raises ``FeedbackValidationError`` only for a structurally invalid record."""
    if not getattr(config, "use_human_feedback", False):
        return {"disabled": True, "persisted": False, "emitted": False}

    # Deterministic gate: reject bad input before anything acts on it.
    validate_feedback(record)

    to_store = _scrubbed(record) if getattr(config, "feedback_scrub_pii", True) else record

    from smart_assignment.feedback.store import append_record

    path = getattr(config, "feedback_log_path", "feedback_data/annotations.jsonl")
    persisted = append_record(to_store, path)

    # Emit the SCRUBBED view too, so PII never reaches the trace backend when the
    # toggle is on (the span omits note text regardless, but context ids stay
    # consistent with what was persisted).
    from smart_assignment.feedback.emit import emit_feedback_span

    emitted = emit_feedback_span(config, to_store)

    return {"disabled": False, "persisted": persisted, "emitted": emitted}
