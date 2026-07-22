"""
Vendor-neutral observability emit for a feedback record.

Feedback arrives *after* the decision span has already closed (a human clicks
thumbs-down seconds later), and an exported OpenTelemetry span cannot be mutated.
So a record is emitted as its OWN short span, ``human_feedback``, **linked** to
the decision's span via the standard OpenTelemetry span-link mechanism and the
original ``trace_id``. That is the one representation that is truly OTLP-neutral:
any backend that ingests OpenTelemetry -- Phoenix now, Langfuse later, Tempo /
Jaeger / anything after -- receives it and can correlate it to the decision by
trace id, with only the exporter *endpoint* differing. No vendor annotation API
is used, so nothing here couples the repo to a backend.

This reuses the exact exporter/provider seam already built in
``shared.tracing``: ``configure_tracing`` installs the global provider + OTLP
exporter, and this module just starts one more span on it. Every guarantee that
module holds carries over -- opt-in, credential-free import, and a silent no-op
on any failure (SDK missing, tracing off, exporter down), so a broken trace
backend can never break feedback capture. The durable JSONL log
(``store.append_record``) is the source of truth; this emit is the convenience
layer on top.

Span attributes are the non-PII quality signal only (annotator kind, label,
score, decision kind, ids). The freeform ``note`` is deliberately NOT put on the
span -- it can carry customer PII -- matching ``shared.tracing``'s "no prompt/response
text on spans" stance. The note lives only in the durable log, gated by the
``feedback_scrub_pii`` toggle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from smart_assignment.feedback.schema import FeedbackRecord

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)

_ATTR_PREFIX = "smart_assignment.feedback."


def _build_link(trace_id: Optional[str], span_id: Optional[str]) -> Any:
    """A one-element list holding an OpenTelemetry ``Link`` to the decision span,
    or ``None`` when coordinates are absent/invalid or the API is unavailable.

    A span link (rather than a parent) is correct here: the feedback is a
    *separate* event that references the decision, not a child of it -- and the
    decision span is long gone by the time feedback arrives."""
    if not (trace_id and span_id):
        return None
    try:
        from opentelemetry.trace import Link, SpanContext, TraceFlags

        ctx = SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        if not ctx.is_valid:
            return None
        return [Link(ctx)]
    except Exception:  # noqa: BLE001 - no link is fine; the trace_id attr still correlates
        logger.debug("Could not build a feedback span link; emitting unlinked.", exc_info=True)
        return None


def emit_feedback_span(config: "Config", record: FeedbackRecord) -> bool:
    """Emit ``record`` as a vendor-neutral OTLP ``human_feedback`` span linked to
    the decision it annotates. Returns ``True`` if a real span was emitted,
    ``False`` when tracing is off/unavailable (a silent no-op) or on any error.

    Never raises: observability is additive and must not break the feedback
    request. When ``configure_tracing`` returns no tracer (flag off, SDK missing,
    no exporter), this is a transparent no-op and the durable log still holds the
    record."""
    try:
        from smart_assignment.shared.tracing import configure_tracing

        tracer = configure_tracing(config)
        if tracer is None:
            return False

        target = record.target
        link = _build_link(target.trace_id, target.span_id)
        attributes = {
            _ATTR_PREFIX + "annotator_kind": record.annotator_kind,
            _ATTR_PREFIX + "label": record.label,
            _ATTR_PREFIX + "decision_kind": target.decision_kind,
            _ATTR_PREFIX + "decision_id": target.decision_id,
        }
        if record.score is not None:
            attributes[_ATTR_PREFIX + "score"] = float(record.score)
        if target.trace_id:
            attributes[_ATTR_PREFIX + "target_trace_id"] = target.trace_id
        if target.span_id:
            attributes[_ATTR_PREFIX + "target_span_id"] = target.span_id
        if target.session_id:
            attributes[_ATTR_PREFIX + "session_id"] = target.session_id
        if record.annotator_id:
            attributes[_ATTR_PREFIX + "annotator_id"] = record.annotator_id
        # A boolean signal that a note exists, WITHOUT the note text (PII).
        attributes[_ATTR_PREFIX + "has_note"] = bool((record.note or "").strip())

        kwargs = {"attributes": attributes}
        if link is not None:
            kwargs["links"] = link
        with tracer.start_as_current_span("human_feedback", **kwargs):
            pass
        return True
    except Exception:  # noqa: BLE001 - a tracing hiccup must never break capture
        logger.debug("Could not emit a feedback span; continuing.", exc_info=True)
        return False
