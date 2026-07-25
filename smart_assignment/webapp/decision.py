"""
Feedback decision context + traced execution for the web app.

Two small, side-effect-free helpers the request paths share so a human
annotation can be (a) curated later and (b) linked to a real trace. Both are
purely observational -- they read a decision, they never change a route, score,
slot, or decision -- and both degrade cleanly when the relevant flag is off.

* ``feedback_context(result)`` distills the *structured, non-PII* facts a curated
  eval case needs from a ``RecommendationResult``: the recommend/escalate
  outcome, the chosen route/window, the order size. The only PII (customer name /
  address) is added by the caller from the display payload and is scrub-gated
  downstream in ``feedback.capture`` -- it is deliberately NOT sourced here.

* ``traced_decision(config)`` runs a decision inside ONE OpenTelemetry span and
  hands back that span's trace coordinates, so feedback links to the exact trace
  the visualization came from -- instead of best-effort-reading a span that may
  already have closed by the time the payload is emitted. When tracing is off it
  is a transparent no-op (the span is a no-op and the coordinates are empty), so
  behavior is unchanged and feedback simply falls back to the ``decision_id``.
  Its ``DecisionSpan.record(result)`` also attaches the decision's non-PII facts
  to the span and -- when ``use_trace_dataset_payloads`` is on and PII scrub is
  off -- the decision's input/output as OpenInference ``input.value`` /
  ``output.value``, making a trace-backend-native dataset (Phoenix / Langfuse)
  replay-ready via OPEN conventions (vendor-free).
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision, RecommendationResult
from smart_assignment.shared.tracing import current_trace_context, llm_span

logger = logging.getLogger(__name__)


def decision_outcome(recommendation: Any) -> Optional[str]:
    """``"recommend"`` | ``"escalate"`` for a ``SlotRecommendation``, or ``None``
    when it can't be read. Defensive on purpose: a context snapshot is a bonus for
    curation, never something that may raise into a request."""
    if recommendation is None:
        return None
    decision = getattr(recommendation, "decision", None)
    if isinstance(decision, Decision):
        return "recommend" if decision == Decision.RECOMMENDED else "escalate"
    review = getattr(recommendation, "requires_human_review", None)
    if review is not None:
        return "escalate" if review else "recommend"
    return None


def feedback_context(result: Optional[RecommendationResult]) -> Dict[str, Any]:
    """The structured, non-PII decision facts a curated eval case needs, pulled
    from a pipeline ``result``. Nones are dropped so the snapshot stays tidy."""
    if result is None:
        return {}
    rec = getattr(result, "recommendation", None)
    ctx: Dict[str, Any] = {"outcome": decision_outcome(rec)}
    if rec is not None:
        ctx["recommended_route_id"] = getattr(rec, "recommended_route_id", None)
        ctx["recommended_window"] = getattr(rec, "recommended_window", None)
        ctx["review_reason"] = getattr(rec, "review_reason", None)
    customer = getattr(result, "customer", None)
    if customer is not None:
        ctx["order_quantity_cases"] = getattr(customer, "order_quantity_cases", None)
        # The stated preference is non-PII (a day + a time window) and is needed to
        # reconstruct a faithful intake when a curated case is replayed as an eval.
        slot = getattr(customer, "preferred_slot", None)
        if slot is not None:
            try:
                ctx["preferred_day"] = slot.day.name
                ctx["preferred_window"] = (
                    f"{slot.window[0].strftime('%H:%M')}-{slot.window[1].strftime('%H:%M')}"
                )
            except Exception:  # noqa: BLE001 - a malformed slot must not break context
                pass
    return {key: value for key, value in ctx.items() if value is not None}


# Span-attribute namespace for the decision facts attached to the decision span.
_DECISION_ATTR = "smart_assignment.decision."


def _replay_payloads(result: RecommendationResult) -> Tuple[str, str]:
    """Build the decision's ``(input, output)`` as JSON strings for a replay-ready
    trace dataset: the intake the pipeline received, and the recommendation it
    produced. Contains PII (name, address) by construction -- the caller gates on
    scrub-off before attaching it."""
    customer = result.customer
    rec = result.recommendation
    intake: Dict[str, Any] = {
        "name": customer.name,
        "address": customer.address,
        "order_quantity_cases": customer.order_quantity_cases,
    }
    slot = getattr(customer, "preferred_slot", None)
    if slot is not None:
        intake["preferred_day"] = slot.day.name
        intake["preferred_window"] = (
            f"{slot.window[0].strftime('%H:%M')}-{slot.window[1].strftime('%H:%M')}"
        )
    output = {
        "decision": rec.decision.value,
        "recommended_route_id": rec.recommended_route_id,
        "recommended_route_name": rec.recommended_route_name,
        "recommended_day": rec.recommended_day,
        "recommended_window": rec.recommended_window,
        "review_reason": rec.review_reason,
        "reasoning": rec.reasoning,
    }
    output = {key: value for key, value in output.items() if value is not None}
    return (
        json.dumps(intake, ensure_ascii=False),
        json.dumps(output, ensure_ascii=False),
    )


class DecisionSpan:
    """Handle yielded by :func:`traced_decision`.

    Call :meth:`record` with the pipeline result inside the ``with`` block to both
    (a) stash the structured feedback context for the payload and (b) attach the
    decision's non-PII facts (outcome, route, window, order size) to the span --
    so the span is informative in the trace backend instead of carrying only a
    role label. After the block, :attr:`coords` holds the span's trace/span ids
    (empty when tracing is off) and :attr:`context` holds the feedback context."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._span: Any = None
        self._config = config
        self.coords: Dict[str, str] = {}
        self.context: Dict[str, Any] = {}

    def record(self, result: Optional[RecommendationResult]) -> Dict[str, Any]:
        """Compute the feedback context from ``result`` and attach its facts to
        the span. Best-effort: attribute-setting never raises into the caller.

        When ``use_trace_dataset_payloads`` is on AND PII scrub is off, it also
        attaches the decision's input/output as OpenInference ``input.value`` /
        ``output.value`` attributes, so a Phoenix/Langfuse dataset built from these
        spans is replay-ready. Scrub-on always suppresses it (the payload carries
        PII), so no PII reaches a trace unless the operator opted in on both flags."""
        self.context = feedback_context(result)
        span = self._span
        if span is not None:
            for key, value in self.context.items():
                try:
                    span.set_attribute(_DECISION_ATTR + key, value)
                except Exception:  # noqa: BLE001 - a tracing hiccup must not break the decision
                    pass
            self._attach_replay_payloads(span, result)
        return self.context

    def _attach_replay_payloads(self, span: Any, result: Optional[RecommendationResult]) -> None:
        """Attach OpenInference input/output payloads to the span for dataset
        replay -- only when opted in (``use_trace_dataset_payloads``) and PII scrub
        is off. Uses OPEN semantic-convention keys, so it's vendor-free yet natively
        read by Phoenix and Langfuse. Never raises into the caller."""
        config = self._config
        if result is None or config is None:
            return
        if not getattr(config, "use_trace_dataset_payloads", False):
            return
        if getattr(config, "feedback_scrub_pii", True):
            return  # PII protection wins: no replay payloads on the trace when scrubbing
        try:
            input_json, output_json = _replay_payloads(result)
            span.set_attribute("input.value", input_json)
            span.set_attribute("input.mime_type", "application/json")
            span.set_attribute("output.value", output_json)
            span.set_attribute("output.mime_type", "application/json")
            # OpenInference span kind, so Phoenix classifies it as a dataset-able step.
            span.set_attribute("openinference.span.kind", "CHAIN")
        except Exception:  # noqa: BLE001 - a tracing hiccup must not break the decision
            logger.debug("Could not attach replay payloads to the decision span.", exc_info=True)


@contextmanager
def traced_decision(
    config: Config, label: str = "webapp.recommendation"
) -> Iterator[DecisionSpan]:
    """Run a decision inside one span; yields a :class:`DecisionSpan` handle.

    Usage::

        with traced_decision(cfg) as decision:
            result = run_the_pipeline(...)
            decision.record(result)      # attaches facts, stashes context
        payload["_decision"] = decision.context
        payload["_trace"] = dict(decision.coords) if decision.coords else None

    ``coords`` is populated *after* the ``with`` body runs but while the span is
    still current, so the ids belong to the span that actually wrapped the
    decision. The body may ``await`` -- the span stays current across it
    (OpenTelemetry context is task-local). With tracing off the span is a no-op,
    ``coords`` stays ``{}``, and feedback falls back to the ``decision_id``."""
    handle = DecisionSpan(config)
    with llm_span(config, label, role="webapp") as span:
        handle._span = span
        yield handle
        captured = current_trace_context()
        if captured:
            handle.coords = captured
