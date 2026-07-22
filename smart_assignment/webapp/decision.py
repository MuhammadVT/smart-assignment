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
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision, RecommendationResult
from smart_assignment.shared.tracing import current_trace_context, llm_span


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
    return {key: value for key, value in ctx.items() if value is not None}


@contextmanager
def traced_decision(
    config: Config, label: str = "webapp.recommendation"
) -> Iterator[Dict[str, str]]:
    """Run a decision inside one span and expose its trace coordinates.

    Usage::

        with traced_decision(cfg) as trace_ctx:
            result = run_the_pipeline(...)
        # trace_ctx is now {"trace_id", "span_id"} (or {} when tracing is off)

    The yielded dict is populated *after* the ``with`` body runs but while the
    span is still current, so the coordinates belong to the span that actually
    wrapped the decision. The body may ``await`` -- the span stays current across
    it (OpenTelemetry context is task-local). A no-op span yields ``{}``."""
    coords: Dict[str, str] = {}
    with llm_span(config, label, role="webapp"):
        yield coords
        captured = current_trace_context()
        if captured:
            coords.update(captured)
