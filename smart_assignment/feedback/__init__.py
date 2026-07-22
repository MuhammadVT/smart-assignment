"""
Human (and automated) feedback capture for production decisions.

A small, vendor-neutral layer that records quality judgments about decisions the
pipeline already made -- a thumbs-up/down, an optional score, a note -- and holds
the repo's standard guarantees (see CLAUDE.md):

* **Opt-in, default off.** Everything is gated by ``Config.use_human_feedback``.
  Flag-off is a no-op that persists nothing and imports nothing further.
* **Never worse than the baseline.** Feedback is purely observational; it changes
  no route, score, slot, or decision. Any *use* of the labels (eval calibration,
  prompt tuning) is a separate, offline, human-driven step (see ``curate``).
* **Auditable & durable.** Every annotation is written to an append-only JSONL
  log (``store``) -- the source of truth -- before any best-effort observability
  emit (``emit``), so the record survives a down trace backend.
* **Vendor-free.** Observability is emitted as a standard OTLP span linked to the
  decision's trace (``emit``); any OTLP backend (Phoenix, Langfuse, ...) ingests
  it with only an endpoint change. No vendor annotation SDK is imported.

Public API is intentionally tiny: build a :class:`FeedbackRecord` and call
:func:`record_feedback`. ``curate`` is imported on demand (it pulls the eval
fixtures) via module ``__getattr__`` so importing this package stays light.
"""

from __future__ import annotations

from typing import Any

from smart_assignment.feedback.capture import record_feedback
from smart_assignment.feedback.schema import (
    ANNOTATOR_CODE,
    ANNOTATOR_HUMAN,
    ANNOTATOR_LLM,
    DECISION_FINAL_RESPONSE,
    FeedbackRecord,
    FeedbackTarget,
    FeedbackValidationError,
    validate_feedback,
)
from smart_assignment.feedback.store import iter_records, read_records

__all__ = [
    "record_feedback",
    "FeedbackRecord",
    "FeedbackTarget",
    "FeedbackValidationError",
    "validate_feedback",
    "iter_records",
    "read_records",
    "ANNOTATOR_HUMAN",
    "ANNOTATOR_LLM",
    "ANNOTATOR_CODE",
    "DECISION_FINAL_RESPONSE",
    "curate_feedback",
    "CuratedCase",
]


def __getattr__(name: str) -> Any:
    """Lazily surface the curation API without importing it (and the eval
    fixtures it pulls) on every ``import smart_assignment.feedback``."""
    if name in ("curate_feedback", "CuratedCase"):
        from smart_assignment.feedback import curate

        return getattr(curate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
