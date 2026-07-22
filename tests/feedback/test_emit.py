"""Unit tests for the vendor-neutral OTLP emit.

The offline CI env has no OpenTelemetry SDK, so the emit must be a silent no-op
that never raises -- exactly the property these assert. The span-link builder's
pure logic (coordinate validation) is exercised directly.
"""

from __future__ import annotations

from smart_assignment.feedback.emit import _build_link, emit_feedback_span
from smart_assignment.feedback.schema import FeedbackRecord, FeedbackTarget
from smart_assignment.shared.config import Config


def _rec(**target_over):
    return FeedbackRecord(
        target=FeedbackTarget(decision_id="d1", **target_over),
        label="thumbs_down",
        score=0.0,
    )


def test_emit_is_noop_when_tracing_off():
    assert emit_feedback_span(Config(use_tracing=False), _rec()) is False


def test_emit_noop_when_sdk_absent_even_if_tracing_on():
    # Tracing flag on but the SDK isn't installed in CI -> configure returns None
    # -> a silent no-op, never a raise.
    assert emit_feedback_span(Config(use_tracing=True), _rec()) is False


def test_build_link_none_without_coordinates():
    assert _build_link(None, None) is None
    assert _build_link("abc", None) is None


def test_build_link_none_when_api_absent_or_invalid():
    # With no OpenTelemetry API installed this returns None (import guarded);
    # with it installed, an all-zero context is invalid and also returns None.
    assert _build_link("0" * 32, "0" * 16) is None
