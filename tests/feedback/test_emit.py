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


# --- Note-on-span behavior gated by the PII toggle (bullet: note visible when
#     scrub is off) -- exercised with a fake tracer so it works without the SDK.


class _FakeSpanCM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTracer:
    def __init__(self):
        self.attributes = None

    def start_as_current_span(self, name, **kwargs):
        self.attributes = kwargs.get("attributes", {})
        return _FakeSpanCM()


def _emit_with_fake_tracer(monkeypatch, config, note):
    tracer = _FakeTracer()
    monkeypatch.setattr(
        "smart_assignment.shared.tracing.configure_tracing", lambda cfg: tracer
    )
    rec = FeedbackRecord(target=FeedbackTarget(decision_id="d1"), label="thumbs_down", note=note)
    assert emit_feedback_span(config, rec) is True
    return tracer.attributes


def test_note_text_on_span_when_scrub_off(monkeypatch):
    note = "wrong route for 5085 Westheimer"
    attrs = _emit_with_fake_tracer(
        monkeypatch, Config(use_tracing=True, feedback_scrub_pii=False), note
    )
    assert attrs["smart_assignment.feedback.has_note"] is True
    # Scrub OFF -> the real note text is on the span (visible in Phoenix/Langfuse).
    assert attrs["smart_assignment.feedback.note"] == note


def test_note_text_absent_from_span_when_scrub_on(monkeypatch):
    attrs = _emit_with_fake_tracer(
        monkeypatch, Config(use_tracing=True, feedback_scrub_pii=True), "some note"
    )
    assert attrs["smart_assignment.feedback.has_note"] is True
    # Scrub ON -> only the boolean, never the text.
    assert "smart_assignment.feedback.note" not in attrs
