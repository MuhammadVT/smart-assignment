"""
Unit tests for the opt-in OpenTelemetry tracing seam (shared/tracing.py).

These run in the offline dev/CI environment, where the `observability` extra
(the OpenTelemetry SDK) is deliberately NOT installed. That is exactly the
condition the seam must survive: the tests assert that with the flag off, and
with the flag on but the SDK absent, tracing is a transparent no-op that never
raises and never changes ``generate_text``'s behavior. The one piece of real
logic that does not need the SDK -- deriving an OTLP endpoint + auth header from
the LANGFUSE_* vars -- is unit-tested directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from smart_assignment.shared import tracing
from smart_assignment.shared.config import Config
from smart_assignment.shared.llm import generate_text


@pytest.fixture(autouse=True)
def _clean_tracing_state(monkeypatch):
    """Reset the module's one-time init cache and clear any exporter env vars so
    each test starts from a clean, deterministic slate regardless of the host's
    real environment."""
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_SERVICE_NAME",
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


# --- Flag off: a true no-op that never touches OpenTelemetry --------------------


def test_get_tracer_is_none_when_flag_off():
    assert tracing._get_tracer(Config(use_tracing=False)) is None


def test_current_trace_context_is_none_without_active_span():
    # No OpenTelemetry SDK / no recording span in the offline env -> None, never
    # a raise. Feedback treats trace coordinates as best-effort on this basis.
    assert tracing.current_trace_context() is None


def test_flag_off_does_not_attempt_configuration():
    # If configuration were attempted with the flag off, this patched _configure
    # would blow up the call. It must never be reached.
    with patch.object(tracing, "_configure", side_effect=AssertionError("must not configure")):
        with tracing.llm_span(Config(use_tracing=False), "llm.generate_text") as span:
            span.set_attribute("smart_assignment.k", "v")  # safe no-op
        assert span is tracing._NOOP_SPAN


def test_llm_span_yields_usable_noop_when_disabled():
    with tracing.llm_span(Config(use_tracing=False), "x", backend="sage") as span:
        # Every mutating method on the stand-in span is safe to call.
        span.set_attribute("a", 1)
        span.set_status("ok")
        span.record_exception(ValueError("y"))


# --- Flag on but SDK absent: still a silent no-op, no raise ---------------------


def test_flag_on_without_sdk_degrades_to_noop():
    # The observability extra is not installed here, so _configure hits an
    # ImportError internally and returns None -- without raising.
    tracer = tracing._get_tracer(Config(use_tracing=True))
    assert tracer is None


def test_llm_span_propagates_caller_exception_unchanged():
    # A no-op span must not swallow the body's exception.
    with pytest.raises(ValueError, match="boom"):
        with tracing.llm_span(Config(use_tracing=True), "x"):
            raise ValueError("boom")


# --- generate_text is unaffected by the flag -----------------------------------


def _litellm_config(**overrides) -> Config:
    return Config(llm_backend="standard", model="openai/gpt-4o-mini", **overrides)


@patch("litellm.completion")
def test_generate_text_unchanged_with_tracing_on(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="a fluent rewrite"))]
    )
    # Tracing on, but SDK absent -> no-op span; the result must be identical to
    # the flag-off path exercised in test_llm.py.
    result = generate_text(_litellm_config(use_tracing=True), "some prompt", role="reasoning")
    assert result == "a fluent rewrite"


@patch("litellm.completion")
def test_generate_text_accepts_role_without_tracing(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    # The new optional role label is accepted with the flag off and changes nothing.
    assert generate_text(_litellm_config(use_tracing=False), "p", role="judgment") == "ok"


# --- Langfuse OTLP settings: pure logic, no SDK needed --------------------------


def test_langfuse_settings_none_when_incomplete(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
    # secret missing -> not enough to build settings.
    assert tracing._langfuse_otlp_settings() is None


def test_langfuse_settings_builds_endpoint_and_basic_auth(monkeypatch):
    import base64

    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000/")  # trailing slash trimmed
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-abc")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-xyz")

    settings = tracing._langfuse_otlp_settings()
    assert settings is not None
    endpoint, headers = settings
    assert endpoint == "http://localhost:3000/api/public/otel/v1/traces"

    expected_token = base64.b64encode(b"pk-lf-abc:sk-lf-xyz").decode()
    assert headers == {"Authorization": f"Basic {expected_token}"}


def test_service_name_defaults_and_override(monkeypatch):
    assert tracing._service_name() == "smart-assignment"
    monkeypatch.setenv("OTEL_SERVICE_NAME", "smart-assignment-dev")
    assert tracing._service_name() == "smart-assignment-dev"


# --- configure_tracing: one-time, flag-gated setup ------------------------------


def test_configure_tracing_is_noop_when_flag_off(monkeypatch):
    # Flag off -> must return None WITHOUT running the real setup.
    monkeypatch.setattr(
        tracing, "_configure", lambda: pytest.fail("must not configure when flag off")
    )
    assert tracing.configure_tracing(Config(use_tracing=False)) is None


def test_configure_tracing_runs_setup_exactly_once(monkeypatch):
    calls = {"n": 0}

    def _fake_configure():
        calls["n"] += 1
        return "TRACER"

    monkeypatch.setattr(tracing, "_configure", _fake_configure)
    config = Config(use_tracing=True)

    assert tracing.configure_tracing(config) == "TRACER"
    assert tracing.configure_tracing(config) == "TRACER"  # cached, no re-run
    assert calls["n"] == 1


# --- _instrument_adk: best-effort, never raises ---------------------------------


def test_instrument_adk_is_noop_without_the_instrumentor():
    # The openinference ADK instrumentor is not installed in the dev/CI env, so
    # the import fails internally and this must simply return, never raise.
    tracing._instrument_adk()


# --- agent wiring: root_agent build installs tracing ----------------------------


def test_build_root_agent_installs_tracing(monkeypatch):
    """``_build_root_agent`` must call ``configure_tracing`` so the ADK instrumentor
    is in place before the agent runs. Uses a credential-free standard/Gemini
    config with the sub-agents off, so a real LlmAgent builds offline."""
    import smart_assignment.agent as agent_mod

    cfg = Config(
        llm_backend="standard",
        model="gemini-2.5-flash",
        use_escalation_triage=False,
        use_address_resolution=False,
    )
    monkeypatch.setattr(agent_mod, "DEFAULT_CONFIG", cfg)

    seen = {}
    monkeypatch.setattr(
        agent_mod.tracing, "configure_tracing", lambda c: seen.setdefault("config", c)
    )

    agent_mod._build_root_agent()
    assert seen["config"] is cfg
