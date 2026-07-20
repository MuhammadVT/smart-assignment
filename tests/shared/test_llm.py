"""
Unit tests for the LLM backend factory (shared/llm.py) -- focused on the
litellm-routing branch of the "standard" backend (SMART_ASSIGNMENT_MODEL as
a "<provider>/<model>" string). The plain-Gemini branch is exercised
implicitly by every other test in the suite (the sandbox/CI default is
SMART_ASSIGNMENT_LLM_BACKEND=standard -- see conftest.py); "sage" requires
the internal Sage SDK and enterprise credentials, out of scope for this
offline suite.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from smart_assignment.shared.config import Config
from smart_assignment.shared.llm import (
    _SAGE_ERROR_SENTINEL,
    _install_sage_response_diagnostic,
    _maybe_install_sage_response_diagnostic,
    _run_coro_blocking,
    generate_text,
    get_llm,
    offload_to_worker_thread,
)


def _litellm_config(**overrides) -> Config:
    return Config(llm_backend="standard", model="openai/gpt-4o-mini", **overrides)


@patch("google.adk.models.lite_llm.LiteLlm")
def test_get_llm_wraps_a_provider_prefixed_model_with_litellm(mock_lite_llm):
    get_llm(_litellm_config())
    mock_lite_llm.assert_called_once_with(model="openai/gpt-4o-mini")


def test_get_llm_returns_bare_gemini_model_string_as_is():
    config = Config(llm_backend="standard", model="gemini-2.5-flash")
    assert get_llm(config) == "gemini-2.5-flash"


# --- Sage LLM Gateway sub-path (Config.use_sage_gateway) ---------------------
#
# GatewayLlm routes the sage call through Sysco's enterprise LLM Gateway
# instead of a registered SAGE agent. Off by default (get_sage_llm, the
# direct-agent path, is exercised above/by the sage end-to-end test below);
# these tests cover the flag turning the sage branch onto the gateway sibling.


def test_get_llm_uses_direct_agent_path_by_default(monkeypatch):
    import smart_assignment.shared.llm as llm_mod

    sentinel = object()
    monkeypatch.setattr(llm_mod, "get_sage_llm", lambda model: sentinel)
    monkeypatch.setattr(
        llm_mod, "get_sage_gateway_llm", lambda model: (_ for _ in ()).throw(AssertionError)
    )
    config = Config(llm_backend="sage", sage_model="sage-gemini-2.5-flash")
    assert get_llm(config) is sentinel


def test_get_llm_routes_through_the_gateway_when_flag_is_on(monkeypatch):
    import smart_assignment.shared.llm as llm_mod

    sentinel = object()
    monkeypatch.setattr(llm_mod, "get_sage_gateway_llm", lambda model: sentinel)
    monkeypatch.setattr(
        llm_mod, "get_sage_llm", lambda model: (_ for _ in ()).throw(AssertionError)
    )
    config = Config(llm_backend="sage", sage_model="gpt-4o", use_sage_gateway=True)
    assert get_llm(config) is sentinel


def test_get_sage_gateway_llm_passes_the_model_through(monkeypatch):
    import smart_assignment.shared.llm as llm_mod
    from smart_assignment.shared.llm import get_sage_gateway_llm

    monkeypatch.setattr(llm_mod, "_check_sage_gateway_env_vars", lambda: None)
    captured = {}

    class FakeGatewayLlm:
        def __init__(self, model):
            captured["model"] = model

    monkeypatch.setattr(llm_mod, "_load_sage_gateway_llm_cls", lambda: FakeGatewayLlm)
    gateway_llm = get_sage_gateway_llm("gpt-4o")
    assert isinstance(gateway_llm, FakeGatewayLlm)
    assert captured["model"] == "gpt-4o"


def test_check_sage_gateway_env_vars_raises_when_credentials_missing(monkeypatch):
    from smart_assignment.shared.llm import _check_sage_gateway_env_vars

    monkeypatch.delenv("LLM_GATEWAY_CLIENT_ID", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_CLIENT_SECRET", raising=False)
    try:
        _check_sage_gateway_env_vars()
    except RuntimeError as exc:
        assert "LLM_GATEWAY_CLIENT_ID" in str(exc)
        assert "LLM_GATEWAY_CLIENT_SECRET" in str(exc)
    else:  # pragma: no cover - explicit failure if no error raised
        raise AssertionError("expected a RuntimeError when gateway credentials are missing")


def test_check_sage_gateway_env_vars_does_not_require_gateway_env(monkeypatch):
    # LLM_GATEWAY_ENV is optional -- the SDK's GatewayClient defaults it to "qa".
    from smart_assignment.shared.llm import _check_sage_gateway_env_vars

    monkeypatch.setenv("LLM_GATEWAY_CLIENT_ID", "id")
    monkeypatch.setenv("LLM_GATEWAY_CLIENT_SECRET", "secret")
    monkeypatch.delenv("LLM_GATEWAY_ENV", raising=False)
    _check_sage_gateway_env_vars()  # must not raise


def test_use_sage_gateway_defaults_to_false():
    assert Config().use_sage_gateway is False


def test_use_sage_gateway_read_from_env(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_USE_SAGE_GATEWAY", "true")
    assert Config.from_env().use_sage_gateway is True


def test_use_sage_gateway_unset_env_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SMART_ASSIGNMENT_USE_SAGE_GATEWAY", raising=False)
    assert Config.from_env().use_sage_gateway is False


def test_generate_text_sage_gateway_end_to_end_from_offloaded_tool(monkeypatch):
    """Same shape as the direct-agent end-to-end test above, but through the
    gateway sub-path: only the gateway SDK boundary (env check + loader) is
    patched, so the real generate_text -> _generate_via_sage_async -> response
    parsing chain is exercised exactly as it would be with a real GatewayLlm."""
    from smart_assignment.shared import llm as llm_mod

    host = {}

    class LoopBoundFakeLlm:
        async def generate_content_async(self, request, stream=False):
            if asyncio.get_running_loop() is not host["loop"]:
                raise RuntimeError(f"loop {host['loop']!r} is not the running loop")
            yield _llm_response_with_text('{"decision": "RECOMMEND"}')

    monkeypatch.setattr(llm_mod, "_check_sage_gateway_env_vars", lambda: None)
    monkeypatch.setattr(
        llm_mod, "_load_sage_gateway_llm_cls", lambda: lambda model: LoopBoundFakeLlm()
    )

    config = Config(llm_backend="sage", sage_model="gpt-4o", use_sage_gateway=True)

    async def server_turn() -> str:
        host["loop"] = asyncio.get_running_loop()

        def sync_tool_body() -> str:
            return generate_text(config, "some prompt")

        return await offload_to_worker_thread(sync_tool_body)

    assert asyncio.run(server_turn()) == '{"decision": "RECOMMEND"}'


@patch("litellm.completion")
def test_generate_text_routes_provider_prefixed_model_through_litellm(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="a fluent rewrite"))]
    )
    result = generate_text(_litellm_config(), "some deterministic trace")
    assert result == "a fluent rewrite"
    mock_completion.assert_called_once_with(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "some deterministic trace"}],
    )


@patch("litellm.completion")
def test_generate_text_strips_whitespace(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="  padded  \n"))]
    )
    assert generate_text(_litellm_config(), "prompt") == "padded"


# --- _run_coro_blocking: the sage path must survive a running event loop -------
#
# Regression for the web-app bug: generate_text() is synchronous but is reached
# from inside the async /api/chat handler (an ADK tool runs on the server's loop
# thread), where a bare asyncio.run() raises "cannot be called from a running
# event loop" and the coroutine is never awaited. _run_coro_blocking must return
# the result in both the loop-present and loop-absent cases.


async def _answer() -> str:
    return "grounded reply"


def test_run_coro_blocking_without_a_running_loop():
    # The CLI/offline case: no loop on this thread -> the asyncio.run path.
    assert _run_coro_blocking(_answer()) == "grounded reply"


def test_run_coro_blocking_inside_a_running_loop():
    # A loop is running on this thread and NO host loop is recorded (the last-resort
    # case). The pre-fix code raised RuntimeError; the fix runs the coroutine on a
    # worker loop and returns its result.
    async def driver() -> str:
        return _run_coro_blocking(_answer())

    assert asyncio.run(driver()) == "grounded reply"


def test_offloaded_call_runs_coroutine_on_the_host_loop():
    """The real web-app path. A tool body offloaded to a worker thread reaches a
    synchronous sage call (``_run_coro_blocking``); its coroutine MUST execute on
    the server's event loop -- the one the sage aiohttp session is bound to -- not
    on a throwaway worker loop. Pre-fix this ran on a worker loop and aiohttp
    raised "loop <...> is not the running loop"."""
    captured = {}

    async def capture_running_loop() -> str:
        captured["loop"] = asyncio.get_running_loop()
        return "grounded reply"

    async def server_turn():
        host_loop = asyncio.get_running_loop()

        def sync_tool_body() -> str:
            # Mirrors generate_text's sage branch running inside an offloaded tool.
            return _run_coro_blocking(capture_running_loop())

        result = await offload_to_worker_thread(sync_tool_body)
        return host_loop, result

    host_loop, result = asyncio.run(server_turn())
    assert result == "grounded reply"
    assert captured["loop"] is host_loop


def test_offloaded_call_survives_a_loop_bound_resource():
    """Reproduces the aiohttp failure directly: a resource bound to the loop that
    first touched it (the server loop), which raises if used from any other loop --
    exactly how the Sage SDK's cached aiohttp ClientSession behaves. The grounded
    call, offloaded to a worker thread, must still reach that resource successfully
    because its coroutine is run back on the host loop."""

    class LoopBoundResource:
        def __init__(self, bound_loop):
            self._bound_loop = bound_loop

        async def use(self) -> str:
            running = asyncio.get_running_loop()
            if running is not self._bound_loop:
                # The exact aiohttp failure my first fix hit.
                raise RuntimeError(f"loop {self._bound_loop!r} is not the running loop")
            return "grounded reply"

    async def server_turn():
        host_loop = asyncio.get_running_loop()
        # The session binds to the server loop the first time it's touched there,
        # just like the agent's own turn binds the shared sage session to L0.
        resource = LoopBoundResource(bound_loop=host_loop)

        def sync_tool_body() -> str:
            return _run_coro_blocking(resource.use())

        return await offload_to_worker_thread(sync_tool_body)

    assert asyncio.run(server_turn()) == "grounded reply"


def _llm_response_with_text(*texts):
    """A real ADK LlmResponse carrying text under content.parts (the actual shape
    the sage LiteLlm handler produces) -- NOT a flat `.text`."""
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=t) for t in texts])
    )


def test_generate_via_sage_async_reads_text_from_content_parts():
    """Regression for `AttributeError: 'LlmResponse' object has no attribute
    'text'`: ADK's LlmResponse exposes text via content.parts[].text. The driver
    must concatenate those, not read a (non-existent) flat `.text`."""
    from smart_assignment.shared.llm import _generate_via_sage_async

    class FakeLlm:
        async def generate_content_async(self, request, stream=False):
            yield _llm_response_with_text("hello ", "world")

    assert asyncio.run(_generate_via_sage_async(FakeLlm(), "prompt")) == "hello world"


def test_generate_via_sage_async_over_real_adk_litellm():
    """The most faithful check: drive the REAL ADK ``LiteLlm`` (what
    ``SageLlmRegistry.get_llm`` returns) through a fake litellm custom provider --
    the exact shape the Sage SDK uses (``SageLiteLlm`` is a litellm ``CustomLLM``).
    This exercises ADK's real response object, so it catches the wrong-attribute
    bug and any future ADK response-shape drift, not just a hand-rolled fake."""
    import litellm
    from google.adk.models.lite_llm import LiteLlm
    from litellm import Choices, CustomLLM, Message, ModelResponse

    from smart_assignment.shared.llm import _generate_via_sage_async

    class _FakeProvider(CustomLLM):
        async def acompletion(self, model, messages, model_response=None, **kwargs):
            mr = model_response or ModelResponse()
            mr.choices = [Choices(message=Message(role="assistant", content='{"ok": true}'))]
            return mr

    litellm.custom_provider_map.append(
        {"provider": "smart_assignment_test", "custom_handler": _FakeProvider()}
    )
    try:
        llm = LiteLlm(model="smart_assignment_test/model")
        assert asyncio.run(_generate_via_sage_async(llm, "prompt")) == '{"ok": true}'
    finally:
        litellm.custom_provider_map[:] = [
            p for p in litellm.custom_provider_map if p.get("provider") != "smart_assignment_test"
        ]


def test_generate_via_sage_async_raises_on_error_response():
    """An error response surfaces as a RuntimeError so the caller falls back with a
    reason, instead of silently returning an empty string that fails JSON parsing."""
    from google.adk.models.llm_response import LlmResponse

    from smart_assignment.shared.llm import _generate_via_sage_async

    class FakeLlm:
        async def generate_content_async(self, request, stream=False):
            yield LlmResponse(error_code="SAFETY", error_message="blocked")

    try:
        asyncio.run(_generate_via_sage_async(FakeLlm(), "prompt"))
    except RuntimeError as exc:
        assert "SAFETY" in str(exc)
    else:  # pragma: no cover - explicit failure if no error raised
        raise AssertionError("expected a RuntimeError on an error response")


def test_generate_text_sage_end_to_end_from_offloaded_tool(monkeypatch):
    """End-to-end over the REAL generate_text sage branch AND the REAL response
    parsing: patch only the SDK boundary (env check + registry) to return a fake
    ADK BaseLlm whose generate_content_async yields a real LlmResponse -- and which,
    like the Sage SDK's cached aiohttp session, only works on the loop it was bound
    to. Reach it exactly as the web app does: from a tool body offloaded off the
    server loop. The grounded text comes back instead of an exception forcing the
    deterministic fallback."""
    from smart_assignment.shared import llm as llm_mod

    host = {}

    class LoopBoundFakeLlm:
        async def generate_content_async(self, request, stream=False):
            if asyncio.get_running_loop() is not host["loop"]:
                raise RuntimeError(f"loop {host['loop']!r} is not the running loop")
            yield _llm_response_with_text('{"decision": "RECOMMEND"}')

    monkeypatch.setattr(llm_mod, "_check_sage_env_vars", lambda: None)
    monkeypatch.setattr(
        llm_mod, "_load_sage_registry", lambda: MagicMock(get_llm=lambda model: LoopBoundFakeLlm())
    )

    config = Config(llm_backend="sage", sage_model="sage-gemini-2.5-flash")

    async def server_turn() -> str:
        host["loop"] = asyncio.get_running_loop()

        def sync_tool_body() -> str:
            return generate_text(config, "some prompt")

        return await offload_to_worker_thread(sync_tool_body)

    assert asyncio.run(server_turn()) == '{"decision": "RECOMMEND"}'


# --- Sage response diagnostic (Plan A: reveal the SDK's masked reply) ---------
#
# The Sage SDK replaces the model's real answer with a fixed "Something went wrong"
# sentinel whenever the agent returns a tool call the grounded path didn't offer.
# _install_sage_response_diagnostic wraps the extractor to log the TRUE
# agent_response in that case, without changing the returned value.


class _FakeSageResponse:
    def __init__(self, agent_response):
        self.data = {"responses": {"agent_response": agent_response}}


class _FakeSageLiteLlm:
    """Mimics the SDK: returns the sentinel for a function call not in `tools`."""

    @staticmethod
    def _extract_response(response, tools):
        ar = response.data["responses"]["agent_response"]
        if isinstance(ar, dict) and ar.get("function_call"):
            name = ar["function_call"].get("name")
            if not any(t == name for t in (tools or [])):
                return {}, _SAGE_ERROR_SENTINEL
        return {}, (ar.get("text") if isinstance(ar, dict) else ar)


def _fresh_fake_sage_cls():
    # A subclass so each test wraps a pristine, un-wrapped _extract_response.
    return type("FakeSage", (_FakeSageLiteLlm,), {})


def test_diagnostic_logs_the_real_agent_response_on_the_sentinel(caplog):
    cls = _fresh_fake_sage_cls()
    _install_sage_response_diagnostic(cls)
    resp = _FakeSageResponse({"function_call": {"name": "lookup_customer", "args": {}}})

    with caplog.at_level("WARNING"):
        function_call, text = cls._extract_response(resp, [])

    # The returned value is unchanged (still the sentinel, so the caller falls back)...
    assert text == _SAGE_ERROR_SENTINEL
    # ...but the real agent_response is now visible in the logs.
    assert "lookup_customer" in caplog.text
    assert "real" in caplog.text.lower()


def test_diagnostic_is_silent_on_a_normal_text_reply(caplog):
    cls = _fresh_fake_sage_cls()
    _install_sage_response_diagnostic(cls)
    resp = _FakeSageResponse({"text": '{"chosen_index": 0}'})

    with caplog.at_level("WARNING"):
        _, text = cls._extract_response(resp, [])

    assert text == '{"chosen_index": 0}'
    assert caplog.text == ""


def test_diagnostic_install_is_idempotent():
    cls = _fresh_fake_sage_cls()
    _install_sage_response_diagnostic(cls)
    wrapped_once = cls._extract_response
    _install_sage_response_diagnostic(cls)
    assert cls._extract_response is wrapped_once  # not double-wrapped


def test_maybe_install_is_a_noop_when_flag_off(monkeypatch):
    # Flag off -> must not even try to import the SDK.
    def _boom():
        raise AssertionError("should not import the SDK when the flag is off")

    monkeypatch.setattr("builtins.__import__", _boom, raising=False)
    # Should simply return without raising.
    _maybe_install_sage_response_diagnostic(Config(debug_sage_raw_response=False))


def test_maybe_install_survives_missing_sdk(monkeypatch):
    # Flag on but the Sage SDK isn't importable (this offline suite) -> no crash.
    _maybe_install_sage_response_diagnostic(Config(debug_sage_raw_response=True))
