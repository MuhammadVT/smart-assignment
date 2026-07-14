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


def test_generate_text_sage_end_to_end_from_offloaded_tool(monkeypatch):
    """End-to-end over the REAL generate_text sage branch: patch only the SDK
    boundary (env check, registry, the async driver) so the sage driver behaves
    like the loop-bound aiohttp session, then reach generate_text exactly as the
    web app does -- from a tool body offloaded off the server loop. The grounded
    text comes back instead of an exception forcing the deterministic fallback."""
    from smart_assignment.shared import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_check_sage_env_vars", lambda: None)
    monkeypatch.setattr(
        llm_mod, "_load_sage_registry", lambda: MagicMock(get_llm=lambda model: object())
    )

    host = {}

    async def fake_generate_via_sage_async(llm, prompt):
        # Like the Sage SDK's cached aiohttp session: only usable on the loop it
        # was bound to (the server loop).
        if asyncio.get_running_loop() is not host["loop"]:
            raise RuntimeError(f"loop {host['loop']!r} is not the running loop")
        return "grounded reply"

    monkeypatch.setattr(llm_mod, "_generate_via_sage_async", fake_generate_via_sage_async)

    config = Config(llm_backend="sage", sage_model="sage-gemini-2.5-flash")

    async def server_turn() -> str:
        host["loop"] = asyncio.get_running_loop()

        def sync_tool_body() -> str:
            return generate_text(config, "some prompt")

        return await offload_to_worker_thread(sync_tool_body)

    assert asyncio.run(server_turn()) == "grounded reply"
