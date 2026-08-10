"""Tests for shared/async_bridge.py -- the "one loop owns the backend session"
invariant.

The failure these guard against is not hypothetical: the sage backend caches one
``aiohttp.ClientSession`` per process, bound to whichever event loop created it,
and reports its unhappiness two different ways -- "loop <...> is not the running
loop" while that loop is alive, "Event loop is closed" after it ends. Both are
modelled below by ``_LoopBoundSession``, which is a faithful stand-in: it is the
same "remember my loop, refuse every other one" behaviour, minus the network.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from smart_assignment.shared.async_bridge import (
    dedicated_loop,
    offload_to_worker_thread,
    reset_loop_binding,
    run_coroutine_blocking,
)


class _LoopBoundSession:
    """Stands in for the SDK's cached aiohttp session: binds to the first loop
    that uses it, then rejects every other one, exactly as aiohttp does."""

    def __init__(self) -> None:
        self.loop = None

    async def request(self, payload: str) -> str:
        running = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running
        elif self.loop is not running:
            raise RuntimeError(f"loop {self.loop!r} is not the running loop")
        elif self.loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return f"answered: {payload}"


async def _answer() -> str:
    return "grounded reply"


async def _running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


# --- the offline/CLI path: repeated synchronous calls, no loop anywhere -------


def test_a_synchronous_call_returns_its_result():
    assert run_coroutine_blocking(_answer()) == "grounded reply"


def test_repeated_synchronous_calls_share_one_never_closing_loop():
    """THE regression. Pre-fix, call 1 ran under ``asyncio.run``, which closed the
    loop the session had just bound to, so call 2 died with "Event loop is
    closed" -- silently degrading ``scripts/run_local.py``, ``outcome_scoring
    --path llm`` and ``/api/recommend`` to the deterministic fallback."""
    session = _LoopBoundSession()
    answers = [run_coroutine_blocking(session.request(f"call {n}")) for n in range(3)]

    assert answers == ["answered: call 0", "answered: call 1", "answered: call 2"]
    assert not session.loop.is_closed()


def test_the_dedicated_loop_is_reused_across_calls():
    first = run_coroutine_blocking(_running_loop())
    second = run_coroutine_blocking(_running_loop())
    assert first is second is dedicated_loop()


def test_the_dedicated_loop_runs_off_the_calling_thread():
    # It must be a real background loop, not something driven by the caller --
    # otherwise a nested call could not block on it.
    loop_thread = run_coroutine_blocking(_thread_ident())
    assert loop_thread != threading.get_ident()


async def _thread_ident() -> int:
    return threading.get_ident()


def test_an_exception_propagates_to_the_synchronous_caller():
    async def boom() -> str:
        raise ValueError("backend said no")

    with pytest.raises(ValueError, match="backend said no"):
        run_coroutine_blocking(boom())


# --- the web-app path: a host loop already owns the session ------------------


def test_an_offloaded_call_runs_on_the_host_loop():
    """The chat path. The ADK agent's own streaming call has already bound the
    session to the server loop, outside this module entirely -- so a nested
    synchronous call must join that loop rather than pick its own."""
    captured = {}

    async def server_turn():
        host = asyncio.get_running_loop()

        def tool_body():
            return run_coroutine_blocking(_running_loop())

        captured["ran_on"] = await offload_to_worker_thread(tool_body)
        return host

    host = asyncio.run(server_turn())
    assert captured["ran_on"] is host


def test_an_offloaded_call_reaches_a_session_bound_to_the_host_loop():
    """The same thing stated as the failure it prevents: a session bound to the
    server loop (as the agent's first turn binds it) stays reachable from a tool
    body running on a worker thread."""

    async def server_turn():
        session = _LoopBoundSession()
        # Bind it here, on the server loop, before any offload -- exactly the
        # order the web app produces.
        await session.request("agent turn")

        def tool_body():
            return run_coroutine_blocking(session.request("grounded call"))

        return await offload_to_worker_thread(tool_body)

    assert asyncio.run(server_turn()) == "answered: grounded call"


def test_the_host_loop_keeps_winning_once_bound():
    """Two requests on the same server loop must not drift apart: the second one
    goes where the first one went."""

    async def server_turn():
        host = asyncio.get_running_loop()
        session = _LoopBoundSession()

        def tool_body(n):
            return run_coroutine_blocking(session.request(f"call {n}"))

        first = await offload_to_worker_thread(tool_body, 1)
        second = await offload_to_worker_thread(tool_body, 2)
        return host, first, second, session.loop

    host, first, second, bound = asyncio.run(server_turn())
    assert (first, second) == ("answered: call 1", "answered: call 2")
    assert bound is host


# --- the eval path: a host loop exists but does NOT own the session ----------


def test_a_later_host_loop_does_not_steal_a_session_already_bound_elsewhere():
    """``eval/test_rationale_faithfulness.py`` in one picture, and the reason the
    binding is remembered rather than recomputed per call.

    Its async test body first makes a *synchronous* grounded call (no host loop
    -> the dedicated loop binds the session), then a deepeval judge call that
    offloads (host loop = pytest-asyncio's). Preferring the host loop there would
    hand the session to a loop that does not own it. Pre-fix this test file could
    not complete a single case."""
    session = _LoopBoundSession()

    # Step 1: the synchronous grounded call, made from async code without an
    # offload -- binds the session to the dedicated loop.
    async def eval_case():
        first = run_coroutine_blocking(session.request("grounded pick"))

        # Step 2: the judge, which DOES offload and so records a host loop.
        def judge_body():
            return run_coroutine_blocking(session.request("judge"))

        second = await offload_to_worker_thread(judge_body)
        return first, second

    first, second = asyncio.run(eval_case())
    assert (first, second) == ("answered: grounded pick", "answered: judge")
    assert session.loop is dedicated_loop()


def test_a_direct_call_from_a_loop_thread_uses_the_dedicated_loop():
    """Synchronous call made from a loop's own thread, with nothing bound yet:
    the dedicated loop is a *different* thread, so it can serve the call while
    this one blocks. This is ``eval/test_rationale_faithfulness.py``'s shape --
    ``_grounded_index`` runs synchronously inside an async test body."""

    async def driver() -> str:
        return run_coroutine_blocking(_answer())

    assert asyncio.run(driver()) == "grounded reply"


def test_blocking_on_the_loop_that_owns_the_session_says_what_to_do():
    """The one genuinely unserviceable case: the session is bound to THIS thread's
    loop and we are asked to block on it. No loop can run the coroutine -- the
    bound one is blocked by us, any other is rejected by the session. Fail with
    the fix in the message instead of surfacing aiohttp's "loop <...> is not the
    running loop" three layers down inside a connection error."""

    async def driver():
        # Bind the session to this loop first, the way the web app's agent turn
        # or an earlier offloaded request does.
        def tool_body():
            return run_coroutine_blocking(_answer())

        await offload_to_worker_thread(tool_body)

        # Now call synchronously from the loop's own thread.
        with pytest.raises(RuntimeError, match="offload_to_worker_thread"):
            run_coroutine_blocking(_answer())

    asyncio.run(driver())


def test_a_dead_bound_loop_is_replaced_rather_than_reused(caplog):
    """A bound loop that has since stopped cannot serve anything. Re-bind and say
    so, instead of raising from deep inside ``run_coroutine_threadsafe``."""

    async def bind_then_finish():
        def body():
            return run_coroutine_blocking(_running_loop())

        return await offload_to_worker_thread(body)

    stale = asyncio.run(bind_then_finish())  # its loop is closed on return
    assert stale.is_closed()

    with caplog.at_level("WARNING"):
        assert run_coroutine_blocking(_answer()) == "grounded reply"
    assert "has stopped" in caplog.text


def test_reset_loop_binding_is_scoped_to_the_binding_not_the_loop():
    """The reset hook must forget the *choice* of loop without tearing down the
    dedicated loop itself -- otherwise the fixture that calls it between tests
    would recreate a thread per test."""
    before = dedicated_loop()
    reset_loop_binding()
    assert dedicated_loop() is before
    assert not before.is_closed()
