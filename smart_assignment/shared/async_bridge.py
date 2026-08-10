"""One event loop owns the LLM backend's HTTP session, for the whole process.

Why this module exists
----------------------
The sage backend is async and keeps ONE HTTP session for the entire process.
[VERIFIED against the installed SDK] ``SageLlmRegistry._handlers`` is a *class*
dict and ``litellm.custom_provider_map`` is a module global, so the
``SageLiteLlm`` handler -- and the ``AsyncSAGEClient`` -> ``AsyncBaseClient``
underneath it -- is created once and reused. ``AsyncBaseClient._get_session()``
caches an ``aiohttp.ClientSession`` the first time it is called and never
reopens it. An aiohttp session belongs to the event loop that created it, so:

* used from a DIFFERENT live loop -> ``RuntimeError: loop <...> is not the
  running loop``
* used after that loop closed -> ``APIConnectionError: Event loop is closed``

``generate_text``/``generate_tool_call`` are synchronous, so something has to
drive that coroutine. Driving it on a *throwaway* loop (``asyncio.run``, or a
one-shot ``ThreadPoolExecutor`` loop) works exactly once: the first call binds
the session, the loop then closes, and every later call in that process hits
"Event loop is closed". That was a live bug -- ``POST /api/recommend`` silently
fell back to the deterministic pick from its second grounded request onward, and
one such call poisoned the chat path for the rest of the process.

The rule this module enforces
-----------------------------
**Every synchronous LLM coroutine runs on the loop that owns the session, and
that loop never closes.** Concretely, ``run_coroutine_blocking`` picks a target
in this order and then remembers it for the process:

1. **The already-bound loop**, if one is recorded and still running. First loop
   wins -- which is what the SDK's cached session enforces anyway.
2. **The host loop** (see ``offload_to_worker_thread``), when nothing is bound
   yet. This is the web app: uvicorn's loop has *already* bound the session via
   the ADK agent's own streaming call, which happens outside this module
   entirely, so we must join it rather than compete with it. It also outlives
   the process's request handling, so it is safe to bind to.
3. **A dedicated, process-owned loop** on a daemon thread, when there is no host
   loop -- the CLI, batch scripts, and the eval suite. Unlike ``asyncio.run`` it
   is started once and never closed, so the session it binds stays usable for
   every later call.

The corollary, and it is load-bearing: **a process must not mix (2) and (3).**
Any entry point that reaches an LLM call from async code has to establish the
host loop via ``offload_to_worker_thread`` -- otherwise whichever request
arrives first binds the session and the other path breaks. That is why
``webapp/app.py``'s ``/api/recommend`` offloads instead of relying on FastAPI's
sync-endpoint threadpool.

Loop-agnostic backends (litellm, google.genai) do not care which loop they run
on; they simply inherit the same, cheaper path.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional, TypeVar

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The loop a caller wants nested synchronous LLM calls to run on. Set by
# ``offload_to_worker_thread`` and read from the worker thread it spawns -- a
# ContextVar because ``asyncio.to_thread`` copies the context across the thread
# boundary. ``None`` (the default) means "no host loop": the CLI/offline case.
_HOST_EVENT_LOOP: "contextvars.ContextVar[Optional[AbstractEventLoop]]" = contextvars.ContextVar(
    "smart_assignment_host_event_loop", default=None
)

# Guards both globals below. Reentrant because ``_select_target_loop`` holds it
# while it may start the dedicated loop.
_LOCK = threading.RLock()

# The dedicated loop, started on first use and then never closed.
_DEDICATED_LOOP: "Optional[AbstractEventLoop]" = None

# The loop that owns the backend's HTTP session for this process. Deliberately a
# plain global, not a ContextVar: the session it describes is process-global, so
# a per-context answer would be a lie.
_BOUND_LOOP: "Optional[AbstractEventLoop]" = None

_LOOP_THREAD_NAME = "smart-assignment-llm-loop"


async def offload_to_worker_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run a blocking, synchronous callable off the current event loop.

    Use this to wrap synchronous pipeline work (an ADK tool body, a web request
    handler, a re-run-for-visualization) that is invoked from async code. It does
    two things, and the second is the important one:

    * runs ``func`` in a worker thread, so the calling loop stays free; and
    * records the calling loop as the *host loop*, so a nested synchronous LLM
      call lands back on it rather than on a throwaway loop (see
      ``run_coroutine_blocking``).

    A tool cannot both block the server loop and run a coroutine on it, which is
    why the offload and the recording have to happen together.
    """
    _HOST_EVENT_LOOP.set(asyncio.get_running_loop())
    return await asyncio.to_thread(func, *args, **kwargs)


def _start_dedicated_loop() -> "AbstractEventLoop":
    """Start a never-closing event loop on a daemon thread and return it.

    Daemon so it never keeps the interpreter alive; there is no shutdown hook to
    forget, and nothing owns state worth draining at exit that the process
    teardown would not discard anyway.
    """
    started = threading.Event()
    holder: dict = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        loop.call_soon(started.set)
        loop.run_forever()

    threading.Thread(target=_run, name=_LOOP_THREAD_NAME, daemon=True).start()
    started.wait()
    return holder["loop"]


def dedicated_loop() -> "AbstractEventLoop":
    """The process-owned loop, started on first call. Never closed."""
    global _DEDICATED_LOOP
    with _LOCK:
        if _DEDICATED_LOOP is None or _DEDICATED_LOOP.is_closed():
            _DEDICATED_LOOP = _start_dedicated_loop()
        return _DEDICATED_LOOP


def _select_target_loop() -> "AbstractEventLoop":
    """Which loop this process's LLM coroutines belong on -- see module docstring
    for the ordering and why it is that order."""
    global _BOUND_LOOP
    with _LOCK:
        if _BOUND_LOOP is not None:
            if _BOUND_LOOP.is_running():
                return _BOUND_LOOP
            # The bound loop finished. Any session bound to it is already dead,
            # so re-binding cannot make things worse -- but it cannot revive the
            # session either, so say so rather than fail silently downstream.
            logger.warning(
                "The event loop owning the LLM backend session has stopped; re-binding. "
                "A backend with a loop-bound HTTP session may fail until this process "
                "restarts."
            )

        host = _HOST_EVENT_LOOP.get()
        _BOUND_LOOP = host if host is not None and host.is_running() else dedicated_loop()
        return _BOUND_LOOP


def run_coroutine_blocking(coro: "Coroutine[Any, Any, _T]") -> _T:
    """Drive an async coroutine to completion from *synchronous* code, on the one
    loop this process's LLM backend session is bound to.

    Blocks the calling thread until the coroutine finishes, and re-raises
    whatever it raised.
    """
    target = _select_target_loop()

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is target:
        # We are ON the loop that owns the session, and being asked to block until
        # it finishes work -- which it cannot start until we stop blocking. There
        # is no loop that can serve this call: any other one hits the session's
        # "not the running loop". So say what is wrong and how to fix it, rather
        # than let aiohttp report it three layers down as a connection error.
        # Callers of generate_text guard with `except Exception` and fall back
        # deterministically, so this degrades safely and logs a real reason.
        coro.close()
        raise RuntimeError(
            "A synchronous LLM call was made from the thread running the event loop "
            "that owns the backend's HTTP session; it cannot be completed without "
            "deadlocking. Wrap the calling code in "
            "smart_assignment.shared.llm.offload_to_worker_thread so it runs on a "
            "worker thread while that loop stays free to serve the call."
        )

    return asyncio.run_coroutine_threadsafe(coro, target).result()


def _quiet_shutdown() -> None:
    """Silence the dedicated loop's teardown chatter at interpreter exit.

    The backend's session is deliberately never closed -- that is the whole
    point -- so aiohttp's ``ClientSession.__del__`` reports it as "Unclosed
    client session" during teardown, through the loop's exception handler. Under
    pytest that lands on already-closed capture streams and becomes a wall of
    "--- Logging error ---" after the results.

    Nothing reported at interpreter exit is actionable, so replace the handler
    *then* -- not before, so genuine runtime errors still surface. It has to be
    installed synchronously: ``call_soon_threadsafe`` only queues the swap, and
    the session can be collected before the loop thread gets to it.

    Covers the bound loop as well as the dedicated one: under ``eval/test_eval.py``
    the session is bound to ADK's runner loop -- bound outside this module by the
    agent's own streaming call -- and that loop is long closed by exit.

    The loops themselves are left alone; the dedicated one's thread is a daemon
    and dies with the process, and stopping it here would strand any later atexit
    handler that still wanted a call. Best-effort throughout: nothing here may
    affect the exit status.
    """
    for loop in {id(each): each for each in (_DEDICATED_LOOP, _BOUND_LOOP) if each}.values():
        _silence_loop(loop)


def _ignore(*_args: Any) -> None:
    """Exception handler that drops everything. Installed only at exit."""


def _silence_loop(loop: "AbstractEventLoop") -> None:
    """Replace ``loop``'s exception handler, from whatever thread we are on."""
    try:
        if not loop.is_running():
            # Stopped or closed: nothing is racing us, so assign directly --
            # call_soon_threadsafe would raise or never run.
            loop.set_exception_handler(_ignore)
            return
        installed = threading.Event()

        def _install() -> None:
            loop.set_exception_handler(_ignore)
            installed.set()

        loop.call_soon_threadsafe(_install)
        installed.wait(timeout=5)
    except (RuntimeError, TypeError):  # already tearing down -- nothing to quiet
        pass


atexit.register(_quiet_shutdown)


def reset_loop_binding() -> None:
    """Forget which loop owns the backend session.

    For tests only. A real process binds once and keeps it -- that is the whole
    point -- but each test wants to look like a fresh process, and a loop bound
    by an earlier test is long dead by the time the next one runs.
    """
    global _BOUND_LOOP
    with _LOCK:
        _BOUND_LOOP = None
