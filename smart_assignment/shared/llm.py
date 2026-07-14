"""
LLM factory: routes every model-creation and content-generation call through
one of two backends, selected by SMART_ASSIGNMENT_LLM_BACKEND:

  "sage"     → the Sysco Sage SDK (enterprise-governed, TLS-injected).
  "standard" → SMART_ASSIGNMENT_MODEL is either a bare Gemini model name
               (e.g. "gemini-2.5-flash", used as-is) or a litellm-style
               "<provider>/<model>" string (e.g. "openai/gpt-4o-mini",
               "anthropic/claude-3-7-sonnet-latest"), wrapped in ADK's
               built-in LiteLlm so litellm handles that provider -- see
               https://docs.litellm.ai/docs/providers for the full list.
               Each provider's own env vars apply (e.g. OPENAI_API_KEY);
               requires the `litellm` extra -- see pyproject.toml.

A single env-var flip switches the entire project between them.

Exports
-------
get_llm(config)
    Returns the value for an ADK LlmAgent ``model=`` parameter.

generate_text(config, prompt)
    One-shot content generation — used by LLMReasoner so that path also flows
    through the same backend.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional, TypeVar

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

    from smart_assignment.shared.config import Config

_T = TypeVar("_T")

# The web app serves each turn on an event loop (uvicorn's), and drives the ADK
# agent + the synchronous pipeline on it. The sage backend is async and its
# aiohttp ``ClientSession`` (inside the Sage SDK's process-global litellm handler)
# is bound to the FIRST event loop that touches it -- the server loop. So a
# synchronous grounded call (``generate_text`` -> sage) MUST run its coroutine on
# that same loop, or aiohttp raises "loop <...> is not the running loop".
#
# A tool cannot both block the server loop (running synchronous pipeline code) and
# run a coroutine on it. The fix: tools offload their blocking body to a worker
# thread (freeing the loop), and record the server loop here so the nested sage
# call can submit its coroutine back to it via ``run_coroutine_threadsafe``. A
# ContextVar is the channel because ``asyncio.to_thread`` copies the context into
# the worker thread. ``None`` (the default) means "no host loop" -- the CLI/offline
# case, where ``asyncio.run`` is correct.
_HOST_EVENT_LOOP: "contextvars.ContextVar[Optional[AbstractEventLoop]]" = (
    contextvars.ContextVar("smart_assignment_host_event_loop", default=None)
)


async def offload_to_worker_thread(
    func: Callable[..., _T], /, *args: Any, **kwargs: Any
) -> _T:
    """Run a blocking, synchronous callable off the current event loop.

    Use this to wrap synchronous pipeline work (an ADK tool body, a
    re-run-for-visualization) that is invoked from async web-app code. It records
    the running loop as the *host loop* so a nested synchronous sage call
    (``generate_text``) can hand its coroutine back to that loop -- keeping the
    sage aiohttp session on the one loop it is bound to -- then runs the callable
    in a worker thread so the host loop stays free to service that coroutine.
    ``asyncio.to_thread`` copies the context, so the recorded loop is visible in
    the worker thread.
    """
    _HOST_EVENT_LOOP.set(asyncio.get_running_loop())
    return await asyncio.to_thread(func, *args, **kwargs)

# ---------------------------------------------------------------------------
# Internal: Sage SDK bootstrap (lazy, cached, runs once per process)
# ---------------------------------------------------------------------------

_SAGE_REGISTRY: Any = None


def _load_sage_registry() -> Any:
    """
    Import SageLlmRegistry and inject enterprise TLS.

    Falls back to the local source-tree layout used in Sysco workshops when
    the SDK is not installed as a wheel — mirrors the pattern from tmp.py.
    Cached after the first successful load so TLS is only injected once.
    """
    global _SAGE_REGISTRY
    if _SAGE_REGISTRY is not None:
        return _SAGE_REGISTRY

    try:
        from sage_adk import SageLlmRegistry  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        # Local workshop layouts supported:
        # 1) <repo>/smart_assignment/sage-ai-sdk-python-sage-adk_1.0.0
        # 2) <repo>/sage-ai-sdk-python-sage-adk_1.0.0
        # From this file (smart_assignment/shared/llm.py), parents[2] is
        # the repository root.
        repo_root = Path(__file__).resolve().parents[2]
        sdk_roots = [
            repo_root / "smart_assignment" / "sage-ai-sdk-python-sage-adk_1.0.0",
            repo_root / "sage-ai-sdk-python-sage-adk_1.0.0",
        ]

        sdk_root = next((root for root in sdk_roots if root.exists()), None)
        if sdk_root is None:
            raise

        local_src_paths = [
            sdk_root / "sage_adk" / "src",
            sdk_root / "sage_core" / "src",
            sdk_root / "sage_client" / "src",
        ]
        for src_path in local_src_paths:
            if src_path.exists():
                src_path_str = str(src_path)
                if src_path_str not in sys.path:
                    sys.path.insert(0, src_path_str)

        if (sdk_root / "sage_adk" / "src").exists():
            from sage_adk import SageLlmRegistry  # type: ignore[import-untyped]
        else:
            raise

    import truststore  # type: ignore[import-untyped]

    truststore.inject_into_ssl()
    _SAGE_REGISTRY = SageLlmRegistry
    return _SAGE_REGISTRY


def _check_sage_env_vars() -> None:
    """Raise RuntimeError early if any required Sage credential is absent."""
    missing = [
        v
        for v in ("SAGE_CLIENT_ID", "SAGE_CLIENT_SECRET", "SAGE_ENVIRONMENT")
        if not os.environ.get(v)
    ]
    if missing:
        raise RuntimeError(
            "SMART_ASSIGNMENT_LLM_BACKEND=sage requires the following "
            f"environment variables to be set: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Internal: async content generation through ADK BaseLlm
# ---------------------------------------------------------------------------


async def _generate_via_sage_async(llm: Any, prompt: str) -> str:
    """Drive one content-generation turn through an ADK BaseLlm object.

    ADK's ``LlmResponse`` has no flat ``.text`` attribute (that was an incorrect
    assumption that raised ``AttributeError: 'LlmResponse' object has no attribute
    'text'`` on the first real sage reply); the generated text lives in
    ``response.content.parts[i].text``. Concatenate those, skip empty/tool-only
    parts, and raise on an error response so the caller falls back deterministically
    with a clear reason rather than silently returning "".
    """
    from google.adk.models.llm_request import LlmRequest  # ADK 2.x
    from google.genai import types

    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )
    chunks: list[str] = []
    async for response in llm.generate_content_async(request, stream=False):
        error_code = getattr(response, "error_code", None)
        if error_code:
            message = getattr(response, "error_message", None) or ""
            raise RuntimeError(f"Sage backend returned an error: {error_code} {message}".strip())
        content = getattr(response, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _run_coro_blocking(coro: "Coroutine[Any, Any, str]") -> str:
    """Drive an async coroutine to completion from *synchronous* code, choosing
    the right loop for wherever the caller happens to be running.

    ``generate_text`` is a synchronous API. It is reached from three contexts:

    1. **The CLI / offline pipeline** -- no event loop on this thread. Just
       ``asyncio.run``.
    2. **A web-app tool offloaded to a worker thread** -- a host loop is recorded
       (see ``offload_to_worker_thread``) and running on another thread. The sage
       aiohttp session is bound to that host loop, so we submit the coroutine to
       it via ``run_coroutine_threadsafe`` and block for the result. Running it on
       any other loop raises "loop <...> is not the running loop"; a fresh
       ``asyncio.run`` loop would be closed after the call and break the next one.
    3. **Directly on a running loop's thread with no host loop recorded** -- a
       last-resort worker loop. Correct for loop-agnostic backends (litellm /
       genai) and strictly better than raising; the sage backend is kept out of
       this case by offloading its call sites.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    host = _HOST_EVENT_LOOP.get()
    if host is not None and host.is_running() and host is not running:
        # Case 2: run on the host loop (where the sage session lives), from here.
        return asyncio.run_coroutine_threadsafe(coro, host).result()

    if running is None:
        # Case 1: no loop on this thread.
        return asyncio.run(coro)

    # Case 3: a loop runs on THIS thread and there's no usable host loop; run the
    # coroutine on a throwaway loop in a worker thread so we don't nest asyncio.run.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _is_litellm_model(model: str) -> bool:
    """True for a litellm-style "<provider>/<model>" string, e.g.
    "openai/gpt-4o-mini" -- as opposed to a bare Gemini model name."""
    return "/" in model


def get_llm(config: "Config") -> Any:
    """
    Return the value for an ADK LlmAgent ``model=`` parameter.

    sage     → SageLlmRegistry LLM object (enterprise-governed, TLS-injected)
    standard → config.model as-is if it's a bare Gemini name, or wrapped in
               ADK's LiteLlm if it's a "<provider>/<model>" string (any
               provider litellm supports, e.g. "openai/gpt-4o-mini")
    """
    if config.llm_backend == "sage":
        _check_sage_env_vars()
        return _load_sage_registry().get_llm(config.sage_model)
    if _is_litellm_model(config.model):
        from google.adk.models.lite_llm import LiteLlm  # requires the `litellm` extra

        return LiteLlm(model=config.model)
    return config.model


def generate_text(config: "Config", prompt: str) -> str:
    """
    One-shot content generation that honours the backend toggle.

    sage     → SageLlmRegistry LLM object via ADK BaseLlm (enterprise-governed)
    standard → litellm.completion(...) if config.model is a
               "<provider>/<model>" string, else google.genai.Client
               directly for a bare Gemini model name

    Raises on failure; callers should guard with ``except Exception``.

    Note: the sage path is async under the hood. ``_run_coro_blocking`` drives it
    to completion whether or not a loop is already running, so this stays a safe
    synchronous call both from the CLI pipeline and from the web app's async
    request handlers (where a bare ``asyncio.run`` would raise).
    """
    if config.llm_backend == "sage":
        _check_sage_env_vars()
        llm = _load_sage_registry().get_llm(config.sage_model)
        return _run_coro_blocking(_generate_via_sage_async(llm, prompt))

    if _is_litellm_model(config.model):
        import litellm  # requires the `litellm` extra

        resp = litellm.completion(
            model=config.model, messages=[{"role": "user", "content": prompt}]
        )
        return (resp.choices[0].message.content or "").strip()

    # bare Gemini model name — matches the original LLMReasoner implementation
    from google import genai  # type: ignore[import-untyped]

    client = genai.Client()
    resp = client.models.generate_content(model=config.model, contents=prompt)
    return (resp.text or "").strip()
