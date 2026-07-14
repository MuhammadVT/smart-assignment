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
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Coroutine

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

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
    """Drive one content-generation turn through an ADK BaseLlm object."""
    from google.adk.models.llm_request import LlmRequest  # ADK 2.x
    from google.genai import types

    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
    )
    chunks: list[str] = []
    async for response in llm.generate_content_async(request, stream=False):
        if response.text:
            chunks.append(response.text)
    return "".join(chunks)


def _run_coro_blocking(coro: "Coroutine[Any, Any, str]") -> str:
    """Drive an async coroutine to completion from *synchronous* code, whether or
    not an event loop is already running on the calling thread.

    ``generate_text`` is a synchronous API reached from the deterministic
    pipeline. In a plain CLI run no loop exists, so ``asyncio.run`` is correct.
    But the same synchronous call is also reached from *inside* a running event
    loop -- the web app's ``/api/chat`` handler is ``async`` and the ADK agent
    invokes the pipeline tools on the server's loop thread -- where ``asyncio.run``
    raises ``RuntimeError: asyncio.run() cannot be called from a running event
    loop`` (leaving the coroutine un-awaited). When a loop is already running we
    therefore run the coroutine to completion on a dedicated worker thread with
    its own loop and block for the result, so the synchronous contract holds in
    both worlds and the grounded path is no longer silently dropped under the web
    app.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop on this thread (the CLI/offline case) -- the simple path.
        return asyncio.run(coro)
    # A loop is already running here; hand the coroutine to a fresh loop on a
    # worker thread and wait for it (this thread is blocked in sync code anyway).
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
