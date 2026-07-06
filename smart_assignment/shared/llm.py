"""
LLM factory: routes every model-creation and content-generation call through
one of three backends, selected by SMART_ASSIGNMENT_LLM_BACKEND:

  "sage"     → the Sysco Sage SDK (enterprise-governed, TLS-injected).
  "standard" → plain Google ADK / genai (Gemini via API key or Vertex).
  "openai"   → an OpenAI model via ADK's built-in LiteLLM wrapper (requires
               the `openai` extra -- see pyproject.toml -- and OPENAI_API_KEY).

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
# Internal: OpenAI via ADK's built-in LiteLLM wrapper
# ---------------------------------------------------------------------------


def _openai_litellm_model(model: str) -> str:
    """litellm's provider-prefixed model name for a bare OpenAI model name."""
    return f"openai/{model}"


def _load_lite_llm(model: str) -> Any:
    """
    Wrap an OpenAI model for use as an ADK LlmAgent's ``model=``, via ADK's
    own LiteLlm class. Requires OPENAI_API_KEY -- litellm reads it directly,
    so there's nothing to check here. Raises ImportError with an actionable
    message (pip install the `openai` extra) if litellm isn't installed.
    """
    from google.adk.models.lite_llm import LiteLlm  # requires the `openai` extra

    return LiteLlm(model=_openai_litellm_model(model))


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_llm(config: "Config") -> Any:
    """
    Return the value for an ADK LlmAgent ``model=`` parameter.

    sage     → SageLlmRegistry LLM object (enterprise-governed, TLS-injected)
    openai   → LiteLlm-wrapped OpenAI model (requires OPENAI_API_KEY)
    standard → plain model-name string (standard ADK / local dev)
    """
    if config.llm_backend == "sage":
        _check_sage_env_vars()
        return _load_sage_registry().get_llm(config.sage_model)
    if config.llm_backend == "openai":
        return _load_lite_llm(config.openai_model)
    return config.model


def generate_text(config: "Config", prompt: str) -> str:
    """
    One-shot content generation that honours the backend toggle.

    sage     → SageLlmRegistry LLM object via ADK BaseLlm (enterprise-governed)
    openai   → litellm.completion(...) directly (requires OPENAI_API_KEY)
    standard → google.genai.Client directly (standard / local dev)

    Raises on failure; callers should guard with ``except Exception``.

    Note: the sage path uses ``asyncio.run()``, which requires no running event
    loop.  LLMReasoner is called from the synchronous pipeline so this is safe;
    if the pipeline is ever made async, replace with an ``await`` call instead.
    """
    if config.llm_backend == "sage":
        _check_sage_env_vars()
        llm = _load_sage_registry().get_llm(config.sage_model)
        return asyncio.run(_generate_via_sage_async(llm, prompt))

    if config.llm_backend == "openai":
        import litellm  # requires the `openai` extra

        resp = litellm.completion(
            model=_openai_litellm_model(config.openai_model),
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    # standard path — matches the original LLMReasoner implementation
    from google import genai  # type: ignore[import-untyped]

    client = genai.Client()
    resp = client.models.generate_content(model=config.model, contents=prompt)
    return (resp.text or "").strip()
