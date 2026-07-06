"""
LLM factory: routes every model-creation and content-generation call through
the Sysco Sage SDK (enterprise-governed, TLS-injected) when
SMART_ASSIGNMENT_LLM_BACKEND=sage, or through the standard Google ADK / genai
path when =standard.  A single env-var flip switches the entire project.

Exports
-------
get_llm(config)
    Returns the value for an ADK LlmAgent ``model=`` parameter.

generate_text(config, prompt)
    One-shot content generation — used by LLMReasoner so that path also flows
    through enterprise governance when the Sage backend is active.
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
        # Local workshop layout: SDK lives as a sibling directory of the
        # project root.  From this file (smart_assignment/shared/llm.py),
        # parents[3] is the same workspace root that tmp.py reached via
        # parents[1].
        workspace_root = Path(__file__).resolve().parents[3]
        sdk_root = workspace_root / "sage-ai-sdk-python-sage-adk_1.0.0"
        local_src_paths = [
            sdk_root / "sage_adk" / "src",
            sdk_root / "sage_core" / "src",
            sdk_root / "sage_client" / "src",
        ]
        for src_path in local_src_paths:
            if src_path.exists():
                sys.path.insert(0, str(src_path))

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_llm(config: "Config") -> Any:
    """
    Return the value for an ADK LlmAgent ``model=`` parameter.

    sage     → SageLlmRegistry LLM object (enterprise-governed, TLS-injected)
    standard → plain model-name string (standard ADK / local dev)
    """
    if config.llm_backend == "sage":
        _check_sage_env_vars()
        return _load_sage_registry().get_llm(config.sage_model)
    return config.model


def generate_text(config: "Config", prompt: str) -> str:
    """
    One-shot content generation that honours the backend toggle.

    sage     → SageLlmRegistry LLM object via ADK BaseLlm (enterprise-governed)
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

    # standard path — matches the original LLMReasoner implementation
    from google import genai  # type: ignore[import-untyped]

    client = genai.Client()
    resp = client.models.generate_content(model=config.model, contents=prompt)
    return (resp.text or "").strip()

