"""
ADK-registry adapter so ADK-INTERNAL model resolution can also address a
Sage-approved model -- distinct from this repo's own ``LlmAgent``/
``generate_text`` calls, which already route through ``shared/llm.py``'s
``get_llm()``/``get_sage_llm()``.

Why this exists: ADK's own ``AgentEvaluator`` resolves its LLM-as-judge model
(``final_response_match_v2``, see ``eval/test_response_match.py``) via
``google.adk.models.registry.LLMRegistry.resolve(judge_model_string)`` --
ADK-core's OWN generic provider registry (a bare "gemini-*" string resolves to
its built-in ``Gemini`` class, "openai/*" to ``LiteLlm``, etc. -- see
``google/adk/models/__init__.py``'s ``_LAZY_PROVIDERS``
[VERIFIED against installed google-adk 2.5.0 source]). That registry has no
pattern matching "sage-*" and never calls into this repo's ``shared/llm.py``,
so:

* a Sage-prefixed ``judge_model`` string -> ``LLMRegistry.resolve()`` finds no
  matching pattern -> ``ValueError`` ("model not found").
* a bare (non-Sage) Gemini ``judge_model`` -> resolves fine, but calls the
  public Google API directly -- unreachable in a Sage-only environment where
  only Sage-approved models may be called at all.

The fix: register a ``BaseLlm`` subclass with ADK's ``LLMRegistry`` against a
"sage-.*" pattern, so ``LLMRegistry.resolve("sage-<model>")`` finds it and ADK
constructs it the normal way (``cls(model="sage-<model>")``). Internally it
just delegates to the SAME ``SageLlmRegistry`` LLM object ``shared/llm.py``'s
own ``get_sage_llm()``/``generate_text()`` already drive (see
``_generate_via_sage_async`` there) -- this is a thin ADK-registry bridge, not
a second Sage integration.

Registering the class costs nothing and needs no credentials -- only
``SageJudgeLlm.supported_models()``, plain class metadata, is read at
registration time. The actual Sage SDK import + auth stays lazy inside
``__init__`` (via ``get_sage_llm()``), exactly like ``shared/llm.py``'s own
``_load_sage_registry()``, so importing this module is safe regardless of
backend -- consistent with this repo's "credential-free import" rule.

Call ``register_sage_judge_model()`` once before any ADK evaluation that might
need to resolve a "sage-*" model (see ``eval/test_response_match.py``); repeat
calls are harmless (``LLMRegistry._register`` just overwrites the same
mapping, logging an info line).

[NOT independently verified against live Sage infrastructure in this repo's
dev environment -- only against real Sage credentials/network access can this
actually be exercised end-to-end; verified here only that registration and
resolution wire up correctly (see tests/eval/test_sage_judge_llm.py) and that
the delegation matches shared/llm.py's own established call pattern.]
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, List

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from pydantic import PrivateAttr

from smart_assignment.shared.llm import get_sage_llm


class SageJudgeLlm(BaseLlm):
    """Adapts a SageLlmRegistry LLM object to ADK's ``BaseLlm`` interface, so
    ADK-internal model resolution can address it by a "sage-*" model string --
    the same object ``shared/llm.py``'s own ``get_sage_llm()`` already drives
    for this repo's OWN agent/grounded calls."""

    _sage_llm: Any = PrivateAttr()

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._sage_llm = get_sage_llm(model)

    @classmethod
    def supported_models(cls) -> List[str]:
        return [r"sage-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        async for response in self._sage_llm.generate_content_async(llm_request, stream=stream):
            yield response


def register_sage_judge_model() -> None:
    """Idempotent. Registers ``SageJudgeLlm`` against the "sage-.*" pattern so
    ADK-internal model resolution (``LLMRegistry.resolve``) can find it. Safe
    to call regardless of backend/credentials -- see module docstring."""
    LLMRegistry.register(SageJudgeLlm)
