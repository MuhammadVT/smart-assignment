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
[VERIFIED against installed google-adk 2.3.0 source]). That registry has no
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

**Resolution alone is not enough**, and this cost ``final_response_match_v2``
every case it ever tried to score: ADK stamps the BARE judge-model id onto the
request it hands the resolved model, and ADK's ``LiteLlm`` lets that override the
Sage handler's own provider-qualified ``"<agent>/model"`` string -- so litellm
sees no provider and raises. ``_addressed_to_sage`` below re-addresses the
request; see its docstring for the verified ADK source references.

[Registration/resolution and the re-addressing are covered hermetically in
tests/eval/test_sage_judge_llm.py. The end-to-end round trip needs real SAGE_*
credentials; it has since been exercised against live Sage, which is how the
addressing bug above was found and confirmed.]
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
        async for response in self._sage_llm.generate_content_async(
            self._addressed_to_sage(llm_request), stream=stream
        ):
            yield response

    def _addressed_to_sage(self, llm_request: LlmRequest) -> LlmRequest:
        """The same request, re-addressed to the model string the Sage handler is
        actually registered under.

        Resolution is only half the job. [VERIFIED against installed google-adk
        2.3.0] ``LlmAsJudge._request_judge`` builds
        ``LlmRequest(model=judge_model_options.judge_model)`` -- the BARE id, e.g.
        ``"sage-gemini-2.5-flash"`` -- and ``LiteLlm.generate_content_async``
        (lite_llm.py, ``effective_model = llm_request.model or self.model``) lets
        the request's model win over the handler's own. But the Sage SDK registers
        its litellm custom provider as ``"<agent>/model"``, so the bare id reaches
        litellm with no provider prefix and ``get_llm_provider`` fails with
        ``BadRequestError: GetLLMProvider Exception - list index out of range``.
        That is why ``final_response_match_v2`` could never score a single case.

        This repo's own calls never hit it: ``shared/llm.py``'s
        ``_sage_call_async`` leaves ``LlmRequest.model`` unset, so the handler's
        model applies by default. The bug is reachable only through ADK-internal
        judging, which is exactly what this adapter fronts.

        Setting the qualified string explicitly, rather than clearing the field,
        keeps this correct whichever way ADK's precedence goes -- a future version
        that stops falling back to ``self.model`` would break a ``None`` here.

        A shallow ``model_copy`` on purpose: ADK's LiteLlm appends to
        ``contents`` as it prepares the call, and sharing that list is the
        pre-existing behavior of passing the request straight through. Only the
        addressing changes.
        """
        return llm_request.model_copy(update={"model": getattr(self._sage_llm, "model", None)})


def register_sage_judge_model() -> None:
    """Idempotent. Registers ``SageJudgeLlm`` against the "sage-.*" pattern so
    ADK-internal model resolution (``LLMRegistry.resolve``) can find it. Safe
    to call regardless of backend/credentials -- see module docstring."""
    LLMRegistry.register(SageJudgeLlm)
