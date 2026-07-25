"""
Hermetic tests for eval/sage_judge_llm.py's ADK-registry adapter -- no real
Sage credentials needed (the SDK boundary is mocked, same pattern as
tests/shared/test_llm.py's sage tests). Verifies:

* Registration makes ADK's own LLMRegistry resolve a "sage-*" model string to
  SageJudgeLlm, instead of raising (its default has no matching pattern).
* Constructing SageJudgeLlm routes through shared/llm.py's get_sage_llm() --
  the SAME seam this repo's own get_llm()/generate_text() already use --
  rather than a second, divergent Sage integration.
* generate_content_async transparently delegates to the wrapped Sage LLM
  object's own generate_content_async.

What this does NOT verify: an actual round-trip against live Sage
infrastructure -- that needs real SAGE_* credentials and network access (see
eval/sage_judge_llm.py's module docstring).
"""

from __future__ import annotations

import asyncio

from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry

from eval.sage_judge_llm import SageJudgeLlm, register_sage_judge_model


def test_register_makes_sage_prefixed_models_resolvable():
    register_sage_judge_model()
    assert LLMRegistry.resolve("sage-gemini-2.5-flash") is SageJudgeLlm


def test_register_is_idempotent():
    register_sage_judge_model()
    register_sage_judge_model()
    assert LLMRegistry.resolve("sage-anything") is SageJudgeLlm


def test_construction_routes_through_shared_llm_get_sage_llm(monkeypatch):
    """The adapter must reuse shared/llm.py's own Sage seam, not build a second
    one -- so a fake there is enough to prove the wiring, with no real SDK."""
    from smart_assignment.shared import llm as llm_mod

    calls = []

    class FakeSageLlm:
        async def generate_content_async(self, request, stream=False):
            yield LlmResponse()

    def fake_get_sage_llm(sage_model: str):
        calls.append(sage_model)
        return FakeSageLlm()

    monkeypatch.setattr(llm_mod, "get_sage_llm", fake_get_sage_llm)
    # eval.sage_judge_llm imported get_sage_llm by name into its own module
    # namespace, so the patch must target that reference too.
    import eval.sage_judge_llm as adapter_mod

    monkeypatch.setattr(adapter_mod, "get_sage_llm", fake_get_sage_llm)

    judge = SageJudgeLlm(model="sage-gemini-2.5-flash")

    assert calls == ["sage-gemini-2.5-flash"]
    assert judge.model == "sage-gemini-2.5-flash"


def test_generate_content_async_delegates_to_the_wrapped_sage_llm(monkeypatch):
    expected = LlmResponse()

    class FakeSageLlm:
        async def generate_content_async(self, request, stream=False):
            assert request == "the-request"
            assert stream is True
            yield expected

    import eval.sage_judge_llm as adapter_mod

    monkeypatch.setattr(adapter_mod, "get_sage_llm", lambda model: FakeSageLlm())

    judge = SageJudgeLlm(model="sage-gemini-2.5-flash")

    async def collect():
        return [r async for r in judge.generate_content_async("the-request", stream=True)]

    results = asyncio.run(collect())
    assert results == [expected]
