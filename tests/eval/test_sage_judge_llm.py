"""
Hermetic tests for eval/sage_judge_llm.py's ADK-registry adapter -- no real
Sage credentials needed (the SDK boundary is mocked, same pattern as
tests/shared/test_llm.py's sage tests). Verifies:

* Registration makes ADK's own LLMRegistry resolve a "sage-*" model string to
  SageJudgeLlm, instead of raising (its default has no matching pattern).
* Constructing SageJudgeLlm routes through shared/llm.py's get_sage_llm() --
  the SAME seam this repo's own get_llm()/generate_text() already use --
  rather than a second, divergent Sage integration.
* generate_content_async delegates to the wrapped Sage LLM object, having
  re-addressed the request to the model string that object is registered
  under -- the fix for final_response_match_v2 failing on every case with
  "GetLLMProvider Exception - list index out of range".

What this does NOT verify: an actual round-trip against live Sage
infrastructure -- that needs real SAGE_* credentials and network access (see
eval/sage_judge_llm.py's module docstring).
"""

from __future__ import annotations

import asyncio

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types

from eval.sage_judge_llm import SageJudgeLlm, register_sage_judge_model

# What the Sage SDK registers its litellm custom provider as; the bare id alone
# means nothing to litellm.
_QUALIFIED = "sage-gemini-2.5-flash/model"


def _judge_wrapping(monkeypatch, fake):
    """A SageJudgeLlm whose Sage seam is ``fake`` -- no SDK, no credentials."""
    import eval.sage_judge_llm as adapter_mod

    monkeypatch.setattr(adapter_mod, "get_sage_llm", lambda model: fake)
    return SageJudgeLlm(model="sage-gemini-2.5-flash")


def _request(model=None):
    """A request shaped like the one ADK's LlmAsJudge builds."""
    return LlmRequest(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text="judge this")])],
    )


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


class _RecordingSageLlm:
    """Stands in for the ADK ``LiteLlm`` ``SageLlmRegistry.get_llm`` returns:
    carries the provider-qualified model and records what it was handed."""

    model = _QUALIFIED

    def __init__(self):
        self.seen = None
        self.stream = None

    async def generate_content_async(self, request, stream=False):
        self.seen, self.stream = request, stream
        yield LlmResponse()


def _drive(judge, request, stream=False):
    async def collect():
        return [r async for r in judge.generate_content_async(request, stream=stream)]

    return asyncio.run(collect())


def test_generate_content_async_delegates_to_the_wrapped_sage_llm(monkeypatch):
    fake = _RecordingSageLlm()
    judge = _judge_wrapping(monkeypatch, fake)

    results = _drive(judge, _request(), stream=True)

    assert len(results) == 1 and isinstance(results[0], LlmResponse)
    assert fake.stream is True


def test_adks_bare_judge_model_is_replaced_with_the_qualified_one(monkeypatch):
    """THE regression. ADK's LlmAsJudge sets ``LlmRequest.model`` to the bare
    judge-model id, and ADK's LiteLlm prefers it over the handler's own -- so the
    bare id reached litellm with no provider and every case died on
    "GetLLMProvider Exception - list index out of range"."""
    fake = _RecordingSageLlm()
    judge = _judge_wrapping(monkeypatch, fake)

    _drive(judge, _request(model="sage-gemini-2.5-flash"))

    assert fake.seen.model == _QUALIFIED


def test_a_request_with_no_model_is_addressed_too(monkeypatch):
    """Not merely "don't override": the field is set either way, so this keeps
    working if a future ADK stops falling back to the handler's own model."""
    fake = _RecordingSageLlm()
    judge = _judge_wrapping(monkeypatch, fake)

    _drive(judge, _request(model=None))

    assert fake.seen.model == _QUALIFIED


def test_re_addressing_changes_nothing_else_about_the_request(monkeypatch):
    fake = _RecordingSageLlm()
    judge = _judge_wrapping(monkeypatch, fake)
    original = _request(model="sage-gemini-2.5-flash")

    _drive(judge, original)

    assert fake.seen.contents == original.contents
    assert fake.seen.config == original.config
    # The caller's own object is left addressed as it was; ADK reuses it across
    # judge samples, and silently rewriting a caller's field is not ours to do.
    assert original.model == "sage-gemini-2.5-flash"


def test_a_handler_without_a_model_attribute_falls_back_to_adks_own_precedence(monkeypatch):
    """Defensive: if the SDK ever returns something without ``.model``, leave the
    field unset so ADK's ``llm_request.model or self.model`` still applies,
    rather than crash inside this adapter."""

    class ModellessSageLlm:
        def __init__(self):
            self.seen = None

        async def generate_content_async(self, request, stream=False):
            self.seen = request
            yield LlmResponse()

    fake = ModellessSageLlm()
    judge = _judge_wrapping(monkeypatch, fake)

    _drive(judge, _request(model="sage-gemini-2.5-flash"))

    assert fake.seen.model is None
