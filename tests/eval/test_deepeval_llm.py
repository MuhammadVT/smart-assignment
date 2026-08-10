"""
Hermetic tests for eval/deepeval_llm.py's SmartAssignmentDeepEvalLLM -- no real
LLM backend needed (the shared/llm.py boundary, generate_text, is mocked).
Skipped cleanly if the optional `eval-quality` extra (deepeval) isn't
installed -- this file must never become a hard hermetic-suite dependency.
Verifies:

* load_model() returns self (no separate model object to construct).
* generate() calls shared/llm.py's generate_text() with the stored config and
  the quality_judge role label.
* a_generate() drives generate() off the calling loop (not mocked) -- proving
  the async path actually works, not just that it was called.
* generate(prompt, schema=...) returns an INSTANCE of that schema, sourced from
  a tool call, and salvages narrated JSON when the model ignores the tool.
* get_model_name() reports sage_model under the sage backend, model otherwise
  -- matching generate_text()'s own active_model selection exactly.

The schema path is the one that matters most here: G-Eval calls
``generate(prompt, schema=Steps)`` and silently retries WITHOUT the schema on
``TypeError``, so a signature regression would not raise -- it would quietly put
the judge back on the prose path that produced JSONDecodeError against the
narrating SAGE agent. These tests pin the signature and the return type.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip(
    "deepeval", reason="install the eval-quality extra: pip install -e '.[eval-quality]'"
)

from deepeval.metrics.g_eval.schema import ReasonScore, Steps  # noqa: E402

from eval.deepeval_llm import SmartAssignmentDeepEvalLLM, tool_for_schema  # noqa: E402
from smart_assignment.shared.config import Config  # noqa: E402


def test_load_model_returns_self():
    judge = SmartAssignmentDeepEvalLLM(Config(llm_backend="standard", model="m"))
    assert judge.load_model() is judge


def test_generate_routes_through_shared_llm_generate_text(monkeypatch):
    import eval.deepeval_llm as adapter_mod

    captured = {}

    def fake_generate_text(config, prompt, role=None):
        captured["config"] = config
        captured["prompt"] = prompt
        captured["role"] = role
        return "the judge's verdict"

    monkeypatch.setattr(adapter_mod, "generate_text", fake_generate_text)

    cfg = Config(llm_backend="standard", model="gemini-3.1-flash-lite")
    judge = SmartAssignmentDeepEvalLLM(cfg)

    result = judge.generate("score this response")

    assert result == "the judge's verdict"
    assert captured["config"] is cfg
    assert captured["prompt"] == "score this response"
    assert captured["role"] == "quality_judge"


def test_a_generate_really_runs_generate_off_the_loop(monkeypatch):
    """Drives the real async path, mocking only the generate_text call underneath
    it -- proving the hand-off works, not just that some mock got invoked."""
    import eval.deepeval_llm as adapter_mod

    def fake_generate_text(config, prompt, role=None):
        return f"async verdict for: {prompt}"

    monkeypatch.setattr(adapter_mod, "generate_text", fake_generate_text)

    cfg = Config(llm_backend="standard", model="m")
    judge = SmartAssignmentDeepEvalLLM(cfg)

    result = asyncio.run(judge.a_generate("prompt-x"))
    assert result == "async verdict for: prompt-x"


# --- the schema path: structured verdicts via the tool channel ----------------


def _judge_returning(monkeypatch, call_args, text=""):
    """A judge whose generate_tool_call returns ``(call_args, text)``, capturing
    the tool declaration it was handed."""
    import eval.deepeval_llm as adapter_mod

    seen = {}

    def fake_generate_tool_call(config, prompt, tool, role=None):
        seen["tool"] = tool
        seen["prompt"] = prompt
        seen["role"] = role
        return call_args, text

    monkeypatch.setattr(adapter_mod, "generate_tool_call", fake_generate_tool_call)
    return SmartAssignmentDeepEvalLLM(Config(llm_backend="sage", sage_model="s")), seen


def test_a_schema_request_returns_an_instance_of_that_schema(monkeypatch):
    judge, seen = _judge_returning(monkeypatch, {"reason": "clear and specific", "score": 8.0})

    verdict = judge.generate("score this", schema=ReasonScore)

    assert isinstance(verdict, ReasonScore)
    assert (verdict.score, verdict.reason) == (8.0, "clear and specific")
    assert seen["tool"]["name"] == "submit_reason_score"
    assert seen["role"] == "quality_judge"


def test_a_schema_request_never_reaches_the_prose_path(monkeypatch):
    """generate_text must not be called at all when a schema is asked for --
    that fallback is exactly the bug (narration -> json.loads('') -> crash)."""
    import eval.deepeval_llm as adapter_mod

    def exploding_generate_text(*args, **kwargs):
        raise AssertionError("the schema path must not fall back to plain text")

    monkeypatch.setattr(adapter_mod, "generate_text", exploding_generate_text)
    judge, _ = _judge_returning(monkeypatch, {"steps": ["a", "b"]})

    assert judge.generate("list steps", schema=Steps).steps == ["a", "b"]


def test_a_list_valued_schema_round_trips(monkeypatch):
    judge, seen = _judge_returning(monkeypatch, {"steps": ["read the rubric", "compare"]})

    verdict = judge.generate("evaluation steps", schema=Steps)

    assert isinstance(verdict, Steps)
    assert verdict.steps == ["read the rubric", "compare"]
    assert seen["tool"]["parameters"]["properties"]["steps"]["type"] == "array"


def test_narrated_json_is_salvaged_when_the_model_ignores_the_tool(monkeypatch):
    """The SAGE agent is conversational and sometimes answers in prose. G-Eval's
    prompt asks for JSON, so the object is usually still in there."""
    narration = (
        "Sure! Here is my evaluation:\n"
        '```json\n{"reason": "omits the window", "score": 4.5}\n```\n'
        "Let me know if you want more detail."
    )
    judge, _ = _judge_returning(monkeypatch, None, narration)

    verdict = judge.generate("score this", schema=ReasonScore)
    assert (verdict.score, verdict.reason) == (4.5, "omits the window")


def test_unparseable_narration_fails_loudly_rather_than_inventing_a_score(monkeypatch):
    judge, _ = _judge_returning(monkeypatch, None, "I'm not able to evaluate that.")

    with pytest.raises(ValueError, match="neither a submit_reason_score call"):
        judge.generate("score this", schema=ReasonScore)


def test_a_schema_request_survives_the_async_path(monkeypatch):
    judge, _ = _judge_returning(monkeypatch, {"reason": "fine", "score": 7.0})

    verdict = asyncio.run(judge.a_generate("score this", schema=ReasonScore))
    assert isinstance(verdict, ReasonScore) and verdict.score == 7.0


def test_a_nested_schema_asks_deepeval_to_use_the_prose_path():
    """TypeError is the contract: G-Eval catches it and retries without a schema.
    Anything else would fail the metric outright."""
    from pydantic import BaseModel

    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        inner: Inner

    with pytest.raises(TypeError, match="nests another model"):
        tool_for_schema(Outer)


def test_the_declaration_carries_no_rubric_of_its_own():
    """G-Eval's prompt owns the rubric and the score range. A description that
    restated either could contradict the prompt the judge was actually given."""
    description = tool_for_schema(ReasonScore)["description"].lower()
    assert not any(word in description for word in ("0-10", "scale", "rubric", "criteria"))


def test_get_model_name_standard_backend():
    cfg = Config(llm_backend="standard", model="gemini-3.1-flash-lite", sage_model="sage-x")
    judge = SmartAssignmentDeepEvalLLM(cfg)
    assert judge.get_model_name() == "gemini-3.1-flash-lite"


def test_get_model_name_sage_backend():
    cfg = Config(
        llm_backend="sage", model="gemini-3.1-flash-lite", sage_model="sage-gemini-2.5-flash"
    )
    judge = SmartAssignmentDeepEvalLLM(cfg)
    assert judge.get_model_name() == "sage-gemini-2.5-flash"
