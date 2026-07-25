"""
Hermetic tests for eval/deepeval_llm.py's SmartAssignmentDeepEvalLLM -- no real
LLM backend needed (the shared/llm.py boundary, generate_text, is mocked).
Skipped cleanly if the optional `eval-quality` extra (deepeval) isn't
installed -- this file must never become a hard hermetic-suite dependency.
Verifies:

* load_model() returns self (no separate model object to construct).
* generate() calls shared/llm.py's generate_text() with the stored config and
  the quality_judge role label.
* a_generate() drives generate() through the REAL offload_to_worker_thread
  (not mocked) -- proving the async path actually works, not just that it was
  called -- and returns the same text.
* get_model_name() reports sage_model under the sage backend, model otherwise
  -- matching generate_text()'s own active_model selection exactly.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip(
    "deepeval", reason="install the eval-quality extra: pip install -e '.[eval-quality]'"
)

from eval.deepeval_llm import SmartAssignmentDeepEvalLLM  # noqa: E402
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


def test_a_generate_drives_generate_through_the_real_offload_helper(monkeypatch):
    """Uses the REAL offload_to_worker_thread (from shared/llm.py), only
    mocking the generate_text call underneath it -- proving the async bridge
    itself works, not just that some mock got invoked."""
    import eval.deepeval_llm as adapter_mod

    def fake_generate_text(config, prompt, role=None):
        return f"async verdict for: {prompt}"

    monkeypatch.setattr(adapter_mod, "generate_text", fake_generate_text)

    cfg = Config(llm_backend="standard", model="m")
    judge = SmartAssignmentDeepEvalLLM(cfg)

    result = asyncio.run(judge.a_generate("prompt-x"))
    assert result == "async verdict for: prompt-x"


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
