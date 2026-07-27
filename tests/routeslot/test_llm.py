"""
The grounded route-slot model helper (routeslot/llm.py). Structured output comes
from a FUNCTION CALL: `generate_tool_call` offers the model one tool whose
arguments ARE the decision, and those args are used directly. When the model
narrates instead of calling the tool, the prose is parsed/repaired (strict ->
brace-slice -> json_repair) and, when nothing parses, the raw reply is logged so
the sage "JSONDecodeError: Expecting value: line 1 column 1 (char 0)" fallback is
diagnosable. The exception still propagates so the caller falls back
deterministically.
"""

from __future__ import annotations

import json

import pytest

from smart_assignment.routeslot import llm as rsllm
from smart_assignment.shared import llm as shared_llm
from smart_assignment.shared.config import Config


def _fake_tool_call(result):
    """Build a fake `generate_tool_call` returning `result` = (call_args, text)."""
    return lambda config, prompt, tool, role=None: result


# --- structured (tool-call) path: the reliable channel ------------------------


def test_tool_call_args_are_used_directly(caplog, monkeypatch):
    # The model CALLED the tool -> its (SDK-repaired) args are the choice dict.
    monkeypatch.setattr(
        shared_llm, "generate_tool_call", _fake_tool_call(({"chosen_index": 2}, ""))
    )

    with caplog.at_level("WARNING"):
        result = rsllm.generate_route_slot_choice(Config(), "prompt")

    assert result == {"chosen_index": 2}
    assert "was not JSON" not in caplog.text


def test_tool_declaration_is_passed_through(monkeypatch):
    # The route-slot decision tool (with its function name) is what gets offered.
    seen = {}

    def capture(config, prompt, tool, role=None):
        seen["tool"] = tool
        return {"chosen_index": 0}, ""

    monkeypatch.setattr(shared_llm, "generate_tool_call", capture)
    rsllm.generate_route_slot_choice(Config(), "prompt")

    assert seen["tool"]["name"] == "submit_route_slot_decision"
    assert "chosen_index" in seen["tool"]["parameters"]["properties"]


# --- narration fallback: parse / repair the prose -----------------------------


def test_prose_wrapped_json_is_salvaged(caplog, monkeypatch):
    # Model narrated but embedded a JSON object -> json_repair lifts it out.
    monkeypatch.setattr(
        shared_llm,
        "generate_tool_call",
        _fake_tool_call((None, 'I recommend {"chosen_index": 0} for this prospect.')),
    )

    with caplog.at_level("WARNING"):
        result = rsllm.generate_route_slot_choice(Config(), "prompt")

    assert result == {"chosen_index": 0}
    assert "was not JSON" not in caplog.text


def test_pure_prose_reply_is_logged_with_the_raw_text(caplog, monkeypatch):
    # No tool call and no JSON anywhere -> unparseable; the raw prose is logged.
    monkeypatch.setattr(
        shared_llm,
        "generate_tool_call",
        _fake_tool_call((None, "I have selected the best route-slot option.")),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(json.JSONDecodeError):
            rsllm.generate_route_slot_choice(Config(), "prompt")

    assert "I have selected the best route-slot option." in caplog.text
    assert "was not JSON" in caplog.text


def test_empty_reply_is_logged_as_length_zero(caplog, monkeypatch):
    # The exact failing case: no tool call, empty text -> json.loads("") -> char 0.
    monkeypatch.setattr(shared_llm, "generate_tool_call", _fake_tool_call((None, "")))

    with caplog.at_level("WARNING"):
        with pytest.raises(json.JSONDecodeError):
            rsllm.generate_route_slot_choice(Config(), "prompt")

    assert "len=0" in caplog.text


def test_narrated_bare_json_still_parses(caplog, monkeypatch):
    # A backend that returns a bare JSON string (no tool call) still works.
    monkeypatch.setattr(
        shared_llm, "generate_tool_call", _fake_tool_call((None, '{"chosen_index": 0}'))
    )

    with caplog.at_level("WARNING"):
        result = rsllm.generate_route_slot_choice(Config(), "prompt")

    assert result == {"chosen_index": 0}
    assert "was not JSON" not in caplog.text
