"""
The grounded route-slot model helper (routeslot/llm.py) logs the raw backend
reply when it can't be parsed as JSON -- so a non-JSON/empty response (the sage
"JSONDecodeError: Expecting value: line 1 column 1 (char 0)" fallback) is
diagnosable from the logs instead of opaque. The exception still propagates so
the caller falls back deterministically.
"""

from __future__ import annotations

import json

import pytest

from smart_assignment.routeslot import llm as rsllm
from smart_assignment.shared import llm as shared_llm
from smart_assignment.shared.config import Config


def test_non_json_reply_is_logged_with_the_raw_text(caplog, monkeypatch):
    # Backend returns prose, not JSON.
    monkeypatch.setattr(shared_llm, "generate_text", lambda config, prompt: "not json at all")

    with caplog.at_level("WARNING"):
        with pytest.raises(json.JSONDecodeError):
            rsllm.generate_route_slot_choice(Config(), "prompt")

    # The raw reply is visible in the log (this is the whole point of the change).
    assert "not json at all" in caplog.text
    assert "was not JSON" in caplog.text


def test_empty_reply_is_logged_as_length_zero(caplog, monkeypatch):
    # The exact failing case: an empty reply -> json.loads("") -> char 0.
    monkeypatch.setattr(shared_llm, "generate_text", lambda config, prompt: "")

    with caplog.at_level("WARNING"):
        with pytest.raises(json.JSONDecodeError):
            rsllm.generate_route_slot_choice(Config(), "prompt")

    assert "len=0" in caplog.text


def test_valid_json_reply_still_parses_and_does_not_log(caplog, monkeypatch):
    monkeypatch.setattr(
        shared_llm, "generate_text", lambda config, prompt: '{"chosen_index": 0}'
    )

    with caplog.at_level("WARNING"):
        result = rsllm.generate_route_slot_choice(Config(), "prompt")

    assert result == {"chosen_index": 0}
    assert "was not JSON" not in caplog.text
