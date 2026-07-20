"""
Hermetic tests for eval/capture.py's load_captured_results()/load_captured_outcomes()
-- no LLM backend needed. Uses a scratch file (monkeypatched _CAPTURED_PATH) so
these never depend on -- or mutate -- the real committed
eval/data/captured_responses.json.
"""

from __future__ import annotations

import json

import eval.capture as capture_mod
from eval.capture import CaptureResult, load_captured_outcomes, load_captured_results


def test_returns_empty_dict_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", tmp_path / "missing.json")
    assert load_captured_results() == {}
    assert load_captured_outcomes() == {}


def test_new_format_entry_parses_final_response_and_escalated(tmp_path, monkeypatch):
    path = tmp_path / "captured.json"
    path.write_text(
        json.dumps({"some_case": {"final_response": "the brief text", "escalated": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    results = load_captured_results()
    assert results == {"some_case": CaptureResult("the brief text", True)}
    assert load_captured_outcomes() == {"some_case": True}


def test_legacy_plain_string_entry_has_unknown_escalated(tmp_path, monkeypatch):
    # Captured before outcome-tracking was added -- escalated must read as
    # None (unknown), not silently guessed as True/False.
    path = tmp_path / "captured.json"
    path.write_text(json.dumps({"legacy_case": "just the text, no dict wrapper"}), encoding="utf-8")
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    results = load_captured_results()
    assert results == {"legacy_case": CaptureResult("just the text, no dict wrapper", None)}
    assert load_captured_outcomes() == {"legacy_case": None}


def test_mixed_legacy_and_new_format_entries(tmp_path, monkeypatch):
    path = tmp_path / "captured.json"
    path.write_text(
        json.dumps(
            {
                "legacy": "old text",
                "recommend_case": {"final_response": "clear response", "escalated": False},
                "escalate_case": {"final_response": "handoff brief", "escalated": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    outcomes = load_captured_outcomes()
    assert outcomes == {"legacy": None, "recommend_case": False, "escalate_case": True}
