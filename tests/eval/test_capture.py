"""Hermetic tests for eval/capture.py's load_captured_results()/load_captured_outcomes()
-- no LLM backend needed. Uses a scratch file (monkeypatched _CAPTURED_PATH) so
these never depend on -- or mutate -- the real committed
eval/data/golden_responses.json.
"""

from __future__ import annotations

import json

import pytest

import eval.capture as capture_mod
from eval.capture import CaptureResult, load_captured_outcomes, load_captured_results


def test_returns_empty_dict_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", tmp_path / "missing.json")
    assert load_captured_results() == {}
    assert load_captured_outcomes() == {}


def test_entry_parses_final_response_and_escalated(tmp_path, monkeypatch):
    path = tmp_path / "captured.json"
    path.write_text(
        json.dumps({"some_case": {"final_response": "the brief text", "escalated": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    results = load_captured_results()
    assert results == {"some_case": CaptureResult("the brief text", True)}
    assert load_captured_outcomes() == {"some_case": True}


def test_decision_id_is_carried_when_present(tmp_path, monkeypatch):
    # The key a judge verdict joins to a human label on (eval/judge_calibration.py).
    # Absent on the hand-written golden fixtures, present on curated cases.
    path = tmp_path / "captured.json"
    path.write_text(
        json.dumps(
            {
                "curated": {
                    "final_response": "text",
                    "escalated": False,
                    "decision_id": "1210bd3e4a984ff0bfb72a5426af3ed6",
                },
                "fixture": {"final_response": "text", "escalated": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    results = load_captured_results()
    assert results["curated"].decision_id == "1210bd3e4a984ff0bfb72a5426af3ed6"
    assert results["fixture"].decision_id is None


def test_mixed_outcomes_are_reported_exactly(tmp_path, monkeypatch):
    path = tmp_path / "captured.json"
    path.write_text(
        json.dumps(
            {
                "recommend_case": {"final_response": "clear response", "escalated": False},
                "escalate_case": {"final_response": "handoff brief", "escalated": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    assert load_captured_outcomes() == {"recommend_case": False, "escalate_case": True}


def test_a_plain_string_entry_raises_instead_of_loading_as_unknown(tmp_path, monkeypatch):
    # The pre-outcome-tracking format was {eval_id: text}, which had to be carried
    # as escalated=None ("unknown") and was then excluded from every scorer. The
    # rename to golden_responses.json is a clean break -- an old file is a
    # different filename and never read -- so a plain string can now only come
    # from a hand-edit, and a loud error beats a case that silently scores nothing.
    path = tmp_path / "captured.json"
    path.write_text(json.dumps({"legacy_case": "just the text, no dict wrapper"}), encoding="utf-8")
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)

    with pytest.raises(ValueError) as excinfo:
        load_captured_results()
    assert "legacy_case" in str(excinfo.value)


def test_entry_shape_written_by_capture(tmp_path, monkeypatch):
    # Pins the on-disk record so a field can't be dropped silently: the text, the
    # outcome, the join key, when it was captured, and what produced it.
    entry = capture_mod._entry(
        CaptureResult("text", escalated=False, decision_id="abc"),
        provenance={"dataset": {"name": "mock"}, "backend": "sage", "model": "m"},
        captured_at="2026-08-10T03:00:00+00:00",
    )
    assert entry == {
        "final_response": "text",
        "escalated": False,
        "decision_id": "abc",
        "captured_at": "2026-08-10T03:00:00+00:00",
        "captured_with": {"dataset": {"name": "mock"}, "backend": "sage", "model": "m"},
    }
    # ...and round-trips back through the reader unchanged.
    path = tmp_path / "captured.json"
    path.write_text(json.dumps({"c": entry}), encoding="utf-8")
    monkeypatch.setattr(capture_mod, "_CAPTURED_PATH", path)
    assert load_captured_results()["c"] == CaptureResult("text", False, "abc")
