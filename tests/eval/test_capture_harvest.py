"""Hermetic tests for eval/capture_harvest.py -- keeping what a live eval run
said, without a live eval run.

ADK's ``InferenceResult``/``Invocation``/``InvocationEvent`` are duck-typed by
stubs shaped exactly as ADK builds them (see ``_invocation_contents``'s note on
``convert_events_to_eval_invocations``), so this needs no backend and no
``google-adk[eval]``.
"""

from __future__ import annotations

import json

import pytest

from eval.capture_harvest import RunHarvester, load_latest_run
from eval.response_extract import HANDOFF_MESSAGE_ARG, HANDOFF_TOOL_NAME


class _Call:
    def __init__(self, name, args=None):
        self.name, self.args = name, args


class _Part:
    def __init__(self, text=None, function_call=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _Content:
    def __init__(self, *parts):
        self.parts = list(parts)


class _Event:
    def __init__(self, content):
        self.author, self.content = "agent", content


class _Events:
    def __init__(self, *events):
        self.invocation_events = list(events)


class _Invocation:
    def __init__(self, events, final_response):
        self.intermediate_data = _Events(*events)
        self.final_response = final_response


class _Result:
    def __init__(self, eval_case_id, inferences):
        self.eval_case_id, self.inferences = eval_case_id, inferences


def _text(value):
    return _Content(_Part(text=value))


def _handoff(message):
    return _Content(_Part(function_call=_Call(HANDOFF_TOOL_NAME, {HANDOFF_MESSAGE_ARG: message})))


def _tool_call():
    return _Content(_Part(function_call=_Call("recommend_or_escalate", {})))


def _recommend_result(eval_id, answer):
    """ADK drops the final event from invocation_events when it carries no
    function call, exposing it only as final_response -- so the closing narration
    of a RECOMMEND lives there and nowhere else."""
    return _Result(eval_id, [_Invocation([_Event(_tool_call())], _text(answer))])


def _escalate_result(eval_id, brief):
    """The final event of an ESCALATE has function calls, so ADK keeps it in
    invocation_events AND exposes it as final_response -- present twice."""
    handoff = _handoff(brief)
    return _Result(eval_id, [_Invocation([_Event(_tool_call()), _Event(handoff)], handoff)])


@pytest.fixture
def harvest_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_ASSIGNMENT_EVAL_CASES", raising=False)
    path = tmp_path / "latest_run_responses.json"
    monkeypatch.setattr("eval.capture_harvest.HARVEST_PATH", path)
    return path


def _written(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_recommend_run_records_the_closing_narration(harvest_path):
    harvester = RunHarvester()
    harvester.observe(_recommend_result("case_a", "I recommend RTE-4100 on Tuesday."))
    assert harvester.write() == 1

    record = _written(harvest_path)["case_a"]
    assert record["final_response"] == "I recommend RTE-4100 on Tuesday."
    assert record["escalated"] is False


def test_an_escalate_run_records_the_handoff_brief(harvest_path):
    # The brief lives in the tool call's args, never in a text part -- the exact
    # blind spot that makes ADK's own response scorers useless on escalations.
    harvester = RunHarvester()
    harvester.observe(_escalate_result("case_b", "SITUATION\nGalleria ..."))
    harvester.write()

    record = _written(harvest_path)["case_b"]
    assert record["final_response"] == "SITUATION\nGalleria ..."
    assert record["escalated"] is True


def test_every_record_carries_provenance_and_a_timestamp(harvest_path):
    harvester = RunHarvester()
    harvester.observe(_recommend_result("case_a", "answer"))
    harvester.write()

    record = _written(harvest_path)["case_a"]
    assert record["captured_at"]
    assert record["captured_with"]["dataset"]["name"] == "mock"
    assert record["captured_with"]["backend"]
    assert record["captured_with"]["model"]
    # Same key set as the committed golden reference: one shape, one reader.
    assert set(record) == {
        "final_response",
        "escalated",
        "decision_id",
        "captured_at",
        "captured_with",
    }


def test_the_last_run_of_a_case_wins(harvest_path):
    # SMART_ASSIGNMENT_EVAL_NUM_RUNS above 1 yields one result per run per case.
    harvester = RunHarvester()
    harvester.observe(_recommend_result("case_a", "first answer"))
    harvester.observe(_recommend_result("case_a", "second answer"))
    assert harvester.write() == 1
    assert _written(harvest_path)["case_a"]["final_response"] == "second answer"


def test_a_failed_inference_contributes_nothing(harvest_path):
    # ADK marks a crashed case FAILURE and returns it with no usable inferences;
    # eval/inference_guard.py already fails the run over it.
    harvester = RunHarvester()
    harvester.observe(_Result("case_dropped", []))
    harvester.observe(_Result("case_none", None))
    assert harvester.write() == 0
    assert _written(harvest_path) == {}


def test_an_empty_run_writes_an_empty_file_rather_than_leaving_a_stale_one(harvest_path):
    harvest_path.write_text(
        json.dumps({"stale_case": {"final_response": "old", "escalated": False}})
    )
    RunHarvester().write()
    # The stale case must be gone: test_quality would otherwise judge last run's
    # prose as though this run had produced it.
    assert _written(harvest_path) == {}


def test_the_harvest_replaces_rather_than_merges(harvest_path):
    first = RunHarvester()
    first.observe(_recommend_result("case_a", "answer a"))
    first.write()

    second = RunHarvester()
    second.observe(_recommend_result("case_b", "answer b"))
    second.write()

    assert sorted(_written(harvest_path)) == ["case_b"]


def test_load_latest_run_round_trips_through_the_shared_reader(harvest_path):
    harvester = RunHarvester()
    harvester.observe(_recommend_result("case_a", "answer"))
    harvester.observe(_escalate_result("case_b", "brief"))
    harvester.write()

    loaded = load_latest_run(harvest_path)
    assert loaded["case_a"].final_response == "answer"
    assert loaded["case_a"].escalated is False
    assert loaded["case_b"].escalated is True
    assert loaded["case_a"].captured_with["dataset"]["name"] == "mock"


def test_load_latest_run_is_empty_when_no_run_has_written(tmp_path):
    assert load_latest_run(tmp_path / "nothing.json") == {}


def test_a_curated_case_records_its_decision_id(tmp_path, monkeypatch):
    # The join key calibration needs; None for the hand-written golden fixtures.
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "eval_id": "curated_x",
                    "context": {
                        "name": "C",
                        "address": "1200 McKinney St, Houston, TX 77010",
                        "order_quantity_cases": 90,
                    },
                    "observed_outcome": "recommend",
                    "provenance": {"decision_id": "abc123"},
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMART_ASSIGNMENT_EVAL_CASES", str(candidates))
    path = tmp_path / "latest.json"
    monkeypatch.setattr("eval.capture_harvest.HARVEST_PATH", path)

    harvester = RunHarvester()
    harvester.observe(_recommend_result("curated_x", "answer"))
    harvester.write()

    assert _written(path)["curated_x"]["decision_id"] == "abc123"
