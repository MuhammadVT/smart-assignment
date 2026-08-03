"""Hermetic tests for eval/inference_guard.py (no LLM backend, no live agent).

The guard exists because ADK drops a crashed eval case instead of failing it, so
a run can report a pass having scored only the survivors. These tests drive that
exact seam -- ``LocalEvalService.perform_inference`` streaming ``InferenceResult``s
-- with stubs, so the behavior is pinned without a backend.
"""

from __future__ import annotations

import asyncio

import pytest

from eval.inference_guard import DroppedEvalCasesError, fail_on_dropped_cases

# google-adk[eval] is an optional extra (see pyproject); the hermetic suite must
# never require it, so skip rather than fail when it is absent.
base_eval_service = pytest.importorskip("google.adk.evaluation.base_eval_service")
local_eval_service = pytest.importorskip("google.adk.evaluation.local_eval_service")

InferenceResult = base_eval_service.InferenceResult
InferenceStatus = base_eval_service.InferenceStatus
LocalEvalService = local_eval_service.LocalEvalService


def _result(eval_case_id: str, status, error_message=None) -> InferenceResult:
    return InferenceResult(
        app_name="test_app",
        eval_set_id="set",
        eval_case_id=eval_case_id,
        session_id="session",
        status=status,
        error_message=error_message,
    )


def _stub_perform_inference(results):
    """A stand-in for ADK's streaming perform_inference that yields `results`."""

    async def _perform_inference(self, inference_request=None):
        for result in results:
            yield result

    return _perform_inference


def _drain(monkeypatch, results):
    """Run the guard over a stubbed inference stream; return the collected drops.

    Raises whatever the guard raises, so a test can assert on it.
    """
    monkeypatch.setattr(
        LocalEvalService, "perform_inference", _stub_perform_inference(results)
    )

    async def _consume():
        async for _ in LocalEvalService.perform_inference(None, inference_request=None):
            pass

    with fail_on_dropped_cases() as dropped:
        asyncio.run(_consume())
    return dropped


def test_all_cases_succeeding_raises_nothing(monkeypatch):
    dropped = _drain(
        monkeypatch,
        [
            _result("bayou_city_bistro_recommend", InferenceStatus.SUCCESS),
            _result("woodlands_fresh_cafe_recommend", InferenceStatus.SUCCESS),
        ],
    )
    assert dropped == []


def test_a_dropped_case_fails_the_run_and_is_named(monkeypatch):
    with pytest.raises(DroppedEvalCasesError) as excinfo:
        _drain(
            monkeypatch,
            [
                _result("bayou_city_bistro_recommend", InferenceStatus.SUCCESS),
                _result(
                    "galleria_grill_escalate_low_score",
                    InferenceStatus.FAILURE,
                    error_message="Request to /generic/answer timed out",
                ),
            ],
        )

    message = str(excinfo.value)
    # Actionable: which case, why, and that it was never scored.
    assert "galleria_grill_escalate_low_score" in message
    assert "timed out" in message
    assert "NEVER SCORED" in message
    # The healthy case is not implicated.
    assert "bayou_city_bistro_recommend" not in message


def test_every_dropped_case_is_reported_not_just_the_first(monkeypatch):
    with pytest.raises(DroppedEvalCasesError) as excinfo:
        _drain(
            monkeypatch,
            [
                _result("galleria_grill_escalate_low_score", InferenceStatus.FAILURE, "a"),
                _result("katy_prairie_escalate_out_of_range", InferenceStatus.FAILURE, "b"),
            ],
        )

    message = str(excinfo.value)
    assert "2 eval case(s)" in message
    assert "galleria_grill_escalate_low_score" in message
    assert "katy_prairie_escalate_out_of_range" in message


def test_results_stream_through_unchanged(monkeypatch):
    # The guard only observes: every result must reach the caller, in order, as the
    # same object -- ADK's own scoring depends on it.
    results = [
        _result("a", InferenceStatus.SUCCESS),
        _result("b", InferenceStatus.FAILURE, "boom"),
        _result("c", InferenceStatus.SUCCESS),
    ]
    monkeypatch.setattr(
        LocalEvalService, "perform_inference", _stub_perform_inference(results)
    )
    seen = []

    async def _consume():
        async for result in LocalEvalService.perform_inference(
            None, inference_request=None
        ):
            seen.append(result)

    with pytest.raises(DroppedEvalCasesError):
        with fail_on_dropped_cases():
            asyncio.run(_consume())

    assert [r.eval_case_id for r in seen] == ["a", "b", "c"]
    assert all(seen[i] is results[i] for i in range(len(results)))


def test_the_patch_is_always_undone(monkeypatch):
    stub = _stub_perform_inference([_result("a", InferenceStatus.FAILURE, "boom")])
    monkeypatch.setattr(LocalEvalService, "perform_inference", stub)

    with pytest.raises(DroppedEvalCasesError):
        with fail_on_dropped_cases():
            async def _consume():
                async for _ in LocalEvalService.perform_inference(
                    None, inference_request=None
                ):
                    pass

            asyncio.run(_consume())

    assert LocalEvalService.perform_inference is stub


def test_a_failure_inside_the_block_is_not_masked(monkeypatch):
    # ADK's own metric assertion is the more specific failure and must win; the
    # dropped cases are logged instead of replacing it.
    stub = _stub_perform_inference([_result("a", InferenceStatus.FAILURE, "boom")])
    monkeypatch.setattr(LocalEvalService, "perform_inference", stub)

    with pytest.raises(AssertionError, match="tool_trajectory_avg_score"):
        with fail_on_dropped_cases():

            async def _consume():
                async for _ in LocalEvalService.perform_inference(
                    None, inference_request=None
                ):
                    pass

            asyncio.run(_consume())
            raise AssertionError("tool_trajectory_avg_score Failed")

    assert LocalEvalService.perform_inference is stub  # still restored
