"""Hermetic tests for the agent-eval scaffolding (no LLM backend needed).

They validate that the golden dataset is schema-valid for ADK's ``AgentEvaluator``
and that the committed JSON stays in sync with the deterministic builder -- so a
stale hand-edit, or a builder change that wasn't regenerated, is caught by the
normal ``pytest`` run even though the eval itself (which replays the live agent)
is not part of this suite.
"""

from __future__ import annotations

import pathlib

from google.adk.evaluation.eval_set import EvalSet

from eval.build_evalset import build_eval_set, load_captured, render_dataset
from eval.golden_cases import (
    _PIPELINE_AFTER_INTAKE,
    GOLDEN_CASES,
    expected_trajectory,
    intake_args,
)

_DATASET_PATH = (
    pathlib.Path(__file__).parents[2] / "eval" / "data" / "slot_recommendation.test.json"
)


def test_built_eval_set_validates_against_adk_schema():
    # If ADK's schema drifts or the builder emits a bad shape, this fails to load.
    eval_set = EvalSet.model_validate(build_eval_set())
    assert eval_set.eval_cases
    assert len(eval_set.eval_cases) == len(GOLDEN_CASES)


def test_committed_dataset_is_in_sync_with_builder():
    # The committed file must equal the builder's output byte-for-byte; regenerate
    # with `python3 -m eval.build_evalset` after changing golden_cases.py (or after
    # `python3 -m eval.capture`). render_dataset() reads the same committed capture
    # file the dataset was built from, so this holds in both Phase 2a and 2b.
    on_disk = _DATASET_PATH.read_text(encoding="utf-8")
    assert on_disk == render_dataset(), (
        "eval/data/slot_recommendation.test.json is stale; "
        "run `python3 -m eval.build_evalset` and commit the result."
    )


def test_absent_capture_leaves_final_response_null():
    # Phase-2a reproduction: with no captured responses, every final_response is
    # None -- structural output is unchanged.
    eval_set = EvalSet.model_validate(build_eval_set(captured={}))
    for case in eval_set.eval_cases:
        assert case.conversation[0].final_response is None


def test_captured_responses_populate_final_response_and_validate():
    # Phase-2b: a captured {eval_id: text} map lands as the model-role final
    # response, and the populated set still validates against ADK's schema. Uses an
    # injected map so the test stays hermetic (no dependency on the capture file).
    first = GOLDEN_CASES[0]
    captured = {first.eval_id: "We can deliver Tuesday 7-10am on route CH-1."}
    eval_set = EvalSet.model_validate(build_eval_set(captured=captured))

    by_id = {case.eval_id: case for case in eval_set.eval_cases}
    populated = by_id[first.eval_id].conversation[0].final_response
    assert populated is not None
    assert populated.parts[0].text == captured[first.eval_id]
    # Cases without a captured entry stay null.
    for case in eval_set.eval_cases:
        if case.eval_id != first.eval_id:
            assert case.conversation[0].final_response is None


def test_load_captured_returns_dict():
    # Whether or not the capture file exists yet, the loader yields a dict the
    # builder can index into.
    assert isinstance(load_captured(), dict)


def test_each_case_has_the_required_pipeline_trajectory():
    # Compared against _PIPELINE_AFTER_INTAKE itself, not a copy of it: a literal
    # duplicated here is exactly what let the dataset keep asserting a pipeline the
    # agent had stopped driving (see test_pinned_trajectory_excludes_on_demand_tools).
    eval_set = EvalSet.model_validate(build_eval_set())
    expected_names = ["intake_customer", *_PIPELINE_AFTER_INTAKE]
    for case in eval_set.eval_cases:
        invocation = case.conversation[0]
        names = [call.name for call in invocation.intermediate_data.tool_uses]
        assert names == expected_names, f"{case.eval_id} trajectory: {names}"


def test_pinned_trajectory_excludes_on_demand_tools():
    """The trajectory may only pin tools the agent is REQUIRED to call.

    ``find_candidate_routes``/``evaluate_and_score_routes`` are on-demand: the
    prompt sends the agent straight from intake to ``recommend_or_escalate``,
    which re-derives both internally. Pinning them makes every live eval case
    score 0.0 -- a failure only a credentialed ``eval/test_eval.py`` run can
    surface, which is why it went unnoticed once before. This catches it in the
    hermetic suite instead.
    """
    on_demand = {"find_candidate_routes", "evaluate_and_score_routes"}
    pinned = on_demand.intersection(_PIPELINE_AFTER_INTAKE)
    assert not pinned, (
        f"{sorted(pinned)} is/are on-demand, not part of the default flow "
        "(see smart_assignment/prompts.py). IN_ORDER matching already tolerates "
        "them when a user asks for them; pinning them fails every case."
    )


def test_intake_args_are_the_ground_truth_customer_fields():
    # The intake call's expected args must be the fixture's real fields, so the
    # trajectory expectation is grounded, not invented.
    for case in GOLDEN_CASES:
        args = intake_args(case.customer)
        assert args["address"] == case.customer.address
        assert args["order_quantity_cases"] == case.customer.order_quantity_cases
        slot = case.customer.preferred_slot
        if slot is None:
            assert "preferred_day" not in args
        else:
            assert args["preferred_day"] == slot.day.name
            assert args["preferred_window_start"] == slot.window[0].strftime("%H:%M")


def test_expected_trajectory_only_intake_carries_args():
    for case in GOLDEN_CASES:
        trajectory = expected_trajectory(case)
        assert trajectory[0][0] == "intake_customer"
        assert trajectory[0][1]  # non-empty args
        for name, args in trajectory[1:]:
            assert args == {}, f"{name} should take no args"


def test_eval_ids_are_unique():
    ids = [case.eval_id for case in GOLDEN_CASES]
    assert len(ids) == len(set(ids))
