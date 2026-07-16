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

from eval.build_evalset import build_eval_set, render_dataset
from eval.golden_cases import GOLDEN_CASES, expected_trajectory, intake_args

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
    # with `python3 -m eval.build_evalset` after changing golden_cases.py.
    on_disk = _DATASET_PATH.read_text(encoding="utf-8")
    assert on_disk == render_dataset(), (
        "eval/data/slot_recommendation.test.json is stale; "
        "run `python3 -m eval.build_evalset` and commit the result."
    )


def test_each_case_has_the_full_pipeline_trajectory():
    eval_set = EvalSet.model_validate(build_eval_set())
    expected_names = [
        "intake_customer",
        "find_candidate_routes",
        "evaluate_and_score_routes",
        "recommend_or_escalate",
    ]
    for case in eval_set.eval_cases:
        invocation = case.conversation[0]
        names = [call.name for call in invocation.intermediate_data.tool_uses]
        assert names == expected_names, f"{case.eval_id} trajectory: {names}"


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
