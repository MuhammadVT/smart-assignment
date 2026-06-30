"""
Runs the slot_recommendation workflow's golden dataset through ADK's
AgentEvaluator. This tests TRAJECTORY (did the workflow call nodes in the
right order, e.g. filter_feasible_slots_node before recommend_slot_agent)
and FINAL RESPONSE quality (does the recommendation match expectations),
not just whether the code runs.

[VERIFIED against installed google-adk 2.3.0] AgentEvaluator.evaluate()
auto-discovers eval criteria from a `test_config.json` file located in
the SAME FOLDER as the `.test.json` dataset file (see
eval/data/test_config.json) -- it is not passed as an explicit argument.

[ASSUMPTION] eval/data/slot_recommendation.test.json is currently a
placeholder -- no real route/capacity data was available to build a
genuine golden dataset. Populate it with real captured trajectories
(via the ADK Web UI) before relying on this for regression detection.

Run with: pytest eval/test_eval.py
"""

from __future__ import annotations

import pathlib

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

REPO_ROOT = pathlib.Path(__file__).parent.parent
AGENT_MODULE_PATH = "smart_assignment.workflows.slot_recommendation"
EVAL_DATASET = str(REPO_ROOT / "eval" / "data" / "slot_recommendation.test.json")


@pytest.mark.asyncio
async def test_slot_recommendation_eval():
    await AgentEvaluator.evaluate(
        agent_module=AGENT_MODULE_PATH,
        eval_dataset_file_path_or_dir=EVAL_DATASET,
    )
