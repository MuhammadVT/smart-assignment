"""
Runs the smart_assignment conversational agent's golden dataset through
ADK's AgentEvaluator. It REPLAYS each scripted intake conversation against the
real ``root_agent`` -- so it needs a live LLM backend -- and scores TRAJECTORY
(did the agent drive the pipeline in the right order: intake_customer ->
find_candidate_routes -> evaluate_and_score_routes -> recommend_or_escalate).

Phase 2a scores trajectory ONLY (see eval/data/test_config.json). The dataset is
generated deterministically from the repo's mock fixtures by
``eval/build_evalset.py`` (regenerate with ``python3 -m eval.build_evalset``);
its expected final responses are captured against a real backend and
final-response scoring is enabled in Phase 2b.

[VERIFIED against installed google-adk 2.3.0] AgentEvaluator.evaluate()
auto-discovers eval criteria from a `test_config.json` file located in
the SAME FOLDER as the `.test.json` dataset file (see
eval/data/test_config.json) -- it is not passed as an explicit argument.

This file is NOT part of the hermetic unit suite (pyproject sets
``testpaths = ["tests"]``); it runs only when explicitly targeted -- locally, or
in the advisory ``agent-eval`` CI job -- because it requires model credentials.

Run with (needs a configured LLM backend): pytest eval/test_eval.py
"""

from __future__ import annotations

import pathlib

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

REPO_ROOT = pathlib.Path(__file__).parent.parent
AGENT_MODULE_PATH = "smart_assignment"
EVAL_DATASET = str(REPO_ROOT / "eval" / "data" / "slot_recommendation.test.json")


@pytest.mark.asyncio
async def test_slot_recommendation_eval():
    await AgentEvaluator.evaluate(
        agent_module=AGENT_MODULE_PATH,
        eval_dataset_file_path_or_dir=EVAL_DATASET,
    )
