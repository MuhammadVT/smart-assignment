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

--- Local dev cost knobs (both optional; NOT used by CI) ---

Every case runs the full agent pipeline against the live LLM, and ADK's own
default is to run each case TWICE (``num_runs=2``) -- e.g. all 4 committed
cases is 8 live conversations per run. Two env vars trim that while iterating:

* ``SMART_ASSIGNMENT_EVAL_IDS`` -- comma-separated eval_id subset (see the
  ``eval_id`` on each ``GoldenCase`` in golden_cases.py), e.g.
  ``SMART_ASSIGNMENT_EVAL_IDS=bayou_city_bistro_recommend``. Parsed by the
  shared ``eval/case_selection.py`` (also used by ``eval/capture.py``, so one
  setting trims cost across both). The subset is rendered fresh from
  golden_cases.py via the same ``build_evalset`` machinery that produces the
  committed dataset, so it can never drift from it, and is written to a
  scratch temp dir -- the committed JSON under eval/data/ is never touched,
  so there's nothing to accidentally commit.
* ``SMART_ASSIGNMENT_EVAL_NUM_RUNS`` -- overrides ADK's num_runs (default 2),
  e.g. ``SMART_ASSIGNMENT_EVAL_NUM_RUNS=1``.

Both unset (the default) reproduces prior behavior exactly: the full committed
dataset, ADK's own num_runs default. See "Running a subset locally while
developing" in eval/README.md.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from eval.build_evalset import render_dataset
from eval.case_selection import select_cases
from eval.golden_cases import GOLDEN_CASES

REPO_ROOT = pathlib.Path(__file__).parent.parent
AGENT_MODULE_PATH = "smart_assignment"
_DATA_DIR = REPO_ROOT / "eval" / "data"
_COMMITTED_DATASET = _DATA_DIR / "slot_recommendation.test.json"
_TEST_CONFIG = _DATA_DIR / "test_config.json"

_NUM_RUNS_ENV = "SMART_ASSIGNMENT_EVAL_NUM_RUNS"


def _eval_dataset_path() -> str:
    """The committed dataset, or -- when SMART_ASSIGNMENT_EVAL_IDS (see
    eval/case_selection.py) names a subset of golden eval_ids -- a scratch
    dataset containing only those cases (see module docstring)."""
    cases = select_cases(GOLDEN_CASES)
    if cases is GOLDEN_CASES:
        return str(_COMMITTED_DATASET)

    scratch_dir = pathlib.Path(tempfile.mkdtemp(prefix="smart_assignment_eval_subset_"))
    dataset_path = scratch_dir / "subset.test.json"
    dataset_path.write_text(render_dataset(cases), encoding="utf-8")
    # AgentEvaluator discovers criteria (IN_ORDER match_type, see golden_cases.py)
    # from a test_config.json in the SAME FOLDER as the dataset file. Without this,
    # the subset would silently fall back to ADK's stricter EXACT default and the
    # escalate cases would flake on their model-authored handoff args.
    shutil.copy(_TEST_CONFIG, scratch_dir / "test_config.json")
    return str(dataset_path)


@pytest.mark.asyncio
async def test_slot_recommendation_eval():
    kwargs = {}
    num_runs_raw = os.environ.get(_NUM_RUNS_ENV)
    if num_runs_raw and num_runs_raw.strip():
        kwargs["num_runs"] = int(num_runs_raw)

    await AgentEvaluator.evaluate(
        agent_module=AGENT_MODULE_PATH,
        eval_dataset_file_path_or_dir=_eval_dataset_path(),
        **kwargs,
    )
