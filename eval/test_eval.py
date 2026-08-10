"""
Runs the smart_assignment conversational agent's golden dataset through
ADK's AgentEvaluator. It REPLAYS each scripted intake conversation against the
real ``root_agent`` -- so it needs a live LLM backend -- and scores TRAJECTORY
(did the agent drive the pipeline in the right order: intake_customer ->
recommend_or_escalate; see ``_PIPELINE_AFTER_INTAKE`` in golden_cases.py for the
tools deliberately left unpinned).

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

--- Cost knobs ---

Every case runs the full agent pipeline against the live LLM, so what this
suite costs is (cases x runs) live conversations.

* Each case is replayed ONCE by default, not twice as ADK would -- see
  ``eval/run_config.py`` for why, and set ``SMART_ASSIGNMENT_EVAL_NUM_RUNS``
  to replay more when run-to-run variance is the actual question.
* ``SMART_ASSIGNMENT_EVAL_CASES`` -- which case SET to score (see
  eval/case_set.py); defaults to the built-in golden fixtures, or point it at a
  curated candidates JSON to replay production-derived cases instead.
* ``SMART_ASSIGNMENT_EVAL_IDS`` -- comma-separated eval_id subset (see the
  ``eval_id`` on each ``GoldenCase`` in golden_cases.py), e.g.
  ``SMART_ASSIGNMENT_EVAL_IDS=woodlands_fresh_cafe_recommend``. A LOCAL-only
  knob (rejected under CI) parsed by the shared ``eval/case_selection.py``,
  which ``eval/capture.py`` also reads, so one setting trims cost across both.
  The subset is rendered fresh from golden_cases.py via the same
  ``build_evalset`` machinery that produces the committed dataset, so it can
  never drift from it, and is written to a scratch temp dir -- the committed
  JSON under eval/data/ is never touched, so there's nothing to accidentally
  commit.

See "Running a subset locally while developing" in eval/README.md.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from eval.build_evalset import render_dataset
from eval.case_selection import select_cases
from eval.case_set import resolve_case_set
from eval.inference_guard import fail_on_dropped_cases
from eval.run_budget import run_budget
from eval.run_config import resolve_num_runs

REPO_ROOT = pathlib.Path(__file__).parent.parent
AGENT_MODULE_PATH = "smart_assignment"
_DATA_DIR = REPO_ROOT / "eval" / "data"
_COMMITTED_DATASET = _DATA_DIR / "slot_recommendation.test.json"
_TEST_CONFIG = _DATA_DIR / "test_config.json"


def _eval_dataset_path() -> str:
    """The committed dataset, or a scratch dataset rendered on the fly.

    The committed file is used only when this run scores exactly what that file
    contains: the default golden case set (eval/case_set.py), unnarrowed by
    SMART_ASSIGNMENT_EVAL_IDS (eval/case_selection.py). A curated case set or an
    eval_id subset is rendered fresh instead, into a scratch temp dir -- so the
    committed JSON under eval/data/ is never touched and there is nothing to
    accidentally commit.

    The condition is checked explicitly rather than by object identity. It used
    to read ``if cases is GOLDEN_CASES``, which was never true: select_cases
    returns ``list(cases)``, a new object every time. So this always rendered a
    scratch copy -- harmless, because tests/eval/test_build_evalset.py pins the
    committed file to be byte-identical to render_dataset(), but not what the
    code said it did.
    """
    case_set = resolve_case_set()
    cases = select_cases(case_set.cases)
    # Compared as an ordered id list, not by count: SMART_ASSIGNMENT_EVAL_IDS
    # returns cases in the order named, so naming all of them in a different
    # order is still not the committed dataset.
    if case_set.is_default and [c.eval_id for c in cases] == [c.eval_id for c in case_set.cases]:
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
    # A case whose inference crashes is dropped by ADK, not failed -- so without
    # this guard the score is silently computed over only the survivors and the
    # run still reports a pass. See eval/inference_guard.py. The budget is the
    # outer ceiling: nothing else stops a hung backend running for hours (see
    # eval/run_budget.py).
    with fail_on_dropped_cases():
        async with run_budget():
            await AgentEvaluator.evaluate(
                agent_module=AGENT_MODULE_PATH,
                eval_dataset_file_path_or_dir=_eval_dataset_path(),
                num_runs=resolve_num_runs(),
            )
