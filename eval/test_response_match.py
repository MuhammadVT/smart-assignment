"""
Phase 2b verification: ``response_match_score`` on RECOMMEND-outcome cases only.

``response_match_score`` cannot meaningfully score an ESCALATE-outcome case, no
matter the response quality -- this is a real ADK limitation, not a threshold to
tune. [VERIFIED against installed google-adk 2.5.0 source]: an escalation ends the
turn on ADK's long-running ``request_input`` tool call. ``Event.is_final_response()``
(google/adk/events/event.py) returns True whenever ``long_running_tool_ids`` is
set, so ADK's own eval harness (``evaluation_generator.py``) treats that TOOL-CALL
event as the turn's "final response" -- but its ``content.parts`` holds a
``function_call`` part, not a ``.text`` part. ``RougeEvaluator._get_text_from_content``
(``final_response_match_v1.py``) only reads ``.text``, so the "actual" side is
always ``""`` for an escalated case, forcing ROUGE-1 to ``0.0`` regardless of how
good the real handoff message (visible in the tool call's own ``message`` arg) was.
``eval/capture.py`` works around this on the REFERENCE side only (it manually pulls
``message`` out of the long-running call for the case it captures) -- there is no
equivalent on the live/actual side during ``AgentEvaluator.evaluate()``, and that
extraction lives inside ADK's own internals, not something this repo controls.

So: the committed ``eval/data/test_config.json`` stays trajectory-only (unaffected
by any of this -- the default full-dataset run and CI keep working exactly as
before). This file scores ``response_match_score`` separately, against a scratch
dataset containing ONLY the captured cases ``eval/capture.py`` recorded as
``escalated: False`` -- same "render fresh, touch nothing under eval/data/"
discipline ``eval/case_selection.py``'s ``SMART_ASSIGNMENT_EVAL_IDS`` subset uses.

Skips cleanly (not a failure) when no case is yet known to be a clean recommend --
e.g. right after a capture that happened to escalate. Capture one with
``python3 -m eval.capture`` to unskip it.

Run with (needs a configured LLM backend): pytest eval/test_response_match.py
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from eval.build_evalset import render_dataset
from eval.capture import load_captured_outcomes
from eval.golden_cases import GOLDEN_CASES

AGENT_MODULE_PATH = "smart_assignment"

# A starting point, not a calibrated number -- ROUGE-1 on paraphrased structured
# text can legitimately vary run to run. Revisit once more recommend-outcome
# cases are captured and there's a real distribution to look at.
_RESPONSE_MATCH_THRESHOLD = 0.5

_SCRATCH_TEST_CONFIG = {
    "criteria": {
        "tool_trajectory_avg_score": {"threshold": 1.0, "match_type": "IN_ORDER"},
        "response_match_score": {"threshold": _RESPONSE_MATCH_THRESHOLD},
    }
}


def _recommend_only_eval_ids() -> list[str]:
    """Captured eval_ids known (not just assumed) to be a clean recommend --
    ``escalated is False`` exactly, so both a real escalate (``True``) and an
    unknown/legacy capture (``None``) are excluded. See module docstring for why
    an escalate case can never usefully be included here."""
    outcomes = load_captured_outcomes()
    return [eval_id for eval_id, escalated in outcomes.items() if escalated is False]


@pytest.mark.asyncio
async def test_response_match_on_recommend_cases():
    eval_ids = _recommend_only_eval_ids()
    if not eval_ids:
        pytest.skip(
            "No captured case is yet known to be a clean recommend (response_match_score "
            "can't meaningfully score an escalate case -- see this module's docstring). "
            "Capture one with `python3 -m eval.capture` (optionally scoped via "
            "SMART_ASSIGNMENT_EVAL_IDS to a case documented as 'recommend' in "
            "golden_cases.py) and re-run."
        )

    by_id = {case.eval_id: case for case in GOLDEN_CASES}
    cases = [by_id[eval_id] for eval_id in eval_ids]

    scratch_dir = pathlib.Path(tempfile.mkdtemp(prefix="smart_assignment_response_match_"))
    dataset_path = scratch_dir / "recommend_subset.test.json"
    dataset_path.write_text(render_dataset(cases), encoding="utf-8")
    (scratch_dir / "test_config.json").write_text(
        json.dumps(_SCRATCH_TEST_CONFIG, indent=2) + "\n", encoding="utf-8"
    )

    await AgentEvaluator.evaluate(
        agent_module=AGENT_MODULE_PATH,
        eval_dataset_file_path_or_dir=str(dataset_path),
    )
