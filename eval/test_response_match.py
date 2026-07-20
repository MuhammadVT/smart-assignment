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

--- final_response_match_v2 (Phase 2b-2): an LLM-as-judge alternative, scored
alongside response_match_score, NOT instead of it ---

response_match_score (v1) is literal ROUGE-1 word overlap -- cheap (no extra LLM
call) but brittle: a correct, differently-worded response can score low.
final_response_match_v2 asks a judge LLM whether the response is valid given the
reference, tolerating paraphrasing/format/order differences -- a materially
better quality signal for prose, at a materially higher cost.

It has the EXACT SAME escalate-case blind spot as v1, verified from the same ADK
source read: [VERIFIED against installed google-adk 2.5.0]
``llm_as_judge_utils.get_text_from_content`` -- even with
``include_intermediate_responses_in_final=True`` -- still bottoms out in a
``.text``-only read of ``Content.parts`` for every event it walks, including the
one holding the escalation handoff (a ``function_call``, not text). So it is
scoped by the exact same ``_recommend_only_eval_ids()`` filter as v1, not the
full dataset.

Cost note: ``LlmAsAJudgeCriterion``'s ``JudgeModelOptions.num_samples`` defaults
to 5 in ADK -- the judge model is sampled 5x per invocation and majority-voted,
i.e. 5 extra LLM calls per case on top of the agent's own live run. Pinned to 1
here deliberately (cheap while iterating with few captured cases); raise it once
there's a reason to trust majority-vote stability over a single judge call. Also
marked ``@experimental`` in ADK's own source -- expect this metric's shape or
behavior to move under future ADK versions.

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
from smart_assignment.shared.config import DEFAULT_CONFIG

AGENT_MODULE_PATH = "smart_assignment"

# Starting points, not calibrated numbers -- both can legitimately vary run to
# run. Revisit once more recommend-outcome cases are captured and there's a real
# distribution of scores to look at.
_RESPONSE_MATCH_THRESHOLD = 0.5  # v1: ROUGE-1 f-measure, [0, 1].
_JUDGE_MATCH_THRESHOLD = 0.5  # v2: fraction of judge samples rating "valid", [0, 1].
_JUDGE_NUM_SAMPLES = 1  # ADK's own default is 5; see module docstring on cost.

# ADK's own JudgeModelOptions.judge_model default is "gemini-2.5-flash", which
# Google has since retired (404 on every real API key -- see the model default
# in shared/config.py for the same issue on this repo's OWN calls), so it must
# always be pinned explicitly. WHICH model depends on the active backend:
#
# "standard" -> a bare, currently-live Gemini id. ADK's judge always calls the
#   raw Gemini API directly (GoogleLLMVariant.GEMINI_API) via its own generic
#   LLMRegistry (bare "gemini-*" -> its built-in Gemini class), never through
#   this repo's shared/llm.py -- so it needs its own bare model id, not
#   whatever SMART_ASSIGNMENT_MODEL happens to be (which under "standard" may
#   itself be a "<provider>/<model>" litellm string ADK's judge can't use).
#
# "sage" -> a Sage-prefixed id, reusing whatever SMART_ASSIGNMENT_SAGE_MODEL is
#   already configured (never invented here -- see eval/sage_judge_llm.py's
#   docstring). ADK's LLMRegistry has no pattern matching "sage-*" out of the
#   box, so register_sage_judge_model() registers an adapter class first; a
#   Sage-only environment (no direct non-Sage-approved model access) cannot
#   reach the "standard" branch's bare Gemini id at all.
if DEFAULT_CONFIG.llm_backend == "sage":
    from eval.sage_judge_llm import register_sage_judge_model

    register_sage_judge_model()
    _JUDGE_MODEL = DEFAULT_CONFIG.sage_model
else:
    _JUDGE_MODEL = "gemini-3.1-flash-lite"

_SCRATCH_TEST_CONFIG = {
    "criteria": {
        "tool_trajectory_avg_score": {"threshold": 1.0, "match_type": "IN_ORDER"},
        "response_match_score": {"threshold": _RESPONSE_MATCH_THRESHOLD},
        "final_response_match_v2": {
            "threshold": _JUDGE_MATCH_THRESHOLD,
            "judge_model_options": {
                "judge_model": _JUDGE_MODEL,
                "num_samples": _JUDGE_NUM_SAMPLES,
            },
        },
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
