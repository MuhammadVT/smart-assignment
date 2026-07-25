"""
Phase 3a: DeepEval G-Eval quality metrics -- reference-free rubrics scored
directly against ``eval/capture.py``'s already-captured
``{final_response, escalated}`` data, NOT through ADK's
``AgentEvaluator``/``EvalSet`` machinery (unlike ``test_eval.py``/
``test_response_match.py``): DeepEval's ``GEval`` metric scores a bare
``(input, actual_output)`` pair directly, so there is no ADK dataset file to
render or scratch ``test_config.json`` to write -- this file only READS
``eval/data/captured_responses.json``, never touches it.

Two rubrics, drawn directly from the human-annotation dimensions in
``deployment/phoenix/README.md``'s "human feedback" table (``brief_quality``,
``response_clarity``), so the automated score and the human-annotation
vocabulary stay aligned. The other two rows in that table --
``decision_correct`` (already covered deterministically by trajectory scoring's
``recommend_or_escalate`` call) and ``slot_reasonable`` -- and grounded-layer
rationale-faithfulness (a different granularity: judging the DECISION LAYER's
own reasoning, not the agent's final customer-facing prose) are deliberately
OUT of scope here; deferred to a later phase.

* ``brief_quality`` -- scored on ESCALATE-outcome captures
  (``escalated is True``). This is the highest-stakes prose
  ``response_match_score``/``final_response_match_v2`` structurally CANNOT
  score at all (ADK's ``request_input`` handoff ends the turn on a
  ``function_call``, not ``.text`` -- see ``test_response_match.py``'s module
  docstring for the full trace through ADK's source) -- the handoff brief is
  exactly what a human specialist acts on, so its quality matters most here.
* ``response_clarity`` -- scored on RECOMMEND-outcome captures
  (``escalated is False``), complementing ``response_match_score``/v2 (which
  check FIDELITY to a captured reference; this checks whether the message
  reads clearly on its own, reference-free -- no ``expected_output`` is set).

Judge model: ``eval/deepeval_llm.py``'s ``SmartAssignmentDeepEvalLLM``, backed
by this repo's own ``generate_text`` (see that module's docstring for why one
adapter covers every ``SMART_ASSIGNMENT_LLM_BACKEND``, including Sage-only).

Cost control: ``SMART_ASSIGNMENT_EVAL_IDS`` (see ``case_selection.py``) narrows
which captured cases get scored, the SAME knob ``test_eval.py``/``capture.py``
already read. ``SMART_ASSIGNMENT_EVAL_NUM_RUNS`` does **not** apply here --
same reasoning as ``capture.py``: nothing in this file re-runs the live agent,
only the judge call scores ALREADY-captured text.

Advisory, needs a live LLM backend + the ``eval-quality`` extra
(``pip install -e ".[dev,eval-quality]"``), NOT in the hermetic ``tests/`` suite
(``testpaths`` in ``pyproject.toml``). Each test skips cleanly (not a failure)
when no captured case is yet known to have the matching outcome.

Run with: pytest eval/test_quality.py
"""

from __future__ import annotations

import os

# Both must be set before `import deepeval` anywhere below -- deepeval reads
# them at import time. [VERIFIED against installed deepeval 2.6.6's
# deepeval/__init__.py]: TELEMETRY_OPT_OUT controls usage-analytics events;
# UPDATE_WARNING_OPT_OUT is a SEPARATE switch for an unrelated outbound HTTPS
# GET to pypi.org (a "newer version available" check, 5s timeout, silently
# swallowed on failure) that TELEMETRY_OPT_OUT does NOT cover. Both are set so
# importing this file never makes an unsolicited call to the public internet --
# relevant in a Sage-only environment where such egress may be blocked/audited.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from typing import List, Tuple

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # [VERIFIED against
# installed deepeval 2.6.6 -- newer DeepEval renamed this to SingleTurnParams,
# which does not exist at 2.6.6. See the pin's comment in pyproject.toml.]

from eval.capture import load_captured_results
from eval.case_selection import select_cases
from eval.deepeval_llm import SmartAssignmentDeepEvalLLM
from eval.golden_cases import GOLDEN_CASES, GoldenCase
from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_QUALITY_JUDGE

# Starting points, not calibrated -- deepeval's own GEval default (0.5) too.
# Revisit once there's a real distribution of scores across more captured cases.
_BRIEF_QUALITY_THRESHOLD = 0.5
_RESPONSE_CLARITY_THRESHOLD = 0.5

_JUDGE_MODEL = SmartAssignmentDeepEvalLLM(DEFAULT_CONFIG.for_role(ROLE_QUALITY_JUDGE))

_BRIEF_QUALITY = GEval(
    name="brief_quality",
    criteria=(
        "Judge whether ACTUAL_OUTPUT -- an escalation/handoff brief written for "
        "a human specialist reviewing a delivery-slot assignment the agent "
        "could not auto-assign for the customer described in INPUT -- is "
        "USEFUL: does it state the situation, the root cause / constraint that "
        "blocked auto-assignment, concrete remediation options, and a clear "
        "question or decision needed? Penalize a brief that is vague, generic, "
        "or missing any of these, such that a specialist could not act on it "
        "without asking follow-up questions."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=_JUDGE_MODEL,
    threshold=_BRIEF_QUALITY_THRESHOLD,
)

_RESPONSE_CLARITY = GEval(
    name="response_clarity",
    criteria=(
        "Judge whether ACTUAL_OUTPUT -- the agent's final message confirming a "
        "delivery-slot recommendation for the customer intake described in "
        "INPUT -- is CLEAR: is the recommended route/day/window unambiguous, is "
        "the reasoning easy to follow, and is the message free of internal "
        "scoring jargon (raw scores, factor weights, internal route/tier "
        "codes without explanation) that a customer would not understand? "
        "Penalize a response a customer would find confusing or would need to "
        "ask what it means."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=_JUDGE_MODEL,
    threshold=_RESPONSE_CLARITY_THRESHOLD,
)


def _cases_with_outcome(escalated: bool) -> List[Tuple[GoldenCase, str]]:
    """(case, final_response) for every SMART_ASSIGNMENT_EVAL_IDS-selected,
    captured case whose known outcome matches ``escalated`` exactly --
    unknown/legacy captures (outcome ``None``) are excluded, same discipline as
    ``test_response_match.py``'s ``_recommend_only_eval_ids``."""
    by_id = {case.eval_id: case for case in select_cases(GOLDEN_CASES)}
    return [
        (by_id[eval_id], result.final_response)
        for eval_id, result in load_captured_results().items()
        if result.escalated is escalated and eval_id in by_id
    ]


@pytest.mark.asyncio
async def test_brief_quality_on_escalate_cases():
    pairs = _cases_with_outcome(escalated=True)
    if not pairs:
        pytest.skip(
            "No captured case is yet known to be a clean escalate (brief_quality "
            "needs one). Capture one with `python3 -m eval.capture` (optionally "
            "scoped via SMART_ASSIGNMENT_EVAL_IDS to a case documented as "
            "'escalate' in golden_cases.py) and re-run."
        )

    failures = []
    for case, final_response in pairs:
        test_case = LLMTestCase(input=case.query, actual_output=final_response)
        await _BRIEF_QUALITY.a_measure(test_case)
        if _BRIEF_QUALITY.score < _BRIEF_QUALITY.threshold:
            failures.append(
                f"{case.eval_id}: {_BRIEF_QUALITY.score:.2f} < "
                f"{_BRIEF_QUALITY.threshold} -- {_BRIEF_QUALITY.reason}"
            )
    assert not failures, "brief_quality below threshold:\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_response_clarity_on_recommend_cases():
    pairs = _cases_with_outcome(escalated=False)
    if not pairs:
        pytest.skip(
            "No captured case is yet known to be a clean recommend "
            "(response_clarity needs one). Capture one with `python3 -m "
            "eval.capture` (optionally scoped via SMART_ASSIGNMENT_EVAL_IDS to "
            "a case documented as 'recommend' in golden_cases.py) and re-run."
        )

    failures = []
    for case, final_response in pairs:
        test_case = LLMTestCase(input=case.query, actual_output=final_response)
        await _RESPONSE_CLARITY.a_measure(test_case)
        if _RESPONSE_CLARITY.score < _RESPONSE_CLARITY.threshold:
            failures.append(
                f"{case.eval_id}: {_RESPONSE_CLARITY.score:.2f} < "
                f"{_RESPONSE_CLARITY.threshold} -- {_RESPONSE_CLARITY.reason}"
            )
    assert not failures, "response_clarity below threshold:\n" + "\n".join(failures)
