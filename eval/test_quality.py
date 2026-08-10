"""
Phase 3a: DeepEval G-Eval quality metrics -- reference-free rubrics scored
against the prose THIS run produced, NOT through ADK's
``AgentEvaluator``/``EvalSet`` machinery (unlike ``test_eval.py``/
``test_response_match.py``): DeepEval's ``GEval`` metric scores a bare
``(input, actual_output)`` pair directly, so there is no ADK dataset file to
render or scratch ``test_config.json`` to write.

The text comes from ``feedback_data/latest_run_responses.json``, harvested by
``eval/test_eval.py`` as it replays each case (see ``eval/capture_harvest.py``),
so a judge verdict always describes the commit under test. There is deliberately
NO fallback to the committed ``eval/data/golden_responses.json``: that file is
the approved *reference*, text a previous run produced and a human blessed, so
judging it answers "was the reference any good?" rather than the only question
these rubrics exist to ask. Missing harvest = skip locally, FAIL under CI (where
test_eval.py runs first, so its absence means the harvest broke).

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
which harvested cases get scored, the SAME knob ``test_eval.py``/``capture.py``
already read, and ``SMART_ASSIGNMENT_EVAL_CASES`` (see ``case_set.py``) selects
the case set. ``SMART_ASSIGNMENT_EVAL_NUM_RUNS`` does **not** apply here --
nothing in this file re-runs the live agent; the judge scores text the eval run
already produced.

Every verdict -- pass AND fail -- is recorded to the durable judge log (see
``eval/judge_log.py``), because only failures reach the assertion message: a
passing score would otherwise leave no trace of whether it scored 0.55 or 0.95.

Advisory, needs a live LLM backend + the ``eval-quality`` extra
(``pip install -e ".[dev,eval-quality]"``), NOT in the hermetic ``tests/`` suite
(``testpaths`` in ``pyproject.toml``). Each test skips cleanly (not a failure)
when this run produced no case with the matching outcome.

Run with (after a live eval run has harvested):

    pytest eval/test_eval.py      # replays the agent, harvests what it said
    pytest eval/test_quality.py   # judges that
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

import logging
from typing import Dict, List, Tuple

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # [VERIFIED against

# installed deepeval 2.6.6 -- newer DeepEval renamed this to SingleTurnParams,
# which does not exist at 2.6.6. See the pin's comment in pyproject.toml.]

from eval.capture import CaptureResult
from eval.capture_harvest import HARVEST_PATH, load_latest_run
from eval.case_selection import ci_active, select_cases
from eval.case_set import resolve_case_set
from eval.deepeval_llm import SmartAssignmentDeepEvalLLM
from eval.golden_cases import GoldenCase
from eval.judge_calibration import DIM_BRIEF_QUALITY, DIM_RESPONSE_CLARITY
from eval.judge_log import measure_and_record
from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_QUALITY_JUDGE

# Starting points, not calibrated -- deepeval's own GEval default (0.5) too.
# Revisit once there's a real distribution of scores across more captured cases.
_BRIEF_QUALITY_THRESHOLD = 0.5
_RESPONSE_CLARITY_THRESHOLD = 0.5

logger = logging.getLogger(__name__)

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


def _require_harvest() -> Dict[str, CaptureResult]:
    """This run's harvested responses, or stop the test.

    Absent means ``pytest eval/test_eval.py`` has not run in this working copy.
    Locally that is a skip with the command to fix it. **Under CI it is a
    failure**: the eval job runs test_eval.py first, so a missing harvest there
    means the harvest broke, and judging anything else would report green over
    prose this commit never produced. Same local-vs-CI asymmetry
    ``eval/case_selection.py`` applies to SMART_ASSIGNMENT_EVAL_IDS.

    There is deliberately no fallback to ``eval/data/golden_responses.json``.
    That file is the approved *reference* -- text a previous run produced and a
    human blessed -- so judging it answers "was the reference any good?", never
    "is this commit's prose any good?", which is the only question these rubrics
    are here to ask.
    """
    harvested = load_latest_run()
    if harvested:
        return harvested

    message = (
        f"No harvested responses at {HARVEST_PATH}. These judges score the prose THIS "
        "run produced, so run the live eval first:  pytest eval/test_eval.py"
    )
    if ci_active():
        pytest.fail(message + "  (CI runs it first, so an empty harvest means it broke.)")
    pytest.skip(message)


def _announce(harvested: Dict[str, CaptureResult]) -> None:
    """Say what is being judged and where it came from.

    A judge score is meaningless without knowing which agent produced the text;
    the file is regenerated every run and never committed, so this line is the
    only place a reader sees it. Grouped by provenance because a merged file can
    legitimately span runs."""
    by_provenance: Dict[str, List[str]] = {}
    for eval_id, result in sorted(harvested.items()):
        provenance = result.captured_with or {}
        dataset = (provenance.get("dataset") or {}).get("name", "?")
        key = (
            f"captured {result.captured_at or '?'} · dataset {dataset} · "
            f"{provenance.get('backend', '?')}/{provenance.get('model', '?')}"
        )
        by_provenance.setdefault(key, []).append(eval_id)
    print(f"\n[quality] judging {len(harvested)} harvested response(s) from {HARVEST_PATH}")
    for key, eval_ids in sorted(by_provenance.items()):
        print(f"[quality]   {key}: {', '.join(eval_ids)}")


def _cases_with_outcome(escalated: bool) -> List[Tuple[GoldenCase, CaptureResult]]:
    """(case, harvested result) for every SMART_ASSIGNMENT_EVAL_IDS-selected case
    whose outcome matches ``escalated`` exactly.

    A harvested id absent from the active case set is *logged*, not silently
    dropped: with SMART_ASSIGNMENT_EVAL_CASES able to point at a curated file
    (eval/case_set.py), a mismatch between what ran and what is being judged is a
    real possibility and must not look like "nothing to score"."""
    harvested = _require_harvest()
    _announce(harvested)
    by_id = {case.eval_id: case for case in select_cases(resolve_case_set().cases)}

    unknown = sorted(set(harvested) - set(by_id))
    if unknown:
        logger.warning(
            "Ignoring %d harvested response(s) not in the active case set: %s",
            len(unknown),
            ", ".join(unknown),
        )
    return [
        (by_id[eval_id], result)
        for eval_id, result in harvested.items()
        if result.escalated is escalated and eval_id in by_id
    ]


async def _score(metric, dimension: str, pairs) -> List[str]:
    """Judge each pair, recording every verdict, and return the failure lines.

    ``run=`` is the provenance of the JUDGED TEXT, taken off the record itself
    rather than left to default to the current process. That default was wrong:
    it stamped every verdict with today's environment even when the text came
    from a different model, which is exactly the attribution
    ``eval/judge_log.py`` separates ``judge`` from ``run`` to preserve -- did the
    score move because the agent changed, or the judge?"""
    failures: List[str] = []
    for case, result in pairs:
        test_case = LLMTestCase(input=case.query, actual_output=result.final_response)
        record = await measure_and_record(
            metric,
            test_case,
            eval_id=case.eval_id,
            dimension=dimension,
            decision_id=result.decision_id or case.decision_id,
            run=result.captured_with or None,
        )
        if not record.passed:
            failures.append(record.failure_line())
    return failures


@pytest.mark.asyncio
async def test_brief_quality_on_escalate_cases():
    pairs = _cases_with_outcome(escalated=True)
    if not pairs:
        pytest.skip(
            "This run produced no escalate-outcome response (brief_quality needs "
            "one). Run `pytest eval/test_eval.py` over a case that escalates -- "
            "see expected_outcome in golden_cases.py -- and re-run."
        )

    failures = await _score(_BRIEF_QUALITY, DIM_BRIEF_QUALITY, pairs)
    assert not failures, "brief_quality below threshold:\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_response_clarity_on_recommend_cases():
    pairs = _cases_with_outcome(escalated=False)
    if not pairs:
        pytest.skip(
            "This run produced no recommend-outcome response (response_clarity "
            "needs one). Run `pytest eval/test_eval.py` over a case that "
            "recommends -- see expected_outcome in golden_cases.py -- and re-run."
        )

    failures = await _score(_RESPONSE_CLARITY, DIM_RESPONSE_CLARITY, pairs)
    assert not failures, "response_clarity below threshold:\n" + "\n".join(failures)
