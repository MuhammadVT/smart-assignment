"""
Phase 3b: does routeslot/'s grounded rationale actually follow from the
evidence it cites -- the semantic gap routeslot/verifier.py's deterministic
checks structurally cannot close (see that module's docstring): a citation
can be numerically correct yet attached to the wrong option, or true on its
own but not actually support the sentence the model wrote around it.
`judgment/verifier.py`'s docstring names the identical residual gap: "a
rationale can attach a *correct* number to the wrong noun ... those are
semantic, not arithmetic, gaps."

Why routeslot/, not judgment/: [VERIFIED against tools/slot_recommendation.py]
with SMART_ASSIGNMENT_USE_ROUTE_SLOT_SCORING=true (this repo's default),
judgment/'s GroundedJudge is bypassed entirely -- the grounded call actually
producing rationale text is routeslot/decide.py's _grounded_index, which
builds the decision_summary/primary_reasons/key_tradeoff/runner_up narrative
that becomes the customer-facing response 3a's response_clarity metric
already scores. Testing routeslot/ tests the code path actually running.
judgment/ shares the identical evidence/schema/verifier recipe and can get
the same treatment later.

Drives routeslot/decide.py's REAL `_grounded_index` (call -> parse ->
verify_choice -> one corrective retry) directly against each golden case's
real evaluations -- the same sequence that decides what ships to users, not a
reimplementation that could drift out of sync with it.

Reference-free, unlike test_quality.py's response-match-adjacent metrics:
G-Eval scores `RouteSlotChoice.prose_fields()` (every free-text field the
deterministic verifier already scans) against `RouteSlotPacket.as_dict()`
(the exact evidence JSON the model itself was given) as CONTEXT. No captured
reference is needed -- the evidence packet is always available fresh,
deterministically, from a golden case's fixture -- so unlike test_quality.py
there is no capture step and nothing under eval/data/ is ever touched.

Skips CASES individually, not the whole file: some golden cases have no
route-slot clearing `route_slot_score_threshold` at all (a deterministic
`_escalate_low_score`, no grounded call happens there -- nothing to score).
The whole test only skips if NO case produced a grounded choice to score.

SMART_ASSIGNMENT_EVAL_IDS (case_selection.py) narrows which golden cases run,
same knob every other eval/test_*.py reads. SMART_ASSIGNMENT_EVAL_NUM_RUNS
does not apply -- one live call per case, no resampling here (routeslot's own
resampling only kicks in on its grounded-ESCALATION path, which is off by
default -- see Config.use_grounded_route_slot_escalation).

Advisory, needs a live LLM backend + the `eval-quality` extra, NOT in the
hermetic `tests/` suite.

Run with: pytest eval/test_rationale_faithfulness.py
"""

from __future__ import annotations

import json
import os

# Must be set before any `deepeval` import -- see eval/deepeval_llm.py's
# module docstring for why pytest's plugin autoload makes in-code setdefault
# insufficient on its own; this at least covers a non-pytest import.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from typing import List, Optional, Tuple  # noqa: E402

import pytest  # noqa: E402
from deepeval.metrics import GEval  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402

from eval.case_selection import select_cases  # noqa: E402
from eval.deepeval_llm import SmartAssignmentDeepEvalLLM  # noqa: E402
from eval.golden_cases import GOLDEN_CASES, GoldenCase  # noqa: E402
from smart_assignment.integrations.geocoding_client import resolve_geocoder  # noqa: E402
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes  # noqa: E402
from smart_assignment.pipeline import evaluate_candidates, geo_lookup  # noqa: E402
from smart_assignment.routeslot.decide import _all_route_slots, _grounded_index  # noqa: E402
from smart_assignment.routeslot.evidence import build_route_slot_packet  # noqa: E402
from smart_assignment.routeslot.schema import RouteSlotChoice  # noqa: E402
from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_QUALITY_JUDGE  # noqa: E402

# Starting point, not calibrated -- deepeval's own GEval default, same as 3a.
_RATIONALE_FAITHFULNESS_THRESHOLD = 0.5

_JUDGE_MODEL = SmartAssignmentDeepEvalLLM(DEFAULT_CONFIG.for_role(ROLE_QUALITY_JUDGE))

_FAITHFULNESS = GEval(
    name="rationale_faithfulness",
    criteria=(
        "Judge whether ACTUAL_OUTPUT -- the reasoned explanation for a "
        "delivery route-slot pick -- is FAITHFUL to CONTEXT (the evidence "
        "packet of real facts about each candidate option). Unfaithful if: "
        "(a) a stated number, day, or time is not accurate for the option it "
        "is attributed to, even if that same number appears correctly "
        "elsewhere in CONTEXT for a DIFFERENT option -- attaching a real "
        "figure to the wrong option is a failure; (b) a comparison or "
        "conclusion contradicts what CONTEXT actually shows; (c) the stated "
        "reasoning does not logically support the chosen option over its "
        "alternatives. Do not penalize reasonable paraphrasing, rounding, or "
        "omitted facts -- only claims that misrepresent or misattribute what "
        "CONTEXT actually says."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    model=_JUDGE_MODEL,
    threshold=_RATIONALE_FAITHFULNESS_THRESHOLD,
)


def _grounded_choice_for(case: GoldenCase) -> Optional[Tuple[RouteSlotChoice, dict]]:
    """Reproduce a golden case's evaluations via the real pipeline (the same
    geo_lookup + evaluate_candidates tools/slot_recommendation.py's
    _find_candidates already wires), build the packet the same way
    routeslot.decide._threshold_decide does (the active path under this
    repo's default Config.use_grounded_route_slot_escalation=False), and
    drive the real _grounded_index.

    Returns None when nothing clears route_slot_score_threshold (a
    deterministic escalation -- the grounded pick never runs) or the LLM
    call/verification never produced a usable choice (mechanical failure,
    already logged by _grounded_index itself)."""
    candidates = geo_lookup(
        case.customer, fetch_candidate_routes(), resolve_geocoder(), DEFAULT_CONFIG
    )
    evaluations = evaluate_candidates(case.customer, candidates, DEFAULT_CONFIG)

    all_pairs = _all_route_slots(evaluations)
    threshold = DEFAULT_CONFIG.route_slot_score_threshold
    eligible = [p for p in all_pairs if p.scored.total_score >= threshold]
    if not eligible:
        return None

    packet = build_route_slot_packet(
        case.customer, evaluations, DEFAULT_CONFIG, min_score=threshold
    )
    _index, choice, _reason = _grounded_index(packet, DEFAULT_CONFIG, choice_fn=None)
    if choice is None:
        return None
    return choice, packet.as_dict()


@pytest.mark.asyncio
async def test_rationale_faithfulness_on_grounded_picks():
    cases: List[GoldenCase] = select_cases(GOLDEN_CASES)

    scored_any = False
    failures = []
    for case in cases:
        result = _grounded_choice_for(case)
        if result is None:
            continue  # nothing above threshold here, or the grounded call didn't verify
        choice, packet_dict = result
        scored_any = True

        test_case = LLMTestCase(
            input=case.query,
            actual_output=" ".join(choice.prose_fields()),
            context=[json.dumps(packet_dict, sort_keys=True)],
        )
        await _FAITHFULNESS.a_measure(test_case)
        if _FAITHFULNESS.score < _FAITHFULNESS.threshold:
            failures.append(
                f"{case.eval_id}: {_FAITHFULNESS.score:.2f} < "
                f"{_FAITHFULNESS.threshold} -- {_FAITHFULNESS.reason}"
            )

    if not scored_any:
        pytest.skip(
            "No golden case produced a grounded route-slot choice to score "
            "(none cleared route_slot_score_threshold, or the LLM call/verification "
            "failed for all of them). Check the LLM backend/credentials, or that "
            "SMART_ASSIGNMENT_EVAL_IDS (if set) names a case with a feasible route."
        )
    assert not failures, "rationale_faithfulness below threshold:\n" + "\n".join(failures)
